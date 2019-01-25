from pixyz.losses import NLL, KullbackLeibler
import torch
from torch import nn

from distribution import Filtering, Transition, Inference, Decoder


class BeliefStateNet(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size):
        super(BeliefStateNet, self).__init__()
        self.fc1 = nn.Linear(x_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)
        self.lstm = nn.LSTM(input_size=processed_x_size, hidden_size=b_size, batch_first=True)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        b, *_ = self.lstm(h)
        return b


class TDVAE(nn.Module):
    def __init__(self, seq_len=16, b_size=50, x_size=1*64*64, processed_x_size=1*64*64, c_size=50, z_size=8):
        super(TDVAE, self).__init__()
        
        self.b_size = b_size
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.c_size = c_size
        self.z_size = z_size
        
        self.belief_state_net = BeliefStateNet(self.x_size, self.processed_x_size, self.b_size)
        
        # distributions
        self.p_b1 = Filtering(b_size=self.b_size, z_size=self.z_size)
        self.p_b2 = self.p_b1.replace_var(b_t1="b_t2", z_t1="z_t2")
        self.p_t = Transition(z_size=self.z_size)
        self.q = Inference(b_size=self.b_size, z_size=self.z_size)
        self.p_d = Decoder(x_size=self.x_size, z_size=self.z_size)
        
        # losses
        self.kl = KullbackLeibler(self.q, self.p_b1)
        self.b_ll = -NLL(self.p_b2)
        self.t_nll = NLL(self.p_t)
        self.d_nll = NLL(self.p_d)
        self.loss_cls = (self.kl+self.b_ll+self.t_nll+self.d_nll).mean()

    def forward(self, batch):
        batch_size, seq_len, *_ = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.belief_state_net(batch)
        
        loss = 0
        # sample every timestep for simplification
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2 = self.p_b2.sample({"b_t2": belief_states[:,t+1]}, reparam=True)["z_t2"]
            z_t1 = self.q.sample({"z_t2": z_t2, 
                                  "b_t1": belief_states[:,t], 
                                  "b_t2": belief_states[:,t+1]}, reparam=True)["z_t1"]
            loss += self.loss_cls.estimate({"x_t2": x_t2,
                                            "z_t1": z_t1,
                                            "z_t2": z_t2,
                                            "b_t1": belief_states[:,t],
                                            "b_t2": belief_states[:,t+1]})
        return loss
    
    def test(self, batch):
        batch_size, seq_len, C, H, W = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.belief_state_net(batch)
        
        # evaluate each loss
        kl, b_ll, t_nll, d_nll = 0, 0, 0, 0
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2 = self.p_b2.sample({"b_t2": belief_states[:,t+1]}, reparam=True)["z_t2"]
            z_t1 = self.q.sample({"z_t2": z_t2, 
                                  "b_t1": belief_states[:,t], 
                                  "b_t2": belief_states[:,t+1]}, reparam=True)["z_t1"]
            kl += self.kl.estimate({"z_t2": z_t2,
                                    "b_t1": belief_states[:,t],
                                    "b_t2": belief_states[:,t+1]}).mean()
            b_ll += self.b_ll.estimate({"z_t2": z_t2,
                                        "b_t2": belief_states[:,t+1]}).mean()
            t_nll += self.t_nll.estimate({"z_t1": z_t1,
                                          "z_t2": z_t2}).mean()
            d_nll += self.d_nll.estimate({"x_t2": x_t2,
                                          "z_t2": z_t2}).mean()
        # prediction
        test_pred = batch.clone()
        for t in range(seq_len-1):
            z_t1 = self.p_b1.sample({"b_t1": belief_states[:,t]})["z_t1"]
            z_t2 = self.p_t.sample({"z_t1": z_t1})["z_t2"]
            x_t2_hat = self.p_d.sample_mean({"z_t2": z_t2}) # batch_size*C*H*W
            test_pred[:,t+1] = x_t2_hat
        test_pred = torch.clamp(test_pred.view(batch_size, seq_len, C, H, W), 0, 1)
        
        return test_pred, kl, b_ll, t_nll, d_nll
