from pixyz.losses import NLL, KullbackLeibler
import torch
from torch import nn

from distribution import Filtering, Transition, Inference, Decoder


class BeliefStateNet(nn.Module):
    def __init__(self, x_size, processed_x_size, b_size):
        super(BeliefStateNet, self).__init__()
        self.fc1 = nn.Linear(x_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)
        self.rnn = nn.LSTMCell(processed_x_size, b_size)

    def forward(self, x, states):
        b, c = states
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        b, c = self.rnn(h, (b, c))
        return b, c


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
        self.kl_1 = KullbackLeibler(self.q, self.p_b1)
        self.kl_2 = KullbackLeibler(self.p_b2, self.p_t)
        self.d_nll = NLL(self.p_d)
        self.loss_cls = (self.kl_1+self.kl_2+self.d_nll).mean()
    
    def rollout_belief(self, batch):
        batch_size, seq_len, *_ = batch.size()
        belief_states = []
        b = batch.new_zeros((batch_size, self.b_size))
        c = batch.new_zeros((batch_size, self.c_size))
        for t in range(seq_len):
            x = batch[:,t]
            b, c = self.belief_state_net(x, (b,c))
            belief_states.append(b)
        return belief_states
        
    
    def forward(self, batch):
        batch_size, seq_len, *_ = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.rollout_belief(batch)
        
        loss = 0    
        # sample every timestep for simplification
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2 = self.p_b2.sample({"b_t2": belief_states[t+1]}, reparam=True)["z_t2"]
            z_t1 = self.q.sample({"z_t2": z_t2, 
                                  "b_t1": belief_states[t], 
                                  "b_t2": belief_states[t+1]}, reparam=True)["z_t1"]
            loss += self.loss_cls.estimate({"x_t2": x_t2,
                                            "z_t1": z_t1,
                                            "z_t2": z_t2,
                                            "b_t1": belief_states[t],
                                            "b_t2": belief_states[t+1]})
        return loss
    
    def test(self, batch):
        batch_size, seq_len, C, H, W = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.rollout_belief(batch)
        
        # evaluate each loss
        kl_1, kl_2, d_nll = 0, 0, 0
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2 = self.p_b2.sample({"b_t2": belief_states[t+1]}, reparam=True)["z_t2"]
            z_t1 = self.q.sample({"z_t2": z_t2, 
                                  "b_t1": belief_states[t], 
                                  "b_t2": belief_states[t+1]}, reparam=True)["z_t1"]
            kl_1 += self.kl_1.estimate({"z_t2": z_t2,
                                        "b_t1": belief_states[t],
                                        "b_t2": belief_states[t+1]}).mean()
            kl_2 += self.kl_2.estimate({"z_t1": z_t1,
                                        "b_t2": belief_states[t+1]}).mean()
            d_nll += self.d_nll.estimate({"x_t2": x_t2,
                                          "z_t2": z_t2}).mean()
        # prediction
        test_pred = batch.clone()
        for t in range(seq_len-1):
            z_t1 = self.p_b1.sample({"b_t1": belief_states[t]})["z_t1"]
            z_t2 = self.p_t.sample({"z_t1": z_t1})["z_t2"]
            x_t2_hat = self.p_d.sample_mean({"z_t2": z_t2}) # batch_size*C*H*W
            test_pred[:,t+1] = x_t2_hat
        test_pred = torch.clamp(test_pred.view(batch_size, seq_len, C, H, W), 0, 1)
        
        return kl_1, kl_2, d_nll, test_pred
