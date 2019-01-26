from pixyz.losses import NLL, KullbackLeibler
import torch
from torch import nn

from distribution import *


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
    def __init__(self, seq_len=16, b_size=50, x_size=1*28*28, processed_x_size=1*28*28, c_size=50, z_size=8):
        super(TDVAE, self).__init__()
        
        self.b_size = b_size
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.c_size = c_size
        self.z_size = z_size
        
        self.belief_state_net = BeliefStateNet(self.x_size, self.processed_x_size, self.b_size)
        
        # distributions
        self.p_B_t1_2 = Belief2(b_size=self.b_size, z_size=self.z_size)
        self.p_B_t1_1 = Belief1(b_size=self.b_size, z_size=self.z_size)
        self.p_B_t2_2 = self.p_B_t1_2.replace_var(b_t1="b_t2", z_t1_2="z_t2_2")
        self.p_B_t2_1 = self.p_B_t1_1.replace_var(b_t1="b_t2", z_t1_2="z_t2_2", z_t1_1="z_t2_1")
        self.q_2 = Smoothing2(b_size=self.b_size, z_size=self.z_size)
        self.q_1 = Smoothing1(b_size=self.b_size, z_size=self.z_size)
        self.p_T_2 = Transition2(z_size=self.z_size)
        self.p_T_1 = Transition1(z_size=self.z_size)
        self.p_D = Decoder(x_size=self.x_size, z_size=self.z_size)
        
        # losses
        self.kl = KullbackLeibler(self.q_1, self.p_B_t1_1) + KullbackLeibler(self.q_2, self.p_B_t1_2)
        self.entropy = - (NLL(self.p_B_t2_1) + NLL(self.p_B_t2_2))
        self.ce = NLL(self.p_T_1) + NLL(self.p_T_2)
        self.nll = NLL(self.p_D)
        self.loss_cls = self.kl+self.entropy+self.ce+self.nll

    def forward(self, batch):
        batch_size, seq_len, *_ = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.belief_state_net(batch)
        
        loss = 0    
        # sample every timestep for simplification
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2_2 = self.p_B_t2_2.sample({"b_t2": belief_states[:,t+1]}, reparam=True)["z_t2_2"]
            z_t2_1 = self.p_B_t2_1.sample({"b_t2": belief_states[:,t+1], 
                                           "z_t2_2": z_t2_2}, reparam=True)["z_t2_2"]
            z_t1_2 = self.q_2.sample({"b_t1": belief_states[:,t], 
                                      "b_t2": belief_states[:,t+1], 
                                      "z_t2_1": z_t2_1, 
                                      "z_t2_2": z_t2_2}, reparam=True)["z_t1_2"]
            z_t1_1 = self.q_1.sample({"b_t1": belief_states[:,t], 
                                      "b_t2": belief_states[:,t+1], 
                                      "z_t2_1": z_t2_1, 
                                      "z_t2_2": z_t2_2,
                                      "z_t1_2": z_t1_2}, reparam=True)["z_t1_1"]
            loss += self.loss_cls.estimate({"x_t2": x_t2,
                                            "z_t1_1": z_t1_1,
                                            "z_t1_2": z_t1_2,
                                            "z_t2_1": z_t2_1,
                                            "z_t2_2": z_t2_2,
                                            "b_t1": belief_states[:,t],
                                            "b_t2": belief_states[:,t+1]})
        return loss
    
    def test(self, batch):
        batch_size, seq_len, C, H, W = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = self.belief_state_net(batch)
        
        # evaluate each loss
        kl, entropy, ce, nll = 0, 0, 0, 0
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2_2 = self.p_B_t2_2.sample({"b_t2": belief_states[:,t+1]})["z_t2_2"]
            z_t2_1 = self.p_B_t2_1.sample({"b_t2": belief_states[:,t+1], 
                                           "z_t2_2": z_t2_2})["z_t2_2"]
            z_t1_2 = self.q_2.sample({"b_t1": belief_states[:,t], 
                                      "b_t2": belief_states[:,t+1], 
                                      "z_t2_1": z_t2_1, 
                                      "z_t2_2": z_t2_2})["z_t1_2"]
            z_t1_1 = self.q_1.sample({"b_t1": belief_states[:,t], 
                                      "b_t2": belief_states[:,t+1], 
                                      "z_t2_1": z_t2_1, 
                                      "z_t2_2": z_t2_2,
                                      "z_t1_2": z_t1_2})["z_t1_1"]
            kl += self.kl.estimate({"b_t1": belief_states[:,t],
                                    "b_t2": belief_states[:,t+1],
                                    "z_t2_1": z_t2_1,
                                    "z_t2_2": z_t2_2,
                                    "z_t1_2": z_t1_2})
            entropy += self.entropy.estimate({"z_t2_1": z_t2_1,
                                              "z_t2_2": z_t2_2,
                                              "b_t2": belief_states[:,t+1]}).mean()
            ce += self.ce.estimate({"z_t2_1": z_t2_1,
                                    "z_t2_2": z_t2_2,
                                    "z_t1_1": z_t1_1,
                                    "z_t1_2": z_t1_2})
            nll += self.nll.estimate({"x_t2": x_t2,
                                      "z_t2_1": z_t2_1,
                                      "z_t2_2": z_t2_2})
        # prediction
        test_pred = batch.clone()
        for t in range(seq_len-1):
            z_t1_2 = self.p_B_t1_2.sample({"b_t1": belief_states[:,t]})["z_t1_2"]
            z_t1_1 = self.p_B_t1_1.sample({"b_t1": belief_states[:,t], "z_t1_2": z_t1_2})["z_t1_1"]
            z_t2_2 = self.p_T_2.sample({"z_t1_1": z_t1_1, "z_t1_2": z_t1_2})["z_t2_2"]
            z_t2_1 = self.p_T_1.sample({"z_t1_1": z_t1_1, "z_t1_2": z_t1_2, "z_t2_2": z_t2_2})["z_t2_1"]
            x_t2_hat = self.p_D.sample_mean({"z_t2_1": z_t2_1, "z_t2_2": z_t2_2}) # batch_size*C*H*W
            test_pred[:,t+1] = x_t2_hat
        test_pred = torch.clamp(test_pred.view(batch_size, seq_len, C, H, W), 0, 1)
        
        return kl, entropy, ce, nll, test_pred
