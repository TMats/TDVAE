from pixyz.losses import NLL, KullbackLeibler
import torch
from torch import nn

from distribution import Filtering, Transition, Inference, Decoder

class TDVAE(nn.Module):
    def __init__(self, batch_size=16, seq_len=16, b_size=50, x_size=1*64*64, c_size=50):
        super(TDVAE, self).__init__()
        
        self.b_size = b_size
        self.c_size = c_size
        
        self.belief_state_net = nn.LSTMCell(x_size, b_size)
        self.p_b = Filtering()
        self.p_t = Transition()
        self.q = Inference()
        self.p_d = Decoder()
    
    def forward(self, batch):
        batch_size, seq_len, *_ = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = []
        # initialize
        b = batch.new_zeros((batch_size, self.b_size))
        c = batch.new_zeros((batch_size, self.c_size))
        
        for t in range(seq_len):
            x = batch[:,t]
            b, c = self.belief_state_net(x, (b,c))
            belief_states.append(b)
        loss = 0
        
        # sample every timestep for simplification
        for t in range(seq_len-1):
            x_t2 = batch[:,t+1]
            z_t2 = self.p_b.sample({"b": belief_states[t+1]}, reparam=True)["z"]
            z_t1 = self.q.sample({"z_t2": z_t2, "b_t1": belief_states[t], "b_t2": belief_states[t+1]}, reparam=True)["z_t1"]
            kl = KullbackLeibler(self.q, self.p_b).estimate({"z_t2": z_t2,
                                                             "b_t1": belief_states[t],
                                                             "b_t2": belief_states[t+1],
                                                             "b": belief_states[t]})
            b_ll = -NLL(self.p_b).estimate({"z": z_t2, "b": belief_states[t+1]})
            t_nll = NLL(self.p_t).estimate({"z_t2": z_t2, "z_t1": z_t1})
            d_nll = NLL(self.p_d).estimate({"x_t2": x_t2, "z_t2": z_t2})
            loss += (kl+b_ll+t_nll+d_nll)
            
        return loss
    
    def generate(self, batch):
        batch_size, seq_len, C, H, W = batch.size()
        batch = batch.view(batch_size, seq_len, -1)
        belief_states = []
        
        # initialize
        b = batch.new_zeros((batch_size, self.b_size))
        c = batch.new_zeros((batch_size, self.c_size))
        
        for t in range(seq_len):
            x = batch[:,t]
            b, c = self.belief_state_net(x, (b,c))
            belief_states.append(b)
        
        test_reconst = batch.clone()
        for t in range(seq_len-1):
            z_t1 = self.p_b.sample({"b": belief_states[t]}, reparam=True)["z"]
            z_t2 = self.p_t.sample({"z_t1": z_t1}, reparam=True)["z_t2"]
            x_t2_hat = self.p_d.sample_mean({"z_t2": z_t2}) # batch_size*C*H*W
            test_reconst[:,t+1] = x_t2_hat
        return test_reconst.view(batch_size, seq_len, C, H, W)