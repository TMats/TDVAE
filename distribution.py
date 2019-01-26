from pixyz.distributions import Normal, Bernoulli
import torch
from torch import nn
from torch.nn import functional as F

class Distribution(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Distribution, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2*output_size)
        
    def forward(self, input):
        h = torch.tanh(self.fc1(input))
        h = h * torch.sigmoid(self.fc2(input))
        mu, logsigma = torch.split(self.fc3(h), self.output_size, dim=-1)
        return mu, logsigma

    
class Belief2(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Belief2, self).__init__(cond_var=["b_t1"], var=["z_t1_2"])
        self.D = Distribution(b_size, h_size, z_size)
        
    def forward(self,b_t1):
        mu, logsigma = self.D(b_t1)
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}
    
class Belief1(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Belief1, self).__init__(cond_var=["b_t1", "z_t1_2"], var=["z_t1_1"])
        self.D = Distribution(b_size+z_size, h_size, z_size)
        
    def forward(self,b_t1, z_t1_2):
        mu, logsigma = self.D(torch.cat((b_t1, z_t1_2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}

    
class Smoothing2(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Smoothing2, self).__init__(cond_var=["b_t1", "b_t2", "z_t2_1", "z_t2_2"], var=["z_t1_2"])
        self.D = Distribution(2*b_size+2*z_size, h_size, z_size)
    
    def forward(self, b_t1, b_t2, z_t2_1, z_t2_2):
        mu, logsigma = self.D(torch.cat((b_t1, b_t2, z_t2_1, z_t2_2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}
    
class Smoothing1(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Smoothing1, self).__init__(cond_var=["b_t1", "b_t2", "z_t2_1", "z_t2_2", "z_t1_2"], var=["z_t1_1"])
        self.D = Distribution(2*b_size+3*z_size, h_size, z_size)
    
    def forward(self, b_t1, b_t2, z_t2_1, z_t2_2, z_t1_2):
        mu, logsigma = self.D(torch.cat((b_t1, b_t2, z_t2_1, z_t2_2, z_t1_2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}


class Transition2(Normal):
    def __init__(self, h_size=50, z_size=8):
        super(Transition2, self).__init__(cond_var=["z_t1_1", "z_t1_2"], var=["z_t2_2"])
        self.D = Distribution(2*z_size, h_size, z_size)
        
    def forward(self, z_t1_1, z_t1_2):
        mu, logsigma = self.D(torch.cat((z_t1_1, z_t1_2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}
    
class Transition1(Normal):
    def __init__(self, h_size=50, z_size=8):
        super(Transition1, self).__init__(cond_var=["z_t1_1", "z_t1_2", "z_t2_2"], var=["z_t2_1"])
        self.D = Distribution(3*z_size, h_size, z_size)
        
    def forward(self, z_t1_1, z_t1_2, z_t2_2):
        mu, logsigma = self.D(torch.cat((z_t1_1, z_t1_2, z_t2_2), dim=1))
        std = torch.exp(logsigma)
        return {"loc": mu, "scale": std}

    
class Decoder(Bernoulli):
    def __init__(self, h_size=50, x_size=1*64*64, z_size=8):
        super(Decoder, self).__init__(cond_var=["z_t2_1", "z_t2_2"], var=["x_t2"])
        self.fc1 = nn.Linear(2*z_size, h_size)
        self.fc2 = nn.Linear(h_size, x_size)

    def forward(self, z_t2_1, z_t2_2):
        h = F.relu(self.fc1(torch.cat((z_t2_1, z_t2_2), dim=1)))
        logit = torch.sigmoid(self.fc2(h))
        return {"probs": logit}
    
