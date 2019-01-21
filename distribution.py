from pixyz.distributions import Normal, Bernoulli
import torch
from torch import nn
from torch.nn import functional as F


class Filtering(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Filtering, self).__init__(cond_var=["b"], var=["z"])
        self.fc1 = nn.Linear(b_size, h_size)
        self.fc21 = nn.Linear(h_size, z_size)
        self.fc22 = nn.Linear(h_size, z_size)
        
    def forward(self,b):
        h = F.relu(self.fc1(b))
        mu = F.relu(self.fc21(h))
        logvar = F.relu(self.fc22(h))
        std = torch.exp(0.5*logvar)
        return {"loc": mu, "scale": std}

    
class Inference(Normal):
    def __init__(self, b_size=50, h_size=50, z_size=8):
        super(Inference, self).__init__(cond_var=["z_t2", "b_t1", "b_t2"], var=["z_t1"])
        self.fc1 = nn.Linear(z_size+b_size*2, h_size)
        self.fc21 = nn.Linear(h_size, z_size)
        self.fc22 = nn.Linear(h_size, z_size)
    
    def forward(self, z_t2, b_t1, b_t2):
        h = F.relu(self.fc1(torch.cat((z_t2, b_t1, b_t2), dim=1)))
        mu = F.relu(self.fc21(h))
        logvar = F.relu(self.fc22(h))
        std = torch.exp(0.5*logvar)
        return {"loc": mu, "scale": std}


class Transition(Normal):
    def __init__(self, h_size=50, z_size=8):
        super(Transition, self).__init__(cond_var=["z_t1"], var=["z_t2"])
        self.fc1 = nn.Linear(z_size, h_size)
        self.fc21 = nn.Linear(h_size, z_size)
        self.fc22 = nn.Linear(h_size, z_size)
        
    def forward(self, z_t1):
        h = F.relu(self.fc1(z_t1))
        mu = F.relu(self.fc21(h))
        logvar = F.relu(self.fc22(h))
        std = torch.exp(0.5*logvar)
        return {"loc": mu, "scale": std}

    
class Decoder(Bernoulli):
    def __init__(self, h_size=50, x_size=1*64*64, z_size=8):
        super(Decoder, self).__init__(cond_var=["z_t2"], var=["x_t2"])
        self.fc1 = nn.Linear(z_size, h_size)
        self.fc2 = nn.Linear(h_size, x_size)

    def forward(self, z_t2):
        h = F.relu(self.fc1(z_t2))
        logit = torch.sigmoid(self.fc2(h))
        return {"probs": logit}
