import torch 
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_size=3, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) 
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, 2)

        self.cls = nn.Linear(24*2, 2)
        
        
    # encode process
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc2(h)
    
    # generate random vector
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        torch.manual_seed(0)
        eps = torch.randn_like(std)
        return mu + eps * std

    # decode process
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)
    
    
    def forward(self, x):
        # print("x")
        # print(x.size())
        mu, log_var = self.encode(x)
        # print("mu")
        # print(mu.size())
        # print("log_var")
        # print(log_var.size())
        z = self.reparameterize(mu, log_var)
        # print("z")
        # print(z.size())
        x_reconst = self.decode(z)
        # print("x_reconst")
        # print(x_reconst.size())
        x_reconst = x_reconst.reshape(x_reconst.shape[0], -1)
        # print(x_reconst.size())
        out = F.softmax(self.cls(x_reconst), dim = 1)
        #out = F.softmax((x_reconst), dim = 1)
        
        return out
