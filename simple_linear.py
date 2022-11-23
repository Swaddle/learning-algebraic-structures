import torch
import torch.nn as nn 
import torch.nn.functional as F

depth = 10

class SimpleLinear64(nn.Module):
    def __init__(self):
        super(SimpleLinear64, self).__init__()  
        self.lin = nn.Linear(64,64)
        self.middle = nn.ModuleList() 
        for j in range(depth): 
            self.middle.append(nn.Linear(64,64))
        self.lout = nn.Linear(64,2)

    def forward(self, x):
        x = F.relu(self.lin(x))
        for j in range(depth): 
            x = F.relu(self.middle[j](x))
        x = self.lout(x)
        return x