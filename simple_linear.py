import torch
import torch.nn as nn 
import torch.nn.functional as F

depth = 2

class SimpleLinear64(nn.Module):
    def __init__(self):
        super(SimpleLinear64, self).__init__()  

        self.lin = nn.Linear(64,128)
       
        self.lblock = nn.ModuleList()         
        self.lblock.append(nn.Linear(128,256))
        self.lblock.append(nn.Linear(256,128))
        self.lblock.append(nn.Linear(128,64))
        self.lblock.append(nn.Linear(64,32))
        self.lblock.append(nn.Linear(32,16))
        self.lblock.append(nn.Linear(16,8))
        self.lblock.append(nn.Linear(8,4))

        self.lout = nn.Linear(4,2)

    def forward(self, x):
        x = F.relu(self.lin(x))

        for j in range(7): 
            x = F.relu(self.lblock[j](x))
        
        x = self.lout(x)
        
        return x