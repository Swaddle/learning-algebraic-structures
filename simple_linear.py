import torch
import torch.nn as nn 


class SimpleLinear64(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()  
        self.lin = nn.Linear(64,64)
        self.middle = nn.ModuleList() 
        for j in range(10): 
            self.middle.append(nn.Linear(64,64))
        self.lout = nn.Linear(64,2)

    def forward(self, x):
        x = nn.ReLU(self.lin(x))
        for j in range(10): 
            x = nn.ReLU(self.middle[i](x))
        x = self.lout(x)
        return x