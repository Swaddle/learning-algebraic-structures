import os 
import torch
import pickle
import numpy 
import itertools


def transPerm(x : torch.Tensor, perm : list) -> torch.Tensor:
    x=x[perm,:]
    x=x[:,perm]
    return x

def genCayleys() -> list[(torch.Tensor,torch.Tensor)]:
    cayleys = []
    directory = os.path.abspath("/home/fennecs/Documents/CayleyML")
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                f=open(file, 'r')
                csv=numpy.genfromtxt(file, delimiter=",")
                f.close()
                cayleys.append(torch.from_numpy(csv).type(torch.float32))

    normalised = list(map(torch.nn.functional.normalize,cayleys))
    perms = list(itertools.permutations(range(8)))
    cayleyPerms = itertools.chain.from_iterable(map(lambda n: map(lambda p: transPerm(n,p), perms), normalised))
    
    return list(cayleyPerms)


out = genCayleys()
f=open("cayleys.p", "wb")
pickle.dump(genCayleys(),f)
f.close()

