import os
import torch
import numpy 
import itertools



import pickle 


cayleys = []

directory = os.path.abspath("/home/fennecs/Documents/CayleyML")
for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            f=open(file, 'r')
            csv=numpy.genfromtxt(file, delimiter=",")
            f.close()
            cayleys.append(torch.from_numpy(csv))


normalised = list(map(torch.nn.functional.normalize,cayleys))
perms = list(itertools.permutations(range(8)))
colperms = itertools.chain.from_iterable(map(lambda n: map(lambda p: n[p,:], perms), normalised))



f = open("latins.p","rb")
latins = map(lambda x: torch.nn.functional.normalize(torch.tensor(x)),pickle.load(f))
f.close 

print(list(latins)[1])
print(list(colperms)[1])