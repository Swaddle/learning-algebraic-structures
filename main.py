import os
import numpy 
import itertools

import datetime
import time 

import typing 
import pickle 

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader

from simple_linear import SimpleLinear64 

def getData():
    f = open("cayleys.p","rb")
    cayleys = pickle.load(f)
    f.close()
    cayleyClass = itertools.repeat(torch.Tensor([1,0]), 201600)

    f = open("latins.p","rb")
    latins = map(lambda x: torch.nn.functional.normalize(torch.tensor(x)),pickle.load(f))
    f.close 
    latinClass = itertools.repeat(torch.Tensor([0,1]), 201600)

    latinData = [[torch.flatten(l),c] for l in latins for c in latinClass] 
    cayleyData = [[torch.flatten(p),c] for p in cayleys for c in cayleyClass]

    data = list(itertools.chain(latinData,cayleyData))
    return data


# train a simple model

model = SimpleLinear64()
data = getData()

length = len(data)
trainSize = int(0.8 * length)
valSize = length - trainSize

trainSplit, valSplit = torch.utils.data.random_split(data, [trainSize, valSize])

trainDataloader = DataLoader(trainSplit, batch_size=64, shuffle=True)
valDataloader = DataLoader(valSplit, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

startTime = time.time()

 # loop over the dataset 20 times
for epoch in range(20): 

    runningLoss = 0.0
    runningVal = 0.0

    for i, data in enumerate(trainDataloader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        valIn, valLabel = next(iter(valDataloader))
        valOut = model(valIn)
        valLoss = criterion(valOut, valLabel)

        # print statistics
        runningLoss += loss.item()
        runningVal += valLoss.item()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            
            totalTime = time.time() - startTime
            totalTimeStr = str(datetime.timedelta(seconds=int(totalTime))) 
            print(f'[{epoch + 1}, {i + 1:5d}] av loss: {runningLoss/2000.0} av val: {runningVal/2000.0} training time: {totalTimeStr}')
            runningLoss = 0.0
            runningVal = 0.0
    
torch.save(model.state_dict(), "model.pt")