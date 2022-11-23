
import pickle 
import itertools 
import numpy as np 

from sage.all import *
from sage.combinat.matrices.latin import *


def toList(mat): 
    return [[float(str(mat[r1,r2])) for r1 in range(8)] for r2 in range(8)]

latins = list(map(toList,itertools.islice(LatinSquare_generator(back_circulant(8)),10)))

f=open("latins.p", "wb")
pickle.dump(latins,f)
f.close()



