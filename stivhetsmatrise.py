import numpy as np

from lesinput import lesinput
from lengder import lengder
from beregne_boyestivhet import beregne_boyestivhet

npunkt, punkt, nelem, elemkonn, tvsnitt = lesinput()
#elemlen = lengder(punkt, elemkonn)
elemlen = [4, 2, 4, 2, 4, 2, 4, 2, 4, 2]

dim = npunkt

K = np.zeros((dim, dim), dtype=float) #Matrise med nuller med dimensjon lik antall knutepunkt

def elementstivhetsmatrise(K, elemkonn): #Allerede faktorisert med 4, trenger bare faktorisere inn EI/L
    for i, j in elemkonn:
        K[i, i] += 4*(EI[i]/elemlen[i]) #k11 i element i
        K[i, j] += 2*(EI[i]/elemlen[i])  #k12 i element i 
        K[j, i] += 2*(EI[i]/elemlen[i]) #k21 i element i
        K[j, j] += 4*(EI[i]/elemlen[i])  #k22 i element i 
    return K

K = elementstivhetsmatrise(K, elemkonn)
print(K)
print(elemlen)
print(K.dtype)