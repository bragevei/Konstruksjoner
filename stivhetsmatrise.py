import numpy as np

from lengder import lengder

def elementstivhetsmatrise(npunkt, punkt, nelem, elemkonn, tvsnitt, EI): #Allerede faktorisert med 4, trenger bare faktorisere inn EI/L
    dim = npunkt
    elemlen = lengder(punkt, elemkonn)
    K = np.zeros((dim, dim), dtype=float) #Matrise med nuller med dimensjon lik antall knutepunkt
    for i, j in elemkonn:
        K[i, i] += 4*(EI[i]/elemlen[i]) #k11 i element i
        K[i, j] += 2*(EI[i]/elemlen[i])  #k12 i element i 
        K[j, i] += 2*(EI[i]/elemlen[i]) #k21 i element i
        K[j, j] += 4*(EI[i]/elemlen[i])  #k22 i element i 
    return K