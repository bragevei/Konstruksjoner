import numpy as np

from lesinput import lesinput
npunkt, punkt, nelem, elemkonn, tvsnitt = lesinput()


dim = npunkt

K = np.zeros((dim, dim)) #Matrise med nuller med dimensjon lik antall knutepunkt
#print(K)
#print(nelem)
#print(elemkonn)

def elementstivhetsmatrise(K, elemkonn): #Allerede faktorisert med 4, trenger bare faktorisere inn EI/L
    for i, j in elemkonn:
        K[i, i] += 4  #k11 i element i
        K[i, j] += 2  #k12 i element i 
        K[j, i] += 2 #k21 i element i
        K[j, j] += 4  #k22 i element i
    return K

K = np.zeros((dim, dim), dtype=int)
K = elementstivhetsmatrise(K, elemkonn)
print(K)