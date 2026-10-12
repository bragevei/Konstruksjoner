import numpy as np

from lengder import lengder
from beregne_boyestivhet import beregne_boyestivhet

def elementstivhetsmatrise(npunkt, punkt, nelem, elemkonn, tvsnitt, EI): #Allerede faktorisert med 4, trenger bare faktorisere inn EI/L
    dim = npunkt
    elemlen = lengder(punkt, elemkonn)
    K = np.zeros((dim, dim), dtype=float) #Matrise med nuller med dimensjon lik antall knutepunkt
    EI = np.asarray(EI, dtype=float)
    if len(EI) != nelem:
        raise ValueError("EI må inneholde én verdi per element.")

    for ielem, (i, j) in enumerate(elemkonn):
        K[i, i] += 4*(EI[ielem]/elemlen[ielem]) #k11 i element i
        K[i, j] += 2*(EI[ielem]/elemlen[ielem])  #k12 i element i
        K[j, i] += 2*(EI[ielem]/elemlen[ielem]) #k21 i element i
        K[j, j] += 4*(EI[ielem]/elemlen[ielem])  #k22 i element i
    return K
