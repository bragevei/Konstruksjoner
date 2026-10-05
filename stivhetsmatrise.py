import numpy as np

from lesinput import lesinput
npunkt, punkt, nelem, elemkonn, tvsnitt = lesinput()


dim = npunkt

K = np.zeros((dim, dim))
print(K)
print(nelem)
print(elemkonn)

def elementstivhetsmatrise(K, elemkonn):
    for i, j in elemkonn:
        K[i, i] += 4  # 00
        K[i, j] += 2  # 01
        K[j, i] += 2 # 10
        K[j, j] += 4  # 11
    return K

K = np.zeros((dim, dim), dtype=int)
K = elementstivhetsmatrise(K, elemkonn)
print(K)