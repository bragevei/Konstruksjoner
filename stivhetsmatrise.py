import numpy as np

from lengder import lengder

def elementstivhetsmatrise(npunkt, punkt, nelem, elemkonn, tvsnitt, EI):
    if int(npunkt) != len(punkt):
        raise ValueError("npunkt må stemme med antall knutepunkter")
    if int(nelem) != len(elemkonn):
        raise ValueError("nelem må stemme med antall elementer")
    elemkonn = np.asarray(elemkonn, dtype=int)
    dim = int(npunkt)
    elemlen = lengder(punkt, elemkonn)
    K = np.zeros((dim, dim), dtype=float)
    EI = np.asarray(EI, dtype=float)
    if EI.ndim != 1 or len(EI) != nelem:
        raise ValueError("EI må inneholde én verdi per element")
    if np.any(EI <= 0):
        raise ValueError("EI må være positiv for alle elementer.")
#Går gjennom elemkonn som og henter indeks i og j fra hver indeks i og j i elemkonn. Elemkonn blir tilsvarende konnektivitetstabell i koden vår.
#Deretter ganges den med faktor som er EI/L og plasseres i indeks riktig indeks i stivhetsmatrisen K
#Resultatet er stivhetsmatrise med EI/L faktorisert inni matrisen.
    for ielem, (i, j) in enumerate(elemkonn):
        faktor = EI[ielem] / elemlen[ielem]
        K[i, i] += 4 * faktor
        K[i, j] += 2 * faktor
        K[j, i] += 2 * faktor
        K[j, j] += 4 * faktor
    return K
