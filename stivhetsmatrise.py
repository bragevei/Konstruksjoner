"""
stivhetsmatrise.py - element- og systemstivhetsmatrise
======================================================

OVERORDNET TENKEMÅTE
--------------------
For en uforskyvelig ramme er endemomentene i et bjelkeelement gitt av
enderotasjonene r1 og r2 (positive med klokka):

    M1 = EI/L * (4*r1 + 2*r2) + M1_fim
    M2 = EI/L * (2*r1 + 4*r2) + M2_fim

Rotasjonsdelen skrives på matriseform som k * [r1, r2], der

    k = EI/L * [[4, 2],
                [2, 4]]

er elementstivhetsmatrisen. Systemstivhetsmatrisen K har én rad og én kolonne
per knutepunkt (én rotasjonsfrihetsgrad per knutepunkt). Den bygges ved å
legge hvert elements k inn på radene/kolonnene til de to knutepunktene
elementet kobler sammen. Hvilke knutepunkter det er, står i
konnektivitetstabellen elem. Der flere elementer møtes i samme knutepunkt,
summeres bidragene, slik at diagonalleddet K[i, i] blir summen av 4EI/L for
alle elementer som er koblet til knutepunkt i.
"""

import numpy as np

from lesinput import E_N1, E_N2


def elementstivhet(EI, L):
    # Elementstivhetsmatrisen k (2x2) for ett bjelkeelement
    return EI / L * np.array([[4.0, 2.0],
                              [2.0, 4.0]])


def stivhetsmatrise(npunkt, nelem, elem, elemlen, EI):
    # Setter sammen systemstivhetsmatrisen K (npunkt x npunkt)
    K = np.zeros((npunkt, npunkt))
    for ielem in range(nelem):
        k = elementstivhet(EI[ielem], elemlen[ielem])
        # Systemfrihetsgradene til elementets ende 1 og ende 2:
        frihetsgrader = [elem[ielem, E_N1], elem[ielem, E_N2]]
        # np.ix_ plukker ut 2x2-blokken med disse radene og kolonnene i K, og k adderes inn der 
        # (K[i,i] += k[0,0], K[i,j] += k[0,1], osv.)
        K[np.ix_(frihetsgrader, frihetsgrader)] += k
    return K
