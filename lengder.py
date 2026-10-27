"""
lengder.py - elementlengder
===========================

Lengden av et element er avstanden mellom de to knutepunktene det kobler
sammen (Pythagoras). Elementlengden inngår i elementstivheten (EI/L), i
fastinnspenningsmomentene og i moment- og skjærkraftberegningen.
"""

import numpy as np

from lesinput import P_X, P_Y, E_N1, E_N2


def lengder(punkt, elem, nelem):
    # Returnerer en vektor med lengden [m] til hvert av de nelem elementene
    elemlen = np.zeros(nelem)
    for ielem in range(nelem):
        n1, n2 = elem[ielem, E_N1], elem[ielem, E_N2]   # knutepunktene til elementet
        dx = punkt[n2, P_X] - punkt[n1, P_X]
        dy = punkt[n2, P_Y] - punkt[n1, P_Y]
        elemlen[ielem] = np.sqrt(dx**2 + dy**2)
        if elemlen[ielem] == 0:
            raise ValueError(f"Element {ielem} har lengde null (samme start- og sluttpunkt)")
    return elemlen
