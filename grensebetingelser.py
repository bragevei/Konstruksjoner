"""
grensebetingelser.py - innføring av grensebetingelser
=====================================================

OVERORDNET TENKEMÅTE
--------------------
I et fast innspent knutepunkt er rotasjonen kjent og lik null. Uten
grensebetingelser kunne likningssystemet K*r = R gitt en rotasjon der.
Vi bruker metode i) fra oppgaveteksten:

  * Raden og kolonnen til den fastholdte frihetsgraden nulles ut i K.
  * Diagonalleddet settes lik 1 (et vilkårlig tall ulikt null).
  * Tilhørende element i lastvektoren R settes lik 0.

Da blir likningen for denne frihetsgraden 1 * r_i = 0, altså r_i = 0, og
siden kolonnen er nullet ut påvirker r_i ikke de andre likningene. Matrisen
forblir symmetrisk og ikke-singulær, og løsningen for de frie rotasjonene er
den samme som om raden og kolonnen hadde blitt fjernet.

Knutepunkter med rot = 1 i inputfilen (ledd og vanlige rammehjørner) har fri
rotasjon og endres ikke.
"""

from lesinput import P_ROT


def bc(npunkt, punkt, K, R):
    # Returnerer kopier av Kn og Rn der fastholdte rotasjoner er innført 
    Kn = K.copy()
    Rn = R.copy()
    for i in range(npunkt):
        if punkt[i, P_ROT] == 0:      # rotasjonen er fastholdt
            Kn[i, :] = 0.0
            Kn[:, i] = 0.0
            Kn[i, i] = 1.0
            Rn[i] = 0.0
    return Kn, Rn
