"""
beregne_boyestivhet.py - tverrsnittsegenskaper og bøyestivhet EI
=================================================================

OVERORDNET TENKEMÅTE
--------------------
Elementstivheten i matrisemetoden er proporsjonal med bøyestivheten EI, der
E kommer fra materialet og I (2. arealmoment) fra tverrsnittet. Hvert element
peker via tverrsnittnummer og materialnummer i inputfilen på én rad i
tverrsnittstabellen (geom) og én rad i materialtabellen (mat). Dermed kan
elementene ha ulike tverrsnitt OG ulike materialer (E-modul).

Tverrsnittsegenskapene (areal A, 2. arealmoment I og avstand z fra
nøytralaksen til ytterste fiber) beregnes ett sted, i tverrsnittsdata(), og
brukes både til stivheten her, til bøyespenningen sigma = M*z/I i
verdiberegning.py og til profilkatalogen i dimensjonering.py. Da er vi sikre
på at alle delene av programmet regner med nøyaktig samme tverrsnitt.

Enheter: Tverrsnittsmålene er i mm og E i MPa (= N/mm^2), mens resten av
programmet regner i kN og m. EI gjøres derfor om til kNm^2:
    E [N/mm^2] * 1e3  = E [kN/m^2]
    I [mm^4]   * 1e-12 = I [m^4]
"""

import numpy as np

from lesinput import (G_TYPE, G_D, G_T, G_H, G_B, G_TF, G_TW, G_I,
                      M_E, E_GEOM, E_MAT, ROR, IPROFIL)


def tverrsnittsdata(rad):
    """
    Beregner tverrsnittsegenskaper for én rad i tverrsnittstabellen.

    Returnerer (A, I, z):
        A : tverrsnittsareal [mm^2]
        I : 2. arealmoment om den sterke aksen [mm^4]
        z : avstand fra nøytralaksen til ytterste fiber [mm]
    Hvis bare I er oppgitt (alle mål lik 0), er A og z ukjente (NaN), og da kan
    bøyespenningen ikke beregnes. Dette gjelder plattformen i oppgave b.
    """
    profiltype = int(rad[G_TYPE])
    I_oppgitt = rad[G_I]

    if profiltype == ROR:
        D, t = rad[G_D], rad[G_T]
        if D == 0 and t == 0 and I_oppgitt > 0:          # Dersom bare I er gitt
            return np.nan, I_oppgitt, np.nan
        if D <= 0 or t <= 0 or 2 * t >= D:
            raise ValueError(f"Tverrsnitt {int(rad[0])}: ugyldige rørmål D = {D}, t = {t}.")
        d = D - 2 * t                                    # Indre diameter
        A = np.pi / 4 * (D**2 - d**2)
        I = np.pi / 64 * (D**4 - d**4)
        z = D / 2                                        # Ytterste fiber = ytre radius

    elif profiltype == IPROFIL:
        h, b, tf, tw = rad[G_H], rad[G_B], rad[G_TF], rad[G_TW]
        if h == 0 and b == 0 and I_oppgitt > 0:          # For når bare I er gitt
            return np.nan, I_oppgitt, np.nan
        if min(h, b, tf, tw) <= 0 or 2 * tf >= h or tw > b:
            raise ValueError(f"Tverrsnitt {int(rad[0])}: ugyldige I-profilmål "
                             f"h = {h}, b = {b}, tf = {tf}, tw = {tw}.")
        hw = h - 2 * tf                                  # steghøyde mellom flensene
        A = 2 * b * tf + hw * tw
        # To flenser: egenbidrag b*tf^3/12 + Steiner-ledd b*tf*(avstand til flensens tyngdepunkt)^2,
        # der avstanden er (h - tf)/2. Steget har bare egenbidrag tw*hw^3/12 fordi tyngdepunktet ligger i nøytralaksen.
        # Flens- og stegtykkelse er uavhengige størrelser (tf og tw), slik oppgaven krever.
        I_flenser = 2 * (b * tf**3 / 12 + b * tf * ((h - tf) / 2)**2)
        I_steg = tw * hw**3 / 12
        I = I_flenser + I_steg
        z = h / 2                                       # symmetrisk profil

    else:
        raise ValueError(f"Ukjent tverrsnittstype {profiltype}.")

    if I_oppgitt > 0:      # en oppgitt I-verdi overstyrer verdien beregnet fra målene
        I = I_oppgitt
    return A, I, z


def beregne_boyestivhet(geom, mat, elem):
    """
    Returnerer bøyestivheten EI [kNm^2] for hvert element.

    For hvert element hentes I fra tverrsnittet elementet peker på og E fra
    materialet elementet peker på.
    """
    I_tverrsnitt = np.array([tverrsnittsdata(rad)[1] for rad in geom])   # [mm^4]
    E = mat[elem[:, E_MAT], M_E] * 1e3                                    # [kN/m^2]
    I = I_tverrsnitt[elem[:, E_GEOM]] * 1e-12                            # [m^4]
    if np.any(E <= 0) or np.any(I <= 0):
        raise ValueError("E og I må være positive for alle elementer.")
    return E * I
