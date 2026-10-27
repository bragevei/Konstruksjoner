"""
verdiberegning.py - endemomenter, moment- og skjærkraftverdier og bøyespenning
==============================================================================

OVERORDNET TENKEMÅTE: SUPERPOSISJON
-----------------------------------
Når knutepunktsrotasjonene r er kjent, finnes de totale endemomentene i hvert
element som (positive med klokka, samme konvensjon som i K*r = R):

    M1 = EI/L * (4*r1 + 2*r2) + M1_fim
    M2 = EI/L * (2*r1 + 4*r2) + M2_fim

For å finne momentet inne på elementet bruker vi superposisjonsprinsippet,
slik oppgaveteksten anbefaler, i stedet for snittmetoden:

    1) Elementet som en FRITT OPPLAGT bjelke belastet med den ytre lasten.
       Momentet M0(x) og skjærkreftene Q0 er kjente standardformler.
    2) Elementet belastet med endemomentene M1 og M2. Dette gir et momentbidrag
       som varierer lineært fra M1 i ende 1 til -M2 i ende 2, og en konstant
       skjærkraft -(M1 + M2)/L.

    Totalt:  M(x) = M0(x) + M1*(1 - x/L) - M2*x/L
             Q(0) = Q0(0) - (M1 + M2)/L,   Q(L) = Q0(L) - (M1 + M2)/L

FORTEGNSKONVENSJON FOR DIAGRAMVERDIENE
--------------------------------------
Endemomentene M1, M2 er positive med klokka (matrisemetodens konvensjon). For
tegning av diagrammer er det mer naturlig med en "bjelkekonvensjon":

  * Positivt moment M(x) gir STREKK på elementets høyre side sett fra ende 1
    mot ende 2. For en bjelke definert fra venstre mot høyre er det UNDERSIDEN
    (positivt = nedbøyning/"smil"), og for en søyle definert nedenfra og opp
    er det HØYRE side.
  * Skjærkraften er Q = dM/dx, dvs. positiv når den dreier en elementbit med
    klokka (oppover på venstre snittflate for en bjelke).

Overgangen mellom konvensjonene: i ende 1 er M(0) = M1, og i ende 2 er
M(L) = -M2 (et moment med klokka i høyre ende gir strekk på oversiden).

POSISJONER
----------
Oppgaven krever moment og spenning i tre punkter langs hvert element:
  * ved begge ender, og
  * rett under punktlasten (hvis elementet har én punktlast), ellers midt på.
For elementer med fordelt last er det tilstrekkelig å bruke midtpunktet selv
om maksimalmomentet ikke nødvendigvis ligger der.
"""

import numpy as np

from beregne_boyestivhet import tverrsnittsdata
from lesinput import E_GEOM, E_MAT, M_FY, E_N1, E_N2


def endemomenter(rot, nelem, elem, elemlen, EI, S_fim):
    """Totale endemomenter [M1, M2] (kNm, positive med klokka) for alle elementer."""
    Mende = np.zeros((nelem, 2))
    for ielem in range(nelem):
        r1, r2 = rot[elem[ielem, E_N1]], rot[elem[ielem, E_N2]]
        faktor = EI[ielem] / elemlen[ielem]
        Mende[ielem, 0] = faktor * (4 * r1 + 2 * r2) + S_fim[ielem, 0]
        Mende[ielem, 1] = faktor * (2 * r1 + 4 * r2) + S_fim[ielem, 1]
    return Mende


def _fritt_opplagt_moment(x, L, q1, q2, punktlaster):
    """
    Moment M0(x) i en fritt opplagt bjelke (bjelkekonvensjon, positiv last i
    lokal y-retning, dvs. "oppover" gir negativt moment).

    Lineært fordelt last q1 -> q2 (standardformel):
        M0(x) = -x*(L - x)/(6L) * [(2L - x)*q1 + (L + x)*q2]
        Midt på (x = L/2) gir dette -(q1 + q2)*L^2/16, og for jevn last q
        nedover det kjente qL^2/8.
    Punktlast P i avstand a fra ende 1 (b = L - a):
        M0(x) = -P*b*x/L        for x <= a
        M0(x) = -P*a*(L - x)/L  for x >= a
        Under lasten: -P*a*b/L.
    """
    M0 = -x * (L - x) / (6 * L) * ((2 * L - x) * q1 + (L + x) * q2)
    for a, P in punktlaster:
        b = L - a
        M0 += -P * b * x / L if x <= a else -P * a * (L - x) / L
    return M0


def _fritt_opplagt_skjaer(L, q1, q2, punktlaster):
    """
    Skjærkraft Q0 ved ende 1 og ende 2 i en fritt opplagt bjelke (Q = dM/dx).

    Lineært fordelt last: Q0(0) = -L/6*(2q1 + q2),  Q0(L) = L/6*(q1 + 2q2)
    Punktlast P i a:      Q0(0) = -P*b/L,           Q0(L) = P*a/L
    (For last nedover er Q0(0) lik opplagerkraften i ende 1, og Q0(L) lik minus
    opplagerkraften i ende 2.)
    """
    Q0 = np.array([-L / 6 * (2 * q1 + q2), L / 6 * (q1 + 2 * q2)])
    for a, P in punktlaster:
        b = L - a
        Q0 += np.array([-P * b / L, P * a / L])
    return Q0


def momentverdier(Mende, nelem, elemlen, fordelt, punktlaster):
    """
    Moment (bjelkekonvensjon) i tre punkter per element: ende 1, midtpunkt
    eller under punktlasten, og ende 2.

    Returnerer:
        xpos : (nelem x 3) posisjonene [m] målt fra ende 1
        Mval : (nelem x 3) momentene [kNm] i disse posisjonene
    """
    xpos = np.zeros((nelem, 3))
    Mval = np.zeros((nelem, 3))
    for ielem in range(nelem):
        L = elemlen[ielem]
        q1, q2 = fordelt[ielem]
        laster = punktlaster[ielem]
        if len(laster) > 1:
            raise ValueError(f"Element {ielem} har flere punktlaster. Del elementet med et "
                             f"ekstra knutepunkt slik at hvert element har maks én punktlast.")
        # Midtre kontrollpunkt: under punktlasten hvis den finnes, ellers midt på.
        xmidt = laster[0][0] if laster else L / 2
        xpos[ielem] = [0.0, xmidt, L]

        M1, M2 = Mende[ielem]
        for k, x in enumerate(xpos[ielem]):
            # Superposisjon: fritt opplagt bjelke + lineært bidrag fra endemomentene
            Mval[ielem, k] = (_fritt_opplagt_moment(x, L, q1, q2, laster)
                              + M1 * (1 - x / L) - M2 * x / L)
    return xpos, Mval


def skjaerkrefter(Mende, nelem, elemlen, fordelt, punktlaster):
    """Skjærkraft [kN] ved ende 1 og ende 2 for alle elementer (Q = dM/dx)."""
    Qval = np.zeros((nelem, 2))
    for ielem in range(nelem):
        L = elemlen[ielem]
        M1, M2 = Mende[ielem]
        Q0 = _fritt_opplagt_skjaer(L, fordelt[ielem, 0], fordelt[ielem, 1], punktlaster[ielem])
        # Endemomentene gir en konstant skjærkraft (stigningstallet til det
        # lineære momentbidraget fra M1 til -M2):
        Qval[ielem] = Q0 - (M1 + M2) / L
    return Qval


def boyespenninger(Mval, geom, mat, elem, nelem):
    """
    Bøyespenning sigma = M*c/I [MPa] i de samme tre punktene som Mval, samt
    utnyttelsen |sigma|/fy for hvert element.

    Enheter: M [kNm] * 1e6 = M [Nmm], c [mm], I [mm^4]  ->  sigma [N/mm^2 = MPa].
    Fortegnet følger momentet: positiv sigma er strekk på elementets høyre side
    sett fra ende 1 mot ende 2 (samme side som positivt moment).
    """
    sigma = np.zeros((nelem, 3))
    utnyttelse = np.zeros(nelem)
    for ielem in range(nelem):
        _, I, c = tverrsnittsdata(geom[elem[ielem, E_GEOM]])
        fy = mat[elem[ielem, E_MAT], M_FY]
        sigma[ielem] = Mval[ielem] * 1e6 * c / I
        utnyttelse[ielem] = np.max(np.abs(sigma[ielem])) / fy if fy > 0 else np.nan
    return sigma, utnyttelse
