"""
systemlastvektor.py - laster, fastinnspenningsmomenter og systemlastvektor
==========================================================================

OVERORDNET TENKEMÅTE
--------------------
Rammen er uforskyvelig, så de eneste ukjente er knutepunktsrotasjonene.
Lastene påvirker likningssystemet K*r = R på to måter:

  1) Laster PÅ elementene (fordelte laster og punktlaster). Vi tenker oss
     først at begge elementendene er fast innspent (r = 0). Da oppstår
     fastinnspenningsmomentene (FIM), som samles i elementlastvektoren
     S_fim = [M1_fim, M2_fim] for hvert element.
  2) Ytre moment som virker direkte i et knutepunkt (f.eks. M1 fra kranen).

Momentlikevekt i et knutepunkt gir at summen av elementenes endemomenter er
lik det ytre momentet i knutepunktet. Endemomentet i et element er
M = k*r + S_fim, slik at likevekt for hele systemet blir

        K * r = R,    der   R = (ytre knutepunktsmomenter) - (sum av S_fim)

Summen av S_fim bygges opp ("assembleres") ved hjelp av konnektivitets-
tabellen: element-ende 1 hører til knutepunkt elem[ielem, E_N1] og ende 2 til
elem[ielem, E_N2].

Fortegnskonvensjon (som i TMR4167): Rotasjoner og endemomenter er positive
MED klokka. Laster er positive i elementets lokale y-retning (90 grader mot
klokka fra elementaksen, dvs. oppover for en bjelke definert fra venstre mot
høyre).

Merk at lastvektoren bare avhenger av lastene og elementlengdene, ikke av
tverrsnittene. Den beregnes derfor bare én gang, også når tverrsnittene
endres under dimensjoneringen.
"""

import numpy as np

from lesinput import (P_X, P_Y, E_N1, E_N2, L_TYPE, L_X1, L_Y1, L_X2, L_Y2,
                      L_V1, L_V2, FORDELT, PUNKT, MOMENT)


def fordel_laster(last, nlast, punkt, npunkt, elem, nelem, elemlen, toleranse=1e-6):
    """
    Finner hvilket element (eller knutepunkt) hver last i inputfilen virker på,
    og uttrykker lasten i elementets lokale koordinater.

    Lastene er gitt med globale koordinater i inputfilen. For hvert element
    regner vi ut enhetsvektoren e langs elementet (fra knutepunkt 1 til 2) og
    normalvektoren n (90 grader mot klokka, lokal y-akse). Et punkt p ligger
    på elementet dersom avstanden normalt på elementet, |(p - p1)*n|, er null
    og avstanden langs elementet, s = (p - p1)*e, ligger mellom 0 og L.

    Returnerer:
        fordelt     : (nelem x 2) lastintensitet [kN/m] i ende 1 og ende 2 av
                      hvert element, lineært varierende mellom endene.
        punktlaster : liste med én liste per element, med (a, P) for hver
                      punktlast, der a [m] er avstanden fra ende 1.
        knutemom    : (npunkt) ytre moment [kNm] i hvert knutepunkt.
    """
    xy = punkt[:, [P_X, P_Y]]
    p1 = xy[elem[:, E_N1]]                       # startpunkt for hvert element
    e = (xy[elem[:, E_N2]] - p1) / elemlen[:, None]   # enhetsvektor langs elementet
    n = np.column_stack((-e[:, 1], e[:, 0]))     # lokal y-akse (90 grader mot klokka)

    fordelt = np.zeros((nelem, 2))
    punktlaster = [[] for _ in range(nelem)]
    knutemom = np.zeros(npunkt)

    for ilast in range(nlast):
        lasttype = int(last[ilast, L_TYPE])
        pa = last[ilast, [L_X1, L_Y1]]           # lastens (start)punkt
        pb = last[ilast, [L_X2, L_Y2]]           # lastens sluttpunkt (bare type 0)
        v1, v2 = last[ilast, L_V1], last[ilast, L_V2]

        if lasttype == MOMENT:
            # Et ytre moment kan kun virke i et knutepunkt og legges så rett inn i R
            treff = np.flatnonzero(np.linalg.norm(xy - pa, axis=1) <= toleranse)
            if len(treff) != 1:
                raise ValueError(f"Last {ilast}: momentet må virke i et knutepunkt")
            knutemom[treff[0]] += v1

        elif lasttype == PUNKT:
            # Avstand langs (s) og normalt på (d) hvert element, regnet fra elementets ende 1
            # Prikkproduktet tas rad for rad med einsum.
            s = np.einsum("ij,ij->i", pa - p1, e)
            d = np.abs(np.einsum("ij,ij->i", pa - p1, n))
            paa_element = (d <= toleranse) & (s >= -toleranse) & (s <= elemlen + toleranse)
            if not np.any(paa_element):
                raise ValueError(f"Last {ilast}: punktlasten ligger ikke på noe element.")
            i_knutepunkt = paa_element & ((s <= toleranse) | (s >= elemlen - toleranse))
            if np.any(i_knutepunkt):
                # En punktlast rett i et knutepunkt gir ingen bøyning i en uforskyvelig ramme.
                # Den påvriker kun elementenes aksialkraft og opplagrene, 
                # og har derfor ingen påvirkning på rotasjonene eller momentene
                continue
            ielem = np.flatnonzero(paa_element)[0]   # punktet ligger inne på ett element
            punktlaster[ielem].append((s[ielem], v1))

        elif lasttype == FORDELT:
            # Lasten går langs linjen fra pa til pb. Den kan dekke flere elementer som ligger på linje 
            # (f.eks. vindlasten over hele den venstre søylen). Hvert element som ligger på linjen får lasten
            # over hele sin lengde, med intensitet funnet ved lineær interpolasjon mellom v1 (i pa) og v2 (i pb).
            lastlengde = np.linalg.norm(pb - pa)
            if lastlengde <= toleranse:
                raise ValueError(f"Last {ilast}: fordelt last må ha ulikt start- og sluttpunkt")
            u = (pb - pa) / lastlengde                       # retning langs lasten
            u_n = np.array([-u[1], u[0]])                    # normal til lastlinjen
            dekket_lengde = 0.0
            for ielem in range(nelem):
                ende1, ende2 = p1[ielem], p1[ielem] + e[ielem] * elemlen[ielem]
                # Ligger begge elementendene på den uendelige linjen gjennom pa og pb?
                if abs((ende1 - pa) @ u_n) > toleranse or abs((ende2 - pa) @ u_n) > toleranse:
                    continue
                t1, t2 = (ende1 - pa) @ u, (ende2 - pa) @ u  # posisjon langs lastlinjen
                # Lengden av overlappet mellom elementet og lasten:
                overlapp = min(max(t1, t2), lastlengde) - max(min(t1, t2), 0.0)
                if overlapp <= toleranse:
                    continue      # elementet ligger utenfor lasten (eller berører bare et endepunkt)
                if overlapp < elemlen[ielem] - toleranse:
                    raise ValueError(f"Last {ilast}: fordelt last må starte og slutte i knutepunkter.")
                # Fortegnet er definert i elementets lokale y-retning. Hvis elementet peker motsatt vei av lastlinjen, 
                # ville lasten fått motsatt fortegn på dette elementet. Det er nesten alltid en feil i inputfilen, 
                # så programmet stopper heller enn å gjette.
                if e[ielem] @ u < 0:
                    raise ValueError(f"Last {ilast}: element {ielem} peker motsatt vei av lasten "
                                     f"Definer lasten i samme retning som elementene "
                                     f"(fra knutepunkt 1 mot knutepunkt 2).")
                fordelt[ielem, 0] += v1 + (v2 - v1) * t1 / lastlengde
                fordelt[ielem, 1] += v1 + (v2 - v1) * t2 / lastlengde
                dekket_lengde += elemlen[ielem]
            if abs(dekket_lengde - lastlengde) > toleranse:
                raise ValueError(f"Last {ilast}: den fordelte lasten dekker ikke hele elementer "
                                 f"langs linjen den er definert på.")

    return fordelt, punktlaster, knutemom


def elementlastvektor(q1, q2, punktlaster, L):
    """
    Fastinnspenningsmomentene S_fim = [M1_fim, M2_fim] for ett element med
    begge ender fast innspent (positive med klokka).

    Lineært fordelt last q1 -> q2 (oppgaven: "splittes i to trekantbidrag"):
        Trekant med q1 i ende 1 og 0 i ende 2: M1 = q1*L^2/20, M2 = -q1*L^2/30
        Trekant med 0 i ende 1 og q2 i ende 2: M1 = q2*L^2/30, M2 = -q2*L^2/20
        Sum:  M1 =  L^2/60 * (3*q1 + 2*q2)
              M2 = -L^2/60 * (2*q1 + 3*q2)
        Kontroll med jevn last q nedover (q1 = q2 = -q):  M1 = -qL^2/12 og
        M2 = +qL^2/12, dvs. mot klokka i venstre ende og med klokka i høyre.

    Punktlast P i avstand a fra ende 1 (b = L - a):
        M1 =  P*a*b^2/L^2
        M2 = -P*a^2*b/L^2

    Flere laster på samme element legges sammen (superposisjon).
    """
    S_fim = np.array([L**2 / 60 * (3 * q1 + 2 * q2),
                      -L**2 / 60 * (2 * q1 + 3 * q2)])
    for a, P in punktlaster:
        b = L - a
        S_fim += np.array([P * a * b**2 / L**2,
                           -P * a**2 * b / L**2])
    return S_fim


def elementlastvektorer(nelem, elemlen, fordelt, punktlaster):
    # Beregner S_fim for alle elementer. Rad ielem = [M1_fim, M2_fim]
    S_fim = np.zeros((nelem, 2))
    for ielem in range(nelem):
        S_fim[ielem] = elementlastvektor(fordelt[ielem, 0], fordelt[ielem, 1],
                                         punktlaster[ielem], elemlen[ielem])
    return S_fim


def systemlastvektor(npunkt, nelem, elem, S_fim, knutemom):
    """
    Setter sammen systemlastvektoren R = (ytre knutepunktsmomenter) - (sum S_fim).

    Konnektivitetstabellen elem forteller hvilket knutepunkt (systemfrihetsgrad)
    hver element-ende hører til: ende 1 -> elem[ielem, E_N1], ende 2 -> elem[ielem, E_N2].
    """
    R = knutemom.astype(float).copy()
    for ielem in range(nelem):
        R[elem[ielem, E_N1]] -= S_fim[ielem, 0]
        R[elem[ielem, E_N2]] -= S_fim[ielem, 1]
    return R
