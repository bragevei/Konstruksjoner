"""
main.py - rammeanalyse med matrisemetoden (TMR4167, prosjektoppgave)
=====================================================================

OVERORDNET TENKEMÅTE
--------------------
Programmet analyserer plane rammer som antas UFORSKYVELIGE: knutepunktene kan
ikke forskyves, bare rotere. Deformasjonen er da fullstendig beskrevet av én
ukjent rotasjon per knutepunkt, og analysen følger disse stegene:

  1. Les inputfilen (knutepunkter, materialer, tverrsnitt, elementer, laster).
  2. Beregn elementlengdene.
  3. Knytt lastene til elementene og beregn fastinnspenningsmomentene
     (elementlastvektorene S_fim). Sett sammen systemlastvektoren R ved hjelp
     av konnektivitetstabellen og legg til ytre knutepunktsmomenter.
  4. Beregn bøyestivheten EI for hvert element ut fra tverrsnitt og material,
     og sett sammen systemstivhetsmatrisen K.
  5. Innfør grensebetingelsene (fastholdte rotasjoner) og løs K*r = R.
  6. Beregn endemomentene  M = EI/L*(4r1 + 2r2) + M1_fim  osv.
  7. Finn moment i tre punkter og skjærkraft i endene med superposisjon,
     og beregn bøyespenningene sigma = M*c/I.

Steg 2-3 avhenger ikke av tverrsnittene og gjøres bare én gang. Steg 4-7
samles i funksjonen analyser(), som kalles mange ganger under dimensjoneringen
i oppgave c), der tverrsnittene endres til spenningskravet er oppfylt.

Kjøring:  python main.py
  - Oppgave b): plattformen fra Øving 8 analyseres med oppgitte stivheter.
  - Oppgave c): rammen i figur 1 dimensjoneres og analyseres.

Enheter: lengder i m, krefter i kN, momenter i kNm, tverrsnittsmål i mm,
E og spenninger i MPa (N/mm^2).
"""

import numpy as np

from lesinput import lesinput, E_N1, E_N2, E_GEOM, P_ROT, G_TYPE, G_D, G_T, G_H, G_B, G_TF, G_TW, ROR
from lengder import lengder
from beregne_boyestivhet import beregne_boyestivhet, tverrsnittsdata
from systemlastvektor import fordel_laster, elementlastvektorer, systemlastvektor
from stivhetsmatrise import stivhetsmatrise
from grensebetingelser import bc
from verdiberegning import endemomenter, momentverdier, skjaerkrefter, boyespenninger
from dimensjonering import dimensjoner


def _format_significant(value, significant=4):
    """Formaterer et tall med minst ``significant`` gjeldende sifre."""
    if value == 0:
        return f"{value:.{significant - 1}f}"
    decimals = significant - 1 - int(np.floor(np.log10(abs(value))))
    return f"{value:.{max(0, decimals)}f}"


def main(input_fil, dimensjoner_tverrsnitt=False, utnyttelse_maks=0.7, lambda_maks=200):
    """
    Kjører hele analysen for én inputfil.

    input_fil              : navn på inputfilen
    dimensjoner_tverrsnitt : True  -> velg profiler slik at |sigma| <= utnyttelse_maks*fy
                                      (oppgave c). Tverrsnittene i inputfilen ignoreres.
                             False -> bruk tverrsnittene i inputfilen (oppgave b,
                                      eller kontroll av et gitt design)
    utnyttelse_maks        : tillatt |sigma|/fy (0.7 i oppgaveteksten)
    lambda_maks            : praktisk slankhetsgrense under dimensjoneringen
                             (se dimensjonering.py). None slår den av.
    Returnerer en ordbok med alle resultatene, slik at de kan brukes videre.
    """
    # ----- 1. Les inputdata -------------------------------------------------
    npunkt, punkt, nmat, mat, ngeom, geom, nelem, elem, nlast, last = lesinput(input_fil)

    # ----- 2. Elementlengder ------------------------------------------------
    elemlen = lengder(punkt, elem, nelem)

    # ----- 3. Laster, elementlastvektorer og systemlastvektor ---------------
    fordelt, punktlaster, knutemom = fordel_laster(last, nlast, punkt, npunkt, elem, nelem, elemlen)
    S_fim = elementlastvektorer(nelem, elemlen, fordelt, punktlaster)
    R = systemlastvektor(npunkt, nelem, elem, S_fim, knutemom)

    # ----- 4-6. Analyse for et gitt sett med tverrsnitt ---------------------
    # Stivheten avhenger av tverrsnittene, så disse stegene må gjøres på nytt
    # hver gang tverrsnittene endres. De legges derfor i en egen funksjon som
    # "husker" alt som er beregnet over (lengder, S_fim, R osv.), og som bare
    # trenger tverrsnittstabellen som input.
    def analyser(geom):
        # Bygger K for de gitte tverrsnittene, løser K*r = R og returnerer 
        # rotasjonene r og endemomentene [M1, M2] for alle elementer
        EI = beregne_boyestivhet(geom, mat, elem)
        K = stivhetsmatrise(npunkt, nelem, elem, elemlen, EI)
        Kn, Rn = bc(npunkt, punkt, K, R)
        rot = np.linalg.solve(Kn, Rn)
        Mende = endemomenter(rot, nelem, elem, elemlen, EI, S_fim)
        return rot, Mende

    # ----- Dimensjonering (oppgave c) ----------------------------------------
    # Dimensjoneringen prøver ulike profiler og trenger momentene i de tre
    # kontrollpunktene for hvert forsøk. Den får derfor en funksjon som
    # regner ut disse for et vilkårlig sett med tverrsnitt.
    historikk = None
    if dimensjoner_tverrsnitt:
        def momenter_for(geom_kandidat):
            # Momentverdiene i de tre kontrollpunktene for et kandidatdesign
            _, Mende = analyser(geom_kandidat)
            return momentverdier(Mende, nelem, elemlen, fordelt, punktlaster)[1]

        geom, _, historikk = dimensjoner(geom, mat, elem, elemlen, momenter_for,
                                         utnyttelse_maks=utnyttelse_maks, lambda_maks=lambda_maks)

    # ----- 7. Endelig analyse og verdiberegning -----------------------------
    rot, Mende = analyser(geom)
    xpos, Mval = momentverdier(Mende, nelem, elemlen, fordelt, punktlaster)
    Qval = skjaerkrefter(Mende, nelem, elemlen, fordelt, punktlaster)
    sigma, utnyttelse = boyespenninger(Mval, geom, mat, elem, nelem)

    # Kontroll av løsningen: i hvert knutepunkt med fri rotasjon skal summen av
    # endemomentene fra elementene være lik det ytre momentet i knutepunktet.
    # np.add.at summerer riktig også når samme knutepunkt forekommer flere ganger.
    momentsum = np.zeros(npunkt)
    np.add.at(momentsum, elem[:, E_N1], Mende[:, 0])
    np.add.at(momentsum, elem[:, E_N2], Mende[:, 1])
    frie = punkt[:, P_ROT] == 1
    likevektsavvik = np.max(np.abs(momentsum[frie] - knutemom[frie]), initial=0.0)

    resultater = dict(punkt=punkt, geom=geom, mat=mat, elem=elem, elemlen=elemlen, rot=rot,
                      S_fim=S_fim, R=R, Mende=Mende, xpos=xpos, Mval=Mval, Qval=Qval,
                      sigma=sigma, utnyttelse=utnyttelse, historikk=historikk,
                      likevektsavvik=likevektsavvik)
    skriv_resultater(input_fil, resultater, utnyttelse_maks)
    return resultater


# ===========================================================================
# Utskrift av resultater
# ===========================================================================
def profilnavn(rad):
    # Lesbart navn og alle mål for et tverrsnitt, slik de skal legges inn i 3D Beam
    if int(rad[G_TYPE]) == ROR:
        if rad[G_D] == 0:
            return "kun oppgitt I"
        return f"Rør D={rad[G_D]:g} t={rad[G_T]:g} (D/t={rad[G_D] / rad[G_T]:.1f})"
    if rad[G_H] == 0:
        return "kun oppgitt I"
    return (f"I-profil h={rad[G_H]:g} b={rad[G_B]:g} tf={rad[G_TF]:g} tw={rad[G_TW]:g} "
            f"(steghøyde mellom flensene = {rad[G_H] - 2 * rad[G_TF]:g})")


def skriv_resultater(input_fil, res, utnyttelse_maks):
    elem, geom = res["elem"], res["geom"]
    nelem = len(elem)
    linje = "=" * 100
    print(f"\n{linje}\n ANALYSE AV {input_fil}\n{linje}")

    if res["historikk"] is not None:
        grupper = sorted(set(elem[:, E_GEOM]))
        print("\nIterasjonshistorikk for dimensjoneringen (profil per tverrsnitt/elementgruppe):")
        print("  iter " + "".join(f"{'tv.snitt ' + str(g):>15s}" for g in grupper))
        for it, navn in enumerate(res["historikk"], start=1):
            print(f"  {it:4d} " + "".join(f"{n:>15s}" for n in navn))

    print("\nKnutepunktsrotasjoner [rad] (positive med klokka):")
    for i, r in enumerate(res["rot"]):
        status = "fastholdt" if res["punkt"][i, P_ROT] == 0 else ""
        print(f"  knutepunkt {i:2d}: {r: .6e}  {status}")

    print("\nEndemomenter, matrisemetodens konvensjon (positive med klokka) [kNm]:")
    print("  elem  kp1  kp2    L [m]    M1_fim     M2_fim        M1          M2")
    for i in range(nelem):
        print(f"  {i:4d} {elem[i, E_N1]:4d} {elem[i, E_N2]:4d} {res['elemlen'][i]:8.3f} "
              f"{res['S_fim'][i, 0] + 0.0:10.3f} {res['S_fim'][i, 1] + 0.0:10.3f} "
              f"{res['Mende'][i, 0]:11.4f} {res['Mende'][i, 1]:11.4f}")

    print("\nVerdier for M- og Q-diagram (positivt M = strekk på høyre side sett fra kp1 mot kp2,"
          "\ndvs. undersiden av bjelker definert venstre->høyre og høyre side av søyler definert"
          " nedenfra og opp. Q = dM/dx):")
    print("  elem  kp1->kp2  x_midt[m]   M(ende 1)   M(midt/P)   M(ende 2)    Q(ende 1)   Q(ende 2)")
    for i in range(nelem):
        M, Q = res["Mval"][i], res["Qval"][i]
        print(f"  {i:4d}  {elem[i, E_N1]:3d}->{elem[i, E_N2]:<3d} {res['xpos'][i, 1]:9.3f} "
              f"{_format_significant(M[0]):>11s} {_format_significant(M[1]):>11s} "
              f"{_format_significant(M[2]):>11s}  {_format_significant(Q[0]):>11s} "
              f"{_format_significant(Q[1]):>11s}")

    if np.all(np.isnan(res["utnyttelse"])):
        print("\n(Bøyespenninger beregnes ikke: tverrsnittsmål eller flytspenning er ikke oppgitt.)")
    else:
        print(f"\nBøyespenninger [MPa] i de samme punktene og utnyttelse |sigma|maks/fy "
              f"(krav: <= {utnyttelse_maks:.0%}):")
        print("  elem  tv.snitt   sigma(ende 1)  sigma(midt/P)  sigma(ende 2)   utnyttelse")
        for i in range(nelem):
            s = res["sigma"][i]
            print(f"  {i:4d}  {elem[i, E_GEOM]:6d}  {s[0]:14.1f} {s[1]:14.1f} {s[2]:14.1f}"
                  f"   {res['utnyttelse'][i]:9.1%}")

        print("\nOppsummering per tverrsnitt (elementgruppe). Målene kan legges direkte inn i 3D Beam:")
        for g in sorted(set(elem[:, E_GEOM])):
            e = np.flatnonzero(elem[:, E_GEOM] == g)
            styrende = e[np.argmax(res["utnyttelse"][e])]
            A, I, c = tverrsnittsdata(geom[g])
            print(f"  Tverrsnitt {g}  (elementer {', '.join(map(str, e))}): {profilnavn(geom[g])} [mm]")
            print(f"       A = {A:.0f} mm^2,  I = {I:.4e} mm^4,  W = {I / c:.4e} mm^3")
            print(f"       største |sigma| = {np.max(np.abs(res['sigma'][e])):.1f} MPa i element "
                  f"{styrende}, utnyttelse = {res['utnyttelse'][styrende]:.1%}")
        if np.nanmax(res["utnyttelse"]) > utnyttelse_maks + 1e-9:
            print(f"\n  ADVARSEL: minst ett element har utnyttelse over {utnyttelse_maks:.0%}!")

    print(f"\nKontroll: største avvik i momentlikevekt i frie knutepunkter = "
          f"{res['likevektsavvik']:.2e} kNm")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # Oppgave b): kontroll mot Øving 8. Plattformen er regnet med p = 1, a = 1
    # og E = 1, så momentene som skrives ut er i enheter av p*a^2.
    main("input_halvt_nedsenkbar_plattform.txt", dimensjoner_tverrsnitt=False)

    # Oppgave c): dimensjonering av rammen i figur 1 (kandidatnummer med A = 1/3/5, B = 1/2/3).
    main("input_ramme1.txt", dimensjoner_tverrsnitt=True)
