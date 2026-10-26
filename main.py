# Pakker
import numpy as np

# Funksjoner
from lesinput import lesinput, sett_standardprofil, standardprofiler, tverrsnittnavn
from beregne_boyestivhet import beregne_boyestivhet
from systemlastvektor import elemlast, elemlast_til_syslast, knutmom
from stivhetsmatrise import elementstivhetsmatrise
from grensebetingelser import bc
from verdiberegning import moment, shear, boyespenn

def main(input_fil="input_ramme1.txt", dimensjoner=True):
    # -----Leser inputdata
    npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata = lesinput(input_fil)
    oppgitte_tverrsnitt = tverrsnittnavn(geom)

    # ------Bygger systemlastvektor
    R = np.zeros(npunkt)

    # -----Beregner elementlastvektor S_fim m/fastinnspenningsmomenter
    S_fim = elemlast(lastdata, punkt, elemkonn)

    # -----Adderer S_fim inn i systemlastvektor R vha. elementkonnektivitet
    # S_fim er fastinnspenningsmoment. Likevekten er K*rot + S_fim = 0,
    # derfor brukes motsatt fortegn som høyreside i rotasjonsligningen.
    R = elemlast_til_syslast(R, -S_fim, elemkonn)

    # -----Adderer knutepunktsmoment inn i systemlastvektor R
    R = knutmom(R, lastdata, punkt)

    # Ser om koden skal dimensjonere (som i c)) eller analysere (som i b))
    if dimensjoner:
        geom, profilvalg, Mval, Qval, sigma_M = (dimensjoner_med_global_stivhet(geom, punkt, elemkonn, tvsnitt, lastdata, R))
        #Printer tverrsnittene som blir funnet
        print("Tverrsnitt per gruppe:")
        for gruppe, (oppgitt, valgt) in enumerate(zip(oppgitte_tverrsnitt, profilvalg)):
            if valgt is None:
                valgt = oppgitt
            print(f"  Gruppe {gruppe}: oppgitt = {oppgitt}; valgt = {valgt}")
    else:
        # Analyse uten å endre tverrsnittene fra inputfilen
        profilvalg = [None] * len(geom)
        Mval, Qval = analyser(geom, punkt, elemkonn, tvsnitt, lastdata, R)
        sigma_M = None

    #-----Printer bøyespenninger for alle elementene
    print("Bøyespenninger:")
    print(sigma_M)

    #-----Printer momentverdier for alle elementer
    print("Momentverdier for tegning av M-diagram (for hånd):")
    print(Mval)

    #-----Printer skjærkraftverdier ved endene for alle elementer
    print("Skjærkraftverdier for tegning av Q-diagram (for hånd):")
    print(Qval)

    return sigma_M, Mval, Qval

# For å iterere gjennom tverrsnittene må vi kunne analysere hvert tversnitt:
def analyser(geom, punkt, elemkonn, tvsnitt, lastdata, R):
    EI = beregne_boyestivhet(geom, tvsnitt)
    K = elementstivhetsmatrise(len(punkt), punkt, len(elemkonn), elemkonn, tvsnitt, EI)
    Kn, Rn = bc(punkt, K, R)
    rot = np.linalg.solve(Kn, Rn)
    Mval = moment(rot, punkt, elemkonn, EI, lastdata)
    Qval = shear(rot, punkt, elemkonn, EI, lastdata)
    return Mval, Qval

# Itererer gjennom tverrsnittene med oppdaterte verdier for andre arealmoment for hvert tverrsnitt,
# slik at vi får en egen stivhetsmatrise for hvert av tverrsnittene:
def dimensjoner_med_global_stivhet(
    geom, punkt, elemkonn, tvsnitt, lastdata, R, maks_iterasjoner=20
):
    geom = np.asarray(geom, dtype=float).copy()
    valg = [None] * len(geom)

    for _ in range(maks_iterasjoner):
        endret = False
        for igruppe, rad in enumerate(geom):
            elementer = np.flatnonzero(tvsnitt == igruppe)
            if len(elementer) == 0:
                continue
            if not np.isfinite(rad[9]) or rad[9] <= 0:
                raise ValueError(f"Gruppe {igruppe} mangler positiv flytespenning")

            valgt = None
            for profil in standardprofiler(rad):
                kandidat_geom = sett_standardprofil(geom, igruppe, profil)
                kandidat_moment, _ = analyser(
                    kandidat_geom, punkt, elemkonn, tvsnitt, lastdata, R
                )
                kandidat_sigma = boyespenn(
                    kandidat_moment, kandidat_geom, tvsnitt
                )[elementer]
                if np.max(np.abs(kandidat_sigma)) <= 0.7 * rad[9]:
                    valgt = profil
                    break
            if valgt is None:
                raise ValueError(
                    f"Ingen standardprofil tilfredsstiller gruppe {igruppe}"
                )

            if valg[igruppe] != valgt[0]:
                geom = sett_standardprofil(geom, igruppe, valgt)
                valg[igruppe] = valgt[0]
                endret = True

        if not endret:
            break
    else:
        raise RuntimeError(f"Profiliterasjonen konvergerte ikke innen {maks_iterasjoner} iterasjoner")

    Mval, Qval = analyser(geom, punkt, elemkonn, tvsnitt, lastdata, R)
    sigma = boyespenn(Mval, geom, tvsnitt)
    for igruppe in range(len(geom)):
        elementer = np.flatnonzero(tvsnitt == igruppe)
        if len(elementer) and np.max(np.abs(sigma[elementer])) > 0.7 * geom[igruppe, 9]:
            raise RuntimeError(f"Endelig profilvalg for gruppe {igruppe} tilfredsstiller ikke spenningskravet")
    return geom, valg, Mval, Qval, sigma


main()