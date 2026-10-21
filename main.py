# Pakker
import numpy as np

# Funksjoner
from lesinput import lesinput
from beregne_boyestivhet import beregne_boyestivhet
from systemlastvektor import elemlast, elemlast_til_syslast, knutmom
from stivhetsmatrise import elementstivhetsmatrise
from grensebetingelser import bc
from verdiberegning import moment, shear, boyespenn


def main(input_fil="input_ramme1.txt"):
    # -----Leser inputdata
    npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata = lesinput(input_fil)

    # -----Beregner bøyestivhet for alle elementer
    EI = beregne_boyestivhet(geom, tvsnitt)

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

    # ------Bygger systemstivhetsmatrisen ved å innaddere elementstivhetsmatriser vha. elementkonnektivitet    
    K = elementstivhetsmatrise(npunkt, punkt, nelem, elemkonn, tvsnitt, EI)

    # ------Innfører grensebetingelser
    Kn, Rn = bc(punkt, K, R)

    # -----Løser ligningssystemet
    rot = np.linalg.solve(Kn, Rn)

    #------Beregner momentverdier for alle element ved endene, og ved midtpunkt for fordelt last og under punktlaster
    #------vha. superposisjonsprinsippet
    Mval = moment(rot, punkt, elemkonn, EI, lastdata)

    #------Beregner skjærkraftverdier for alle element ved endene vha. enkel derivasjon (Q=dM/ds) for Q-bidrag fra moment
    #------pga. bjelkeenderotasjoner, og bruker superposisjonsprinsippet for å addere til Q-bidrag fra ytre last
    Qval = shear(rot, punkt, elemkonn, EI, lastdata)

    #------Beregner bøyespenning for alle element ved endene, og ved midtpunkt for fordelt last og under punktlaster
    sigma_M = boyespenn(Mval, geom, tvsnitt)

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

main()