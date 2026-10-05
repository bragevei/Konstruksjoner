# Pakker
import numpy as np

# Funksjoner
from lesinput import lesinput
from lengder import lengder


def main():

    # -----Leser inputdata
    npunkt, punkt, nelem, elemkonn, tvsnitt = lesinput()

    # -----Beregner elementlengder
    elemlen = lengder(punkt, elemkonn)

    # -----Beregner bøyestivhet for alle elementer
    # Lag funksjonen selv
    # EI = boyestivhet(tvsnitt, geom, ...

    # ------Bygger systemlastvektor
    R = np.zeros(npunkt)
    #for ilast in lastdata:
        # -----Beregner elementlastvektor S_fim m/fastinnspenningsmomenter for elementer med ytre last
        # Lag funksjonen selv
        # S_fim = elemlast(elemlen, ...

        # -----Adderer elementlastvektor S_fim inn i systemlastvektor R vha. elementkonnektivitet
        # Lag funksjonen selv
        # R = elemlast_til_syslast(R, S_fim, elemkonn )

        # -----Adderer knutepunktsmoment inn i systemlastvektor R
        # Lag funksjonen selv
        # R = knutmom(R, ...

    # ------Bygger systemstivhetsmatrisen ved å innaddere elementstivhetsmatriser vha. elementkonnektivitet
    # Lag funksjonen selv
    # K = stivmat(nelem, npunkt, tvsnitt, elemkonn, elemlen, EI, ...

    # ------Innfører grensebetingelser
    # Lag funksjonen selv basert på valgt metode for innføring av grensebetingelser
    # Kn, Rn = bc(punkt, K, R)

    # -----Løser ligningssystemet------
    # rot = np.linalg.solve(Kn, Rn)
    
    #------Beregner momentverdier for alle element ved endene, 
    #------og ved midtpunkt for fordelt last og under punktlaster
    #------vha. superposisjonsprinsippet
    # Lag funksjonen selv
    # Mval = moment(nlast, last, elemkonn, elemlen, rot, tvsnitt, EI, ...

    #------Beregner skjærkraftverdier for alle element ved endene
    #------vha. enkel derivasjon (Q=dM/ds) for Q-bidrag fra moment pga.
    #------bjelkeenderotasjoner, og bruker superposisjonsprinsippet
    #------for å addere til Q-bidrag fra ytre last
    # Lag funksjonen selv
    # Qval = shear(nlast, last, elemkonn, elemlen, rot, tvsnitt, ...

    #------Beregner bøyespenning for alle element ved endene, 
    #------og ved midtpunkt for fordelt last og under punktlaster
    # Lag funksjonen selv
    # sigma_M = boyespenn(Mval, tvsnitt, ...

    #-----Printer bøyespenninger for alle elementene
    # print("Bøyespenninger:")
    # print(sigma_M)

    #-----Printer momentverdier for alle elementer
    # print("Momentverdier for tegning av M-diagram (for hånd):")
    # print(Mval)

    #-----Printer skjærkraftverdier ved endene for alle elementer
    # print("Skjærkraftverdier for tegning av Q-diagram (for hånd):")
    # print(Qval)

main()