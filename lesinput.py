import numpy as np

def lesinput():

    # Åpner inputfilen
    fid = open("input.txt", "r")

    # Leser totalt antall punkt
    comlin = fid.readline()            #  Leser kommentarlinje. Må leses fordi 'readline' leser 1 linje, linje for linje
    npunkt = int(fid.readline())       # 'fid.readline()' leser en linje, 'int(...)' gjør at linjen tolkes som et heltall

    # x- og y-koordinater til knutepunktene og grensebetingelse
    # Knutepunktnummer tilsvarer radnummer
    # x-koordinat lagres i kolonne 1, y-koordinat i kolonne 2
    # Grensebetingelse lagres i kolonne 3; 1 = fast innspent og 0 = fri rotasjon
    punkt = np.loadtxt(fid, dtype = int, max_rows = npunkt)     # 'max_rows = npunkt' sorger for at vi bare leser 
																# de 'npunkt' neste linjene i tekstfilen

    # Leser antall elementer
    comlin = fid.readline() 
    nelem = int(fid.readline())

    # Kolonne 1 og 2: Elementkonnektivitet, dvs. sammenheng mellom elementfrihetsgrader og systemfrihetsgrader
    # Kolonne 3: E-modul
    # Kolonne 4: Tverrsnittstype
    # Elementnummer tilsvarer radnummer
    # Systemfrihetsgrad for lokal frihetsgrad 1 lagres i kolonne 1
    # Systemfrihetsgrad for lokal frihetsgrad 2 lagres i kolonne 2
    # Det anbefales at nummerering av systemfrihetsgrad starter på 0, slik at det samsvarerer med indeksering i Python
    elem = np.loadtxt(fid, dtype = int, max_rows = nelem)

    # Elementkonnektivitet
    # Kolonne 1: Systemfrihetsgrad for elementfrihetsgrad 1
    # Kolonne 2: Systemfrihetsgrad for elementfrihetsgrad 2
    elemkonn = elem[0:nelem,0:2]

    # Tverrsnittsdata
    # Kolonne 1: E-modul
    # Kolonne 2: Tverrsnittstype, I-profil=1 og rørprofil=2
    tvsnitt = elem[0:nelem,2:4]

    # Leser antall laster som virker på rammen
    comlin = fid.readline() 
    ngeom = int(fid.readline())

    # Leser geometridata for tverrsnittstypene
    # Bestem selv verdiene som er nødvendig å lese inn, samt hva verdiene som leses inn skal representere
    # geom = np.loadtxt(fid, dtype = float, max_rows = ngeom)

    # Leser antall laster som virker på rammen
    comlin = fid.readline() 
    nlast = int(fid.readline())

    # Leser lastdata
    # Bestem selv verdiene som er nødvendig å lese inn, samt hva verdiene som leses inn skal representere
    # lastdata = np.loadtxt(fid, dtype = float, max_rows = nlast)     # <-- Forslag til innlesing av lastdata

    # Lukker input-filen
    fid.close()

    return npunkt, punkt, nelem, elemkonn, tvsnitt