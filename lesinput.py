import numpy as np

def lesinput():

    # Åpner inputfilen
    fid = open("input_ramme1.txt", "r")

    # Leser totalt antall punkt
    comlin = fid.readline()            #  Leser kommentarlinje. Må leses fordi 'readline' leser 1 linje, linje for linje
    npunkt = int(fid.readline())       # 'fid.readline()' leser en linje, 'int(...)' gjør at linjen tolkes som et heltall

    # x- og y-koordinater til knutepunktene og grensebetingelse
    # Knutepunktnummer tilsvarer radnummer
    # x-koordinat lagres i kolonne 1, y-koordinat i kolonne 2
    # Grensebetingelse lagres i kolonne 3; 1 = fast innspent og 0 = fri rotasjon
    punkt = np.loadtxt(fid, dtype = int, max_rows = npunkt)     # 'max_rows = npunkt' sorger for at vi bare leser 
																# de 'npunkt' neste linjene i tekstfilen

    # Leser antall tverrsnittsgeometrier
    comlin = fid.readline() 
    ngeom = int(fid.readline())
    
    geom = np.loadtxt(fid, dtype = float, max_rows = ngeom)


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

    # Leser elementlengder
    # x_start, y_start, x_slutt, y_slutt
    # Kolonne 1: x-koordinat til startpunkt
    # Kolonne 2: y-koordinat til startpunkt
    # Kolonne 3: x-koordinat til sluttpunkt
    # Kolonne 4: y-koordinat til sluttpunkt
    # comlin = fid.readline() 
    # elemlen=np.loadtxt(fid, dtype=float)

    # x_start = elem[:, 0]
    # y_start = elem[:, 1]
    # x_slutt = elem[:, 2]
    # y_slutt = elem[:, 3]

    # Leser antall laster som virker på rammen
    # Kolonne 1 og 2: x- og y-koordinat til startpunkt
    # Kolonne 3 og 4: x- og y-koordinat til sluttpunkt
    # Kolonne 5: laststørrelse
    # Kolonne 6: lasttype
    # 0 = fordelt last
    # 1 = punktlast
    # 2 = moment
    # comlin = fid.readline() 
    # lastdata = np.loadtxt(fid, dtype=float)

    # last_x_start = lastdata[:, 0]
    # last_y_start = lastdata[:, 1]
    # last_x_slutt = lastdata[:, 2]
    # last_y_slutt = lastdata[:, 3]
    # last_storrelse = lastdata[:, 4]
    # last_type = lastdata[:, 5]

    # Lukker input-filen
    fid.close()

    return npunkt, punkt, nelem, elemkonn, tvsnitt
