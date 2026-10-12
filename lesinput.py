import numpy as np

def _neste_linjer(linjer, antall, seksjon):
    #Leser et gitt antall datalinjer fra inputfilen
    data = []
    while len(data) < antall:
        try:
            linje = next(linjer)
        except StopIteration as feil:
            raise ValueError(
                f"Inputfilen mangler data i seksjonen '{seksjon}'"
            ) from feil

        linje = linje.split("#", 1)[0].strip()
        if linje:
            data.append(linje.split())

    return data


def _antall(linjer, seksjon):
    #Leser neste heltall fra inputfilen
    try:
        linje = next(linjer)
    except StopIteration as feil:
        raise ValueError(
            f"Inputfilen mangler antall for seksjonen '{seksjon}'"
        ) from feil

    linje = linje.split("#", 1)[0].strip()
    while not linje:
        try:
            linje = next(linjer).split("#", 1)[0].strip()
        except StopIteration as feil:
            raise ValueError(
                f"Inputfilen mangler antall for seksjonen '{seksjon}'"
            ) from feil

    try:
        verdi = int(linje)
    except ValueError as feil:
        raise ValueError(
            f"Antallet i seksjonen '{seksjon}' må være et heltall."
        ) from feil

    if verdi < 0:
        raise ValueError(f"Antallet i seksjonen '{seksjon}' kan ikke være negativt")
    return verdi


def _tabell(data, kolonner, seksjon):
    #Konverterer rader til en numerisk tabell med riktig antall kolonner
    if not data:
        return np.empty((0, kolonner), dtype=float)

    for rad in data:
        if len(rad) != kolonner:
            raise ValueError(
                f"Seksjonen '{seksjon}' må ha {kolonner} kolonner per rad"
            )

    try:
        return np.asarray(data, dtype=float)
    except ValueError as feil:
        raise ValueError(
            f"Seksjonen '{seksjon}' inneholder en verdi som ikke er numerisk"
        ) from feil


def lesinput(input_fil="input_ramme1.txt"):
    # Leser rammedata fra input fil.

    # Returnerer antall og data i samme rekkefølge som i inputfilen:
    # npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata

    # punkt inneholder x-koordinat, y-koordinat, fastholdingsbetingelse og knutepunkt-ID. 
    # geom inneholder gruppe-ID og tverrsnittsdata.
    # tvsnitt inneholder E-modul og gruppe-ID per element, mens lastdata inneholder lasttype, koordinater og laststørrelse.
    
    try:
        with open(input_fil, "r", encoding="utf-8") as fil:
            linjer = iter(fil)

            npunkt = _antall(linjer, "knutepunkter")
            punkt = _tabell(
                _neste_linjer(linjer, npunkt, "knutepunkter"),
                4,
                "knutepunkter",
            )

            ngeom = _antall(linjer, "tverrsnittsgeometrier")
            geom = _tabell(
                _neste_linjer(linjer, ngeom, "tverrsnittsgeometrier"),
                7,
                "tverrsnittsgeometrier",
            )

            nelem = _antall(linjer, "elementer")
            elem = _tabell(
                _neste_linjer(linjer, nelem, "elementer"),
                3,
                "elementer",
            )

            nlast = _antall(linjer, "laster")
            lastdata = _tabell(
                _neste_linjer(linjer, nlast, "laster"),
                6,
                "laster",
            )
    except OSError as feil:
        raise OSError(f"Kunne ikke åpne inputfilen '{input_fil}'") from feil

    # Indekser brukes direkte i NumPy-arrays og må derfor være heltall.
    elemkonn = elem[:, 0:2].astype(int)
    tvsnitt = elem[:, 2]

    return npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata
