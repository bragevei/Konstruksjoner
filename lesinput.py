import numpy as np

def _neste_linjer(linjer, antall, seksjon):
    #Leser et gitt antall datalinjer fra inputfilen
    data = []
    while len(data) < antall:
        try:
            linje = next(linjer)
        except StopIteration as feil:
            raise ValueError(f"Inputfilen mangler data i seksjonen '{seksjon}'") from feil

        linje = linje.split("#", 1)[0].strip()
        if linje:
            data.append(linje.split())

    return data


def _antall(linjer, seksjon):
    #Leser neste heltall fra inputfilen
    try:
        linje = next(linjer)
    except StopIteration as feil:
        raise ValueError(f"Inputfilen mangler antall for seksjonen '{seksjon}'") from feil

    linje = linje.split("#", 1)[0].strip()
    while not linje:
        try:
            linje = next(linjer).split("#", 1)[0].strip()
        except StopIteration as feil:
            raise ValueError(f"Inputfilen mangler antall for seksjonen '{seksjon}'") from feil

    try:
        verdi = int(linje)
    except ValueError as feil:
        raise ValueError(f"Antallet i seksjonen '{seksjon}' må være et heltall.") from feil

    if verdi < 0:
        raise ValueError(f"Antallet i seksjonen '{seksjon}' kan ikke være negativt")
    return verdi


def _tabell(data, kolonner, seksjon):
    #Konverterer rader til en numerisk tabell med riktig antall kolonner
    if not data:
        return np.empty((0, kolonner), dtype=float)

    for rad in data:
        if len(rad) != kolonner:
            raise ValueError(f"Seksjonen '{seksjon}' må ha {kolonner} kolonner per rad")

    try:
        return np.asarray(data, dtype=float)
    except ValueError as feil:
        raise ValueError(f"Seksjonen '{seksjon}' inneholder en verdi som ikke er numerisk") from feil


def lesinput(input_fil):
    # Leser rammedata fra input fil.

    # Returnerer antall og data i samme rekkefølge som i inputfilen:
    # npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata

    # punkt inneholder x-koordinat, y-koordinat, fastholdingsbetingelse og knutepunkt-ID. 
    # geom inneholder gruppe-ID og tverrsnittsdata.
    # tvsnitt inneholder E-modul og gruppe-ID per element. lastdata inneholder
    # lasttype, koordinater og laststørrelse, der lasttype 3 er trekantlast
    # med maksimum i startpunktet og null i sluttpunktet.
    
    try:
        with open(input_fil, "r", encoding="utf-8") as fil:
            linjer = iter(fil)

            npunkt = _antall(linjer, "knutepunkter")
            punkt = _tabell(_neste_linjer(linjer, npunkt, "knutepunkter"), 4, "knutepunkter",)

            ngeom = _antall(linjer, "tverrsnittsgeometrier")
            geom = _tabell(_neste_linjer(linjer, ngeom, "tverrsnittsgeometrier"), 10, "tverrsnittsgeometrier",)

            nelem = _antall(linjer, "elementer")
            elem = _tabell(_neste_linjer(linjer, nelem, "elementer"), 3, "elementer",)

            nlast = _antall(linjer, "laster")
            lastdata = _tabell(_neste_linjer(linjer, nlast, "laster"), 6, "laster",)

    except OSError as feil:
        raise OSError(f"Kunne ikke åpne inputfilen '{input_fil}'") from feil

    lasttyper = lastdata[:, 0].astype(int) if nlast else np.empty(0, dtype=int)
    if nlast and not np.all(lastdata[:, 0] == lasttyper):
        raise ValueError("Lasttype må være et heltall.")
    ugyldige_lasttyper = sorted(set(lasttyper) - {0, 1, 2, 3})
    if ugyldige_lasttyper:
        raise ValueError(f"Ukjent lasttype {ugyldige_lasttyper[0]}; bruk 0, 1, 2 eller 3.")

    # Interne beregninger bruker radindekser. Inputfilen kan bruke vilkårlige,
    # men unike ID-er, så oversett disse én gang her.
    node_ids_raw = elem[:, 0:2]
    punkt_ids_raw = punkt[:, 3]
    group_ids_raw = geom[:, 0]
    if not np.all(node_ids_raw == node_ids_raw.astype(int)):
        raise ValueError("Knutepunkt-ID-er i elementtabellen må være heltall")
    if not np.all(punkt_ids_raw == punkt_ids_raw.astype(int)):
        raise ValueError("Knutepunkt-ID-er må være heltall")
    if not np.all(group_ids_raw == group_ids_raw.astype(int)):
        raise ValueError("Tverrsnittsgruppe-ID-er må være heltall")
    if not np.all(elem[:, 2] == elem[:, 2].astype(int)):
        raise ValueError("Tverrsnittsgruppe-ID-er i elementtabellen må være heltall")

    node_ids = node_ids_raw.astype(int)
    punkt_ids = punkt_ids_raw.astype(int)
    if len(np.unique(punkt_ids)) != len(punkt_ids):
        raise ValueError("Knutepunkt-ID-ene må være unike")
    node_index = {node_id: index for index, node_id in enumerate(punkt_ids)}
    try:
        elemkonn = np.asarray([[node_index[start], node_index[slutt]] for start, slutt in node_ids],dtype=int,)
    except KeyError as feil:
        raise ValueError(
            f"Elementene refererer til ukjent knutepunkt-ID {feil.args[0]}"
        ) from feil

    group_ids = group_ids_raw.astype(int)
    if len(np.unique(group_ids)) != len(group_ids):
        raise ValueError("Tverrsnittsgruppe-ID-ene må være unike. ")
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    try:
        tvsnitt = np.asarray([group_index[group_id] for group_id in elem[:, 2].astype(int)], dtype=int, )
    except KeyError as feil:
        raise ValueError(f"Elementene refererer til ukjent tverrsnittsgruppe-ID {feil.args[0]}") from feil

    # Resten av programmet bruker radindeksen som systemindeks.
    punkt = punkt.copy()
    punkt[:, 3] = np.arange(npunkt)

    return npunkt, punkt, ngeom, geom, nelem, elemkonn, tvsnitt, nlast, lastdata