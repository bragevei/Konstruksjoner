import numpy as np

### De øverste funksjonene og variablene brukes for å finne det rette
_HEA_PROFILER = (
    ("HEA100", 96, 100, 5, 8),
    ("HEA120", 114, 120, 5, 8),
    ("HEA140", 133, 140, 5.5, 8.5),
    ("HEA160", 152, 160, 6, 9),
    ("HEA180", 171, 180, 6, 9.5),
    ("HEA200", 190, 200, 6.5, 10),
    ("HEA220", 210, 220, 7, 11),
    ("HEA240", 230, 240, 7.5, 12),
    ("HEA260", 250, 260, 7.5, 12.5),
    ("HEA280", 270, 280, 8, 13),
    ("HEA300", 290, 300, 8.5, 14),
    ("HEA320", 310, 300, 9, 15.5),
    ("HEA340", 330, 300, 9.5, 16.5),
    ("HEA360", 350, 300, 10, 17.5),
    ("HEA400", 390, 300, 11, 19),
    ("HEA450", 440, 300, 11.5, 21),
    ("HEA500", 490, 300, 12, 23),
    ("HEA600", 590, 300, 13, 25),
    ("HEA700", 690, 300, 14.5, 32),
    ("HEA800", 790, 300, 15, 35.5),
)

_ROR_PROFILER = tuple(
    (f"ROR{diameter}x{tykkelse}", diameter, tykkelse)
    for diameter in (48.3, 60.3, 76.1, 88.9, 101.6, 114.3, 139.7, 168.3,
                     177.8, 193.7, 219.1, 244.5, 273, 323.9, 355.6, 406.4,
                     457, 508, 609.6)
    for tykkelse in (2.3, 2.6, 3.2, 4, 5, 6.3, 8, 10, 12.5, 14.2, 16, 20, 25, 30, 40)
    if 2 * tykkelse < diameter and diameter/tykkelse < 30   # Sørger for at indre diameter ikke blir for liten, og at røret oppfyller D/t < 30
)


def _profilens_treghetsmoment(profil):
    if profil[0].startswith("HEA"):
        _, h, b, tw, tf = profil
        return 2 * (b * tf**3 / 12 + b * tf * (h / 2 - tf / 2)**2) + tw * (h - 2 * tf)**3 / 12
    _, diameter, tykkelse = profil
    return np.pi * (diameter**4 - (diameter - 2 * tykkelse)**4) / 64


def standardprofiler(rad):
    #Returnerer standardprofilene som passer til tverrsnittstypen
    return _HEA_PROFILER if int(rad[7]) == 1 else _ROR_PROFILER

def profilens_ytterfiber(profil):
    return profil[1] / 2

def sett_standardprofil(geom, igruppe, profil):
    #Setter én standardprofil inn i en geometrimatrisekopi
    resultat = np.asarray(geom, dtype=float).copy()
    if profil[0].startswith("HEA"):
        _, h, b, tw, tf = profil
        resultat[igruppe, 1:5] = (tf, b, h - 2 * tf, tw)
    else:
        _, diameter, tykkelse = profil
        resultat[igruppe, 5:7] = (diameter, tykkelse)
    resultat[igruppe, 10] = _profilens_treghetsmoment(profil)
    return resultat

def tverrsnittnavn(geom):
    #Returnerer en lesbar beskrivelse av hvert oppgitt tverrsnitt
    geom = np.asarray(geom, dtype=float)
    if geom.ndim != 2 or geom.shape[1] != 11:
        raise ValueError("geom må ha elleve kolonner for tverrsnittsbeskrivelse")

    navn = []
    for rad in geom:
        if int(rad[7]) == 0:
            navn.append(f"ROR{rad[5]:g}x{rad[6]:g}")
        elif int(rad[7]) == 1:
            navn.append(
                f"I-profil h={rad[1]:g}, b={rad[2]:g}, "
                f"h_steg={rad[3]:g}, t_steg={rad[4]:g}"
            )
        else:
            raise ValueError(f"Ukjent tverrsnittstype {rad[7]:g}")
    return navn

def dimensjoner_tverrsnitt(geom, tvsnitt, Mval):
    # Velger minste standardprofil som holder bøyespenningen under 0,7 fy.

    # Mval skal ha momentene ved elementstart, kontrollpunkt og elements slutt i kNm.
    # Geometrien returneres i samme format som input, med valgte mål og I-verdi i mm-enheter.
    # Returverdiens valg inneholder profilnavn per tverrsnittsgruppe.

    geom = np.asarray(geom, dtype=float)
    tvsnitt = np.asarray(tvsnitt, dtype=int)
    Mval = np.asarray(Mval, dtype=float)
    if geom.ndim != 2 or geom.shape[1] != 11:
        raise ValueError("geom må ha elleve kolonner for profilvalg")
    if Mval.ndim != 2 or Mval.shape[1] != 3 or len(Mval) != len(tvsnitt):
        raise ValueError("Mval må ha formen (antall elementer, 3)")
    if np.any(tvsnitt < 0) or np.any(tvsnitt >= len(geom)):
        raise ValueError("tvsnitt inneholder en ugyldig gruppe-ID")

    resultat = geom.copy()
    valg = []
    for igruppe, rad in enumerate(geom):
        elementer = np.flatnonzero(tvsnitt == igruppe)
        if len(elementer) == 0:
            valg.append(None)
            continue
        flytspenning = rad[9]
        if not np.isfinite(flytspenning) or flytspenning <= 0:
            raise ValueError(f"Gruppe {igruppe} mangler positiv flytespenning")
        maks_moment = float(np.max(np.abs(Mval[elementer])))
        tillatt = 0.7 * flytspenning
        profiler = _HEA_PROFILER if int(rad[7]) == 1 else _ROR_PROFILER
        valgt = None
        for profil in profiler:
            treghetsmoment = _profilens_treghetsmoment(profil)
            ytterfiber = (profil[1] / 2) if profil[0].startswith("HEA") else profil[1] / 2
            spenning = maks_moment * 1e6 * ytterfiber / treghetsmoment
            if spenning <= tillatt:
                valgt = (profil, treghetsmoment)
                break
        if valgt is None:
            raise ValueError(f"Ingen standardprofil tilfredsstiller gruppe {igruppe}")

        profil, treghetsmoment = valgt
        if profil[0].startswith("HEA"):
            _, h, b, tw, tf = profil
            resultat[igruppe, 1:5] = (tf, b, h - 2 * tf, tw)
        else:
            resultat[igruppe, 5] = profil[1]
            resultat[igruppe, 6] = profil[2]
        resultat[igruppe, 9] = flytspenning
        resultat[igruppe, 10] = treghetsmoment
        valg.append(profil[0])
    return resultat, valg


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
    # geom inneholder gruppe-ID og tverrsnittsdata, inkludert flytspenning.
    # tvsnitt inneholder E-modul og gruppe-ID per element. lastdata inneholder
    # lasttype, koordinater og laststørrelse, der lasttype 3 er trekantlast
    # med maksimum i startpunktet og null i sluttpunktet.
    
    try:
        with open(input_fil, "r", encoding="utf-8") as fil:
            linjer = iter(fil)

            npunkt = _antall(linjer, "knutepunkter")
            punkt = _tabell(_neste_linjer(linjer, npunkt, "knutepunkter"), 4, "knutepunkter",)

            ngeom = _antall(linjer, "tverrsnittsgeometrier")
            geomdata = _neste_linjer(linjer, ngeom, "tverrsnittsgeometrier")
            kolonner = len(geomdata[0]) if geomdata else 11
            if kolonner not in (10, 11) or any(len(rad) != kolonner for rad in geomdata):
                raise ValueError("Seksjonen 'tverrsnittsgeometrier' må ha 10 eller 11 kolonner per rad")
            geom = _tabell(geomdata, kolonner, "tverrsnittsgeometrier")
            if kolonner == 10:
                geom = np.insert(geom, 9, np.nan, axis=1)

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