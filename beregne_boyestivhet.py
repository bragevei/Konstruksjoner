#hvis geom = 0 så er det en O-profil, hvis geom = 1 så er det en I-profil
#antar tynnvegget  O-profil.
import numpy as np
def beregne_boyestivhet(geom, tvsnitt):
    # Beregner ett EI-tall for hvert element.

    # geom inneholder én rad per tverrsnittsgruppe, mens tvsnitt inneholder gruppe-ID-en til hvert element. 
    # Derfor beregnes først bøyestivheten for gruppene, og deretter slås riktig gruppe opp for hvert element.

    geom = np.asarray(geom, dtype=float)
    tvsnitt = np.asarray(tvsnitt, dtype=int)

    if geom.ndim != 2 or geom.shape[1] < 7:
        raise ValueError("geom må ha minst sju kolonner.")
    if tvsnitt.ndim != 1:
        raise ValueError("tvsnitt må inneholde én gruppe-ID per element.")
    if np.any(tvsnitt < 0) or np.any(tvsnitt >= len(geom)):
        raise ValueError("tvsnitt inneholder en ugyldig gruppe-ID.")

    flensbredde = geom[:, 1]
    steghoyde = geom[:, 2]
    radius = geom[:, 3]
    tykkelse = geom[:, 4]
    profiltype = geom[:, 5].astype(int)
    elastisitetsmodul = geom[:, 6]

    treghetsmoment = np.zeros(len(geom), dtype=float)
    for igruppe, type_ in enumerate(profiltype):
        if type_ == 1:
            treghetsmoment[igruppe]=(flensbredde[igruppe]*steghoyde[igruppe]**3/12 +
                2*(flensbredde[igruppe]*tykkelse[igruppe]**3/12 + flensbredde[igruppe] 
                *tykkelse[igruppe]*(steghoyde[igruppe]/2 + tykkelse[igruppe]/2)**2))
            
        elif type_ == 0:
            treghetsmoment[igruppe] = (np.pi * radius[igruppe] ** 3 * tykkelse[igruppe])
        else:
            raise ValueError(f"Ukjent tverrsnittstype {type_} i gruppe {igruppe}")

    EI_grupper = elastisitetsmodul * treghetsmoment
    return EI_grupper[tvsnitt]