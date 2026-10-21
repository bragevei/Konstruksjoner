import numpy as np


def _beregn_treghetsmoment(rad, igruppe):
    # Beregner I fra geometrien i en rad i geom
    flenshoyde = rad[1]
    flensbredde = rad[2]
    steghoyde = rad[3]
    stegtykkelse = rad[4]
    radius = rad[5]
    tykkelse = rad[6]
    profiltype = int(rad[7])
    
    if profiltype == 0:
        if radius <= 0 or tykkelse <= 0:
            raise ValueError(f"Gruppe {igruppe} mangler positiv radius og tykkelse for rørprofil")
        # Rørprofil
        return (np.pi*((2*radius)**4-(2*radius-2*tykkelse)**4))/64

    if profiltype == 1:
        if min(flenshoyde, flensbredde, steghoyde, stegtykkelse) <= 0:
            raise ValueError(f"Gruppe {igruppe} mangler positive mål for I-profil")
        # To flenser rundt nøytralaksen og et steg mellom dem
        flenser = 2 * (flensbredde * flenshoyde**3 / 12 + flensbredde*flenshoyde * (steghoyde / 2 + flenshoyde / 2) ** 2)
        steg = stegtykkelse * steghoyde**3 / 12
        return flenser + steg

    raise ValueError(f"Ukjent tverrsnittstype {profiltype} i gruppe {igruppe}; ""bruk 0 for rørprofil eller 1 for I-profil")

def beregne_boyestivhet(geom, tvsnitt):
    # Beregner EI i kN*m^2 for hvert element.

    # geom skal ha kolonnene:
    # gruppeID, flenshøyde, flensbredde, steghøyde, stegtykkelse,
    # radius, tykkelse, profiltype, E [MPa], I-verdi [mm^4]

    #Dersom I-verdi er større enn null, brukes denne direkte. Dersom verdien er null, beregnes treghetsmomentet fra geometrien. 
    # tvsnitt inneholder gruppe-ID-en for hvert element.
    
    geom = np.asarray(geom, dtype=float)
    tvsnitt_raw = np.asarray(tvsnitt)
    tvsnitt = tvsnitt_raw.astype(int)

    if geom.ndim != 2 or geom.shape[1] != 10:
        raise ValueError("geom må ha nøyaktig ti kolonner i det nye tverrsnittsformatet")
    if tvsnitt.ndim != 1:
        raise ValueError("tvsnitt må inneholde én gruppe-ID per element")
    if not np.all(tvsnitt_raw == tvsnitt):
        raise ValueError("tvsnitt må inneholde heltallige gruppe-ID-er")
    if np.any(tvsnitt < 0) or np.any(tvsnitt >= len(geom)):
        raise ValueError("tvsnitt inneholder en ugyldig gruppe-ID")

    elastisitetsmodul = geom[:, 8]
    oppgitt_I = geom[:, 9]
    if np.any(elastisitetsmodul <= 0):
        raise ValueError("Elastisitetsmodulen E må være positiv")
    if np.any(oppgitt_I < 0):
        raise ValueError("Oppgitt treghetsmoment I kan ikke være negativt")

    treghetsmoment = np.empty(len(geom), dtype=float)
    for igruppe, rad in enumerate(geom):
        if oppgitt_I[igruppe] > 0:
            treghetsmoment[igruppe] = oppgitt_I[igruppe]
        else:
            treghetsmoment[igruppe] = _beregn_treghetsmoment(rad, igruppe)

    # E [MPa] = N/mm^2. Konverter E til kN/m^2 og I til m^4.
    elastisitetsmodul_kN_m2 = elastisitetsmodul * 1e3
    treghetsmoment_m4 = treghetsmoment * 1e-12
    EI_grupper = elastisitetsmodul_kN_m2 * treghetsmoment_m4
    return EI_grupper[tvsnitt]