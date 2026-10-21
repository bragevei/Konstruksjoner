
import numpy as np

def _lokal_posisjon(x, y, start, slutt, lengde):
    #Returnerer avstanden fra elementstart til punktet (x, y)
    vektor=slutt-start
    return float(np.dot(np.array([x, y])-start, vektor)/lengde)

def _kraftkoeffisienter(s, lengde):
    # Koeffisienter for en oppoverrettet kraftlast i avstand s fra
    # elementstart. Rotasjon er positiv medurs, derfor kommer fortegnet
    # motsatt av fortegnet for en positiv vertikal forskyvning.
    return (-(s - 2.0 * s**2 / lengde + s**3 / lengde**2), -(-s**2 / lengde + s**3 / lengde**2),)

def _momentkoeffisienter(s, lengde):
    #Koeffisienter for et ytre moment i avstand s fra elementstart.
    r = s / lengde
    return 1.0 - 4.0*r + 3.0*r**2, -2.0*r + 3.0*r**2


def _lastintegraler(a, b, lengde, startverdi, sluttverdi):
    # Integraler av en lineært varierende last mot bjelkefunksjonene
    if b <= a:
        return 0.0, 0.0

    lengde_last = b - a
    q_stigning = (sluttverdi - startverdi) / lengde_last
    q_konstant = startverdi - q_stigning * a

    def integral_maktall(makt):
        return (b ** (makt + 1) - a ** (makt + 1)) / (makt + 1)

    i1 = (q_konstant * (integral_maktall(1) - 2.0 * integral_maktall(2) / lengde + integral_maktall(3) / lengde**2)
        + q_stigning * (integral_maktall(2) - 2.0 * integral_maktall(3) / lengde + integral_maktall(4) / lengde**2))
    i2 = (q_konstant * (-integral_maktall(2) / lengde + integral_maktall(3) / lengde**2)
        + q_stigning * (-integral_maktall(3) / lengde+ integral_maktall(4) / lengde**2))
    return i1, i2

def elemlast(lastdata, punkt, elemkonn, toleranse=1e-8):
    # Beregner S_fim for alle elementer.

    # Parametere
    # ----------
    #   lastdata : array_like, shape (nlast, 6)
    #       Rader med lasttype, x_start, y_start, x_slutt, y_slutt, laststorrelse.
    #       Lasttype 0 er konstant fordelt last, 1 er punktlast,
    #       2 er et påført moment og 3 er trekantlast.
    #   punkt : array_like, shape (npunkt, 4)
    #       Knutepunktdata. De to første kolonnene er x- og y-koordinater.
    #   elemkonn : array_like, shape (nelem, 2)
    #       Elementenes start- og sluttknutepunkt.
    #   toleranse : float
    #       Tillatt avvik ved kontroll av at en last ligger på et element.

    # Returnerer
    # ----------
    #   numpy.ndarray, shape (nelem, 2)
    #       Fastinnspenningsmomentene ved elementstart og elements slutt.
    #       S_fim er motsatt fortegn av den ekvivalente lasten, slik at den kan adderes til systemlastvektoren.

    punkt=np.asarray(punkt, dtype=float)
    elemkonn=np.asarray(elemkonn, dtype=int)
    lastdata=np.asarray(lastdata, dtype=float)

    if lastdata.size==0:
        return np.zeros((len(elemkonn), 2), dtype=float)
    if lastdata.ndim !=2 or lastdata.shape[1] !=6:
        raise ValueError("Lastdata må ha seks kolonner")

    node_coordinates={int(node_id): coordinates[:2] for coordinates, node_id in zip(punkt, punkt[:, 3])}
    sfim=np.zeros((len(elemkonn), 2), dtype=float)

    for ilast, last in enumerate(lastdata):
        lasttype = int(last[0])
        if lasttype not in (0, 1, 2, 3) or lasttype != last[0]:
            raise ValueError(f"Ukjent lasttype {last[0]:g} på lastlinje {ilast + 1}")
        start_last = last[1:3]
        slutt_last = last[3:5]
        storrelse = last[5]
        lastvektor = slutt_last - start_last
        lastlengde = np.linalg.norm(lastvektor)
        if lasttype == 3 and lastlengde <= toleranse:
            raise ValueError(f"Trekantlast på lastlinje {ilast + 1} må ha ulikt start- og sluttpunkt")
        treff = []

        for ielem, (inode, jnode) in enumerate(elemkonn):
            try:
                start = node_coordinates[inode]
                slutt = node_coordinates[jnode]
            except KeyError as feil:
                raise ValueError(f"Element {ielem} refererer til ukjent knutepunkt {feil.args[0]}") from feil
            elementvektor = slutt - start
            lengde = np.linalg.norm(elementvektor)
            if lengde == 0:
                raise ValueError(f"Element {ielem} har lengde lik null")

            a = _lokal_posisjon(*start_last, start, slutt, lengde)
            punkt_paa_element = start + a * elementvektor / lengde
            avstand_start = np.linalg.norm(start_last - punkt_paa_element)

            if lasttype in (1, 2):
                if (avstand_start <= toleranse and -toleranse <= a <= lengde + toleranse):
                    treff.append((ielem, a, a, lengde))
                continue

            b = _lokal_posisjon(*slutt_last, start, slutt, lengde)
            punkt_paa_element = start + b * elementvektor / lengde
            if (np.linalg.norm(np.cross(elementvektor, start_last - start)) / lengde <= toleranse
                and np.linalg.norm(np.cross(elementvektor, slutt_last - start)) / lengde <= toleranse
                and max(0.0, min(a, b)) <= min(lengde, max(a, b)) + toleranse):
                    overlapp_start = max(0.0, min(a, b))
                    overlapp_slutt = min(lengde, max(a, b))
                    if lasttype == 3:
                        enhetsvektor = elementvektor / lengde
                        punkt_start = start + overlapp_start * enhetsvektor
                        punkt_slutt = start + overlapp_slutt * enhetsvektor
                        q_start = storrelse * (1.0 - np.dot(punkt_start - start_last, lastvektor) / lastlengde**2)
                        q_slutt = storrelse * (1.0 - np.dot(punkt_slutt - start_last, lastvektor) / lastlengde**2)
                        treff.append((ielem, overlapp_start, overlapp_slutt, lengde, q_start, q_slutt,))
                    else:
                        treff.append((ielem, overlapp_start, overlapp_slutt, lengde,))

        if (lasttype in (1, 2) and len(treff) > 1 and all(min(abs(a), abs(a - lengde)) <=
            toleranse for _, a, _, lengde in treff)):
            # En punktlast i et knutepunkt gir ikke fastinnspenningsmoment.
            continue

        if not treff or (lasttype in (1, 2) and len(treff) != 1):
            raise ValueError(
                f"Lastlinje {ilast + 1} må ligge på "
                f"{'nøyaktig ett element' if lasttype in (1, 2) else 'minst ett element'}; "
                f"fant {len(treff)}")

        for treffdata in treff:
            ielem, a, b, lengde = treffdata[:4]
            if lasttype == 0:
                # Direkte integrasjon av en konstant fordelt kraftlast.
                integral_start = ((b**2 - a**2) / 2.0 - 2.0 * (b**3 - a**3) / (3.0 * lengde) + (b**4 - a**4) / (4.0 * lengde**2))
                integral_slutt = (-(b**3 - a**3) / (3.0 * lengde) + (b**4 - a**4) / (4.0 * lengde**2))
                # Positiv kraft er oppover, mens rotasjon er positiv medurs.
                ekvivalent = -storrelse * np.array([integral_start, integral_slutt])
            elif lasttype == 3:
                integral_start, integral_slutt = _lastintegraler(a, b, lengde, treffdata[4], treffdata[5])
                ekvivalent = -np.array([integral_start, integral_slutt])
            elif lasttype == 1:
                koeff_start, koeff_slutt = _kraftkoeffisienter(a, lengde)
                ekvivalent = storrelse * np.array([koeff_start, koeff_slutt])
            else:
                koeff_start, koeff_slutt = _momentkoeffisienter(a, lengde)
                ekvivalent = storrelse * np.array([koeff_start, koeff_slutt])

            sfim[ielem] -= ekvivalent
    return sfim


def elemlast_til_syslast(R, S_fim, elemkonn):
    # Assemblerer elementlastvektoren inn i systemlastvektoren.

    # S_fim[ielem, 0] hører til elementets startknutepunkt, og S_fim[ielem, 1] hører til elementets sluttknutepunkt.
    # Koblingen mellom lokale elementfrihetsgrader og systemfrihetsgrader bestemmes av elemkonn.

    # Parametere
    # ----------
    # R : array_like, shape (npunkt,)
    #     Systemlastvektor før elementbidragene er lagt til.
    # S_fim : array_like, shape (nelem, 2)
    #     Fastinnspenningsmomenter for hvert element.
    # elemkonn : array_like, shape (nelem, 2)
    #     Knutepunktindeks for elementets start og slutt.

    # Returnerer
    # ----------
    # numpy.ndarray, shape (npunkt,)
    # Oppdatert systemlastvektor. Inputvektoren R endres ikke.
    
    R = np.asarray(R, dtype=float)
    S_fim = np.asarray(S_fim, dtype=float)
    elemkonn = np.asarray(elemkonn, dtype=int)

    if R.ndim != 1:
        raise ValueError("R må være en endimensjonal systemlastvektor")
    if S_fim.ndim != 2 or S_fim.shape[1] != 2:
        raise ValueError("S_fim må ha formen (antall elementer, 2)")
    if elemkonn.ndim != 2 or elemkonn.shape[1] != 2:
        raise ValueError("elemkonn må ha formen (antall elementer, 2)")
    if S_fim.shape[0] != elemkonn.shape[0]:
        raise ValueError("S_fim og elemkonn må ha samme antall elementer")
    if np.any(elemkonn < 0) or np.any(elemkonn >= len(R)):
        raise ValueError("elemkonn inneholder en knutepunktindeks utenfor R")

    systemlast = R.copy()
    for (start, slutt), (moment_start, moment_slutt) in zip(elemkonn, S_fim):
        systemlast[start] += moment_start
        systemlast[slutt] += moment_slutt

    return systemlast


def knutmom(R, lastdata, punkt, toleranse=1e-8):
    # Adderer ytre knutepunktsmomenter til systemlastvektoren.

    # Lastdata med lasttype == 2 tolkes som et moment på knutepunktet angitt av x_start og y_start. Momentets størrelse og
    # fortegn hentes fra kolonnen laststorrelse. Laster av andre typer ignoreres, siden de behandles av funksjonen elemlast.

    # Parametere
    # ----------
    # R : array_like, shape (npunkt,)
    #     Systemlastvektor før knutepunktsmomentene legges til.
    # lastdata : array_like, shape (nlast, 6)
    #     Lastdata i formatet fra inputfilen.
    # punkt : array_like, shape (npunkt, 4)
    #     Knutepunktdata. De to første kolonnene er x- og y-koordinater.
    # toleranse : float
    #     Tillatt avstand mellom lastens koordinater og knutepunktets
    #     koordinater.

    # Returnerer
    # ----------
    # numpy.ndarray, shape (npunkt,)
    #     Oppdatert systemlastvektor. Inputvektoren R endres ikke

    R = np.asarray(R, dtype=float)
    lastdata = np.asarray(lastdata, dtype=float)
    punkt = np.asarray(punkt, dtype=float)

    if R.ndim != 1:
        raise ValueError("R må være en endimensjonal systemlastvektor")
    if punkt.ndim != 2 or punkt.shape[1] < 2:
        raise ValueError("punkt må ha minst to kolonner")
    if len(R) != len(punkt):
        raise ValueError("R må ha én komponent per knutepunkt")
    if lastdata.size == 0:
        return R.copy()
    if lastdata.ndim != 2 or lastdata.shape[1] != 6:
        raise ValueError("lastdata må ha seks kolonner")

    systemlast = R.copy()
    for ilast, last in enumerate(lastdata):
        if int(last[0]) != 2:
            continue

        koordinat = last[1:3]
        avstander = np.linalg.norm(punkt[:, :2] - koordinat, axis=1)
        treff = np.flatnonzero(avstander <= toleranse)
        if len(treff) != 1:
            raise ValueError(f"Knutepunktsmoment på lastlinje {ilast + 1} må treffe "f"nøyaktig ett knutepunkt; fant {len(treff)}")
        systemlast[treff[0]] += last[5]

    return systemlast