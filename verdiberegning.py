import numpy as np

def _elementdata(punkt, elemkonn):
    #Returnerer koordinater og lengde for hvert element
    punkt = np.asarray(punkt, dtype=float)
    elemkonn = np.asarray(elemkonn, dtype=int)
    if punkt.ndim != 2 or punkt.shape[1] < 4:
        raise ValueError("punkt må ha minst fire kolonner")
    if elemkonn.ndim != 2 or elemkonn.shape[1] != 2:
        raise ValueError("elemkonn må ha formen (antall elementer, 2)")
    koordinater = {int(rad[3]): rad[:2] for rad in punkt}
    data = []

    for ielem, (start_id, slutt_id) in enumerate(elemkonn):
        try:
            start = koordinater[start_id]
            slutt = koordinater[slutt_id]
        except KeyError as feil:
            raise ValueError(f"Element {ielem} refererer til ukjent knutepunkt {feil.args[0]}") from feil
        vektor = slutt - start
        lengde = np.linalg.norm(vektor)
        if lengde == 0:
            raise ValueError(f"Element {ielem} har lengde lik null")
        data.append((start, slutt, lengde))
    return data


def _lokal_posisjon(koordinat, start, slutt, lengde, toleranse):
    vektor = slutt - start
    s = float(np.dot(koordinat - start, vektor) / lengde)
    avstand = np.linalg.norm(koordinat - (start + s * vektor / lengde))
    if avstand <= toleranse and -toleranse <= s <= lengde + toleranse:
        return min(max(s, 0.0), lengde)
    return None


def _laster_for_element(lastdata, start, slutt, lengde, toleranse):
    # Finner punktlast, fordelt last eller trekantlast som virker på elementet.
    laster = []
    for last in np.asarray(lastdata, dtype=float):
        lasttype = int(last[0])
        if lasttype not in (0, 1, 2, 3) or lasttype != last[0]:
            raise ValueError(f"Ukjent lasttype {last[0]:g}")
        if lasttype == 2:
            continue
        a = _lokal_posisjon(last[1:3], start, slutt, lengde, toleranse)
        b = _lokal_posisjon(last[3:5], start, slutt, lengde, toleranse)

        if lasttype == 1 and a is not None:
            if min(a, lengde - a) > toleranse:
                laster.append(("punkt", a, last[5]))
        elif lasttype in (0, 3):
            elementvektor = slutt - start
            a_lokal = float(np.dot(last[1:3] - start, elementvektor) / lengde)
            b_lokal = float(np.dot(last[3:5] - start, elementvektor) / lengde)
            avstand_a = np.linalg.norm(last[1:3] - (start + a_lokal * elementvektor / lengde))
            avstand_b = np.linalg.norm(last[3:5] - (start + b_lokal * elementvektor / lengde))
            if avstand_a <= toleranse and avstand_b <= toleranse:
                overlap_start = max(0.0, min(a_lokal, b_lokal))
                overlap_slutt = min(lengde, max(a_lokal, b_lokal))
                if overlap_slutt - overlap_start > toleranse:
                    if lasttype == 0:
                        laster.append(("fordelt", overlap_start, overlap_slutt, last[5]))
                    else:
                        lastvektor = last[3:5] - last[1:3]
                        lastlengde = np.linalg.norm(lastvektor)
                        if lastlengde <= toleranse:
                            raise ValueError("Trekantlast må ha ulikt start- og sluttpunkt")
                        enhetsvektor = elementvektor / lengde
                        punkt_start = start + overlap_start * enhetsvektor
                        punkt_slutt = start + overlap_slutt * enhetsvektor
                        q_start = last[5] * (1.0 - np.dot(punkt_start - last[1:3], lastvektor)/ lastlengde**2)
                        q_slutt = last[5] * (1.0 - np.dot(punkt_slutt - last[1:3], lastvektor)/ lastlengde**2)
                        laster.append(("trekant", overlap_start, overlap_slutt, q_start, q_slutt,))

    if any(last[0] == "punkt" for last in laster) and any(last[0] in ("fordelt", "trekant") for last in laster):
        raise ValueError("Et element kan ikke ha både punktlast og fordelt last")
    if sum(last[0] == "punkt" for last in laster) > 1:
        raise ValueError("Verdiberegning støtter én punktlast eller én fordelt last per element")
    return laster


def _lastmoment(posisjoner, last, lengde):
    #Momentbidrag fra ytre last med fastholdte elementender
    if not last:
        return np.zeros(len(posisjoner)), np.zeros(2)

    moment = np.zeros(len(posisjoner))
    skjær = np.zeros(2)
    for lastdata in last:
        if lastdata[0] == "punkt":
            _, a, kraft = lastdata
            b = lengde - a
            # Kraft er positiv oppover og moment positivt medurs.
            venstremoment = kraft * a * b**2 / lengde**2
            venstreskjær = -kraft * b**2 * (3 * a + b) / lengde**3
            moment += np.where(
                posisjoner <= a,
                venstremoment + venstreskjær * posisjoner,
                venstremoment + venstreskjær * posisjoner
                + kraft * (posisjoner - a),
            )
            skjær += np.array([venstreskjær, venstreskjær + kraft])
        else:
            if lastdata[0] == "fordelt":
                _, a, b, verdi_start = lastdata
                verdi_slutt = verdi_start
            else:
                _, a, b, verdi_start, verdi_slutt = lastdata
            lengde_last = b - a
            q_start, q_slutt = verdi_start, verdi_slutt
            q_stigning = (q_slutt - q_start) / lengde_last
            q_konstant = q_start - q_stigning * a

            def integral_maktall(makt):
                return (b ** (makt + 1) - a ** (makt + 1)) / (makt + 1)

            integral_start = (
                q_konstant * (
                    integral_maktall(1)
                    - 2 * integral_maktall(2) / lengde
                    + integral_maktall(3) / lengde**2
                )
                + q_stigning * (
                    integral_maktall(2)
                    - 2 * integral_maktall(3) / lengde
                    + integral_maktall(4) / lengde**2
                )
            )
            integral_slutt = (
                q_konstant * (
                    -integral_maktall(2) / lengde
                    + integral_maktall(3) / lengde**2
                )
                + q_stigning * (
                    -integral_maktall(3) / lengde
                    + integral_maktall(4) / lengde**2
                )
            )
            moment_start = integral_start
            moment_slutt = -integral_slutt
            lastmoment_om_start = (
                q_konstant * (lengde_last * lengde - integral_maktall(1))
                + q_stigning
                * (lengde * integral_maktall(1) - integral_maktall(2))
            )
            venstreskjær = (
                moment_slutt - moment_start
                - lastmoment_om_start
            ) / lengde

            def lastmoment_i_punkt(x):
                if x <= a:
                    return 0.0
                slutt = min(x, b)
                lastintegral = (
                    q_konstant * (
                        (slutt - a) * x - (slutt**2 - a**2) / 2
                    )
                    + q_stigning * (
                        (slutt**2 - a**2) * x / 2
                        - (slutt**3 - a**3) / 3
                    )
                )
                return (
                    lastintegral
                )

            moment += (
                moment_start
                + venstreskjær * posisjoner
                + np.array([lastmoment_i_punkt(x) for x in posisjoner])
            )
            skjær += np.array(
                [
                    venstreskjær,
                    venstreskjær + q_start * lengde_last
                    + 0.5 * (q_slutt - q_start) * lengde_last,
                ]
            )
    return moment, skjær


def moment(rot, punkt, elemkonn, EI, lastdata, toleranse=1e-8):
    # Beregner totalt moment ved tre posisjoner i hvert element

    # Kolonnene er elementstart, lastposisjon/midtpunkt og elements slutt. Totalmomentet er summen av momentet fra 
    # enderotasjonene og momentet fra ytre last
    
    rot = np.asarray(rot, dtype=float)
    elemkonn = np.asarray(elemkonn, dtype=int)
    EI = np.asarray(EI, dtype=float)
    elementer = _elementdata(punkt, elemkonn)
    if len(rot) != len(np.asarray(punkt)):
        raise ValueError("rot må ha én verdi per knutepunkt")
    if len(EI) != len(elemkonn):
        raise ValueError("EI må ha én verdi per element")

    Mval = np.zeros((len(elemkonn), 3))
    for ielem, ((_, _, lengde), (start_id, slutt_id)) in enumerate(
        zip(elementer, elemkonn)
    ):
        last = _laster_for_element(lastdata, elementer[ielem][0], elementer[ielem][1], lengde, toleranse,)
        punktlaster = [lastdata for lastdata in last if lastdata[0] == "punkt"]
        if punktlaster:
            posisjoner = np.array([0.0, punktlaster[0][1], lengde])
        else:
            posisjoner = np.array([0.0, lengde / 2, lengde])

        faktor = EI[ielem] / lengde
        endemomenter = faktor * np.array(
            [
                4 * rot[start_id] + 2 * rot[slutt_id],
                2 * rot[start_id] + 4 * rot[slutt_id],
            ])
        rotasjonsmoment = np.linspace(
            endemomenter[0], -endemomenter[1], num=3
        )
        lastmoment, _ = _lastmoment(posisjoner, last, lengde)
        print(lastmoment)
        Mval[ielem] = rotasjonsmoment + lastmoment
    return Mval


def shear(rot, punkt, elemkonn, EI, lastdata, toleranse=1e-8):
    # Beregner skjærkraft ved start og slutt av hvert element
    # Bidraget fra enderotasjonene finnes som dM/ds. Bidraget fra ytre last legges deretter til ved superposisjon
    
    rot = np.asarray(rot, dtype=float)
    elemkonn = np.asarray(elemkonn, dtype=int)
    EI = np.asarray(EI, dtype=float)
    elementer = _elementdata(punkt, elemkonn)
    if len(EI) != len(elemkonn):
        raise ValueError("EI må ha én verdi per element")

    Qval = np.zeros((len(elemkonn), 2))
    for ielem, ((start, slutt, lengde), (start_id, slutt_id)) in enumerate(
        zip(elementer, elemkonn)
    ):
        faktor = EI[ielem] / lengde
        Mstart = faktor * (4 * rot[start_id] + 2 * rot[slutt_id])
        Mslutt = faktor * (2 * rot[start_id] + 4 * rot[slutt_id])
        Qrotasjon = (-Mslutt - Mstart) / lengde
        last = _laster_for_element(lastdata, start, slutt, lengde, toleranse)
        _, Qlast = _lastmoment(
            np.array([0.0, lengde]), last, lengde
        )
        Qval[ielem] = Qrotasjon + Qlast
    return Qval


def boyespenn(Mval, geom, tvsnitt):
    # Beregner bøyespenning fra momentverdiene
    # Bruker sigma = M*c/I. Resultatet har samme form som Mval
    
    Mval = np.asarray(Mval, dtype=float)
    geom = np.asarray(geom, dtype=float)
    tvsnitt = np.asarray(tvsnitt, dtype=int)
    if Mval.ndim != 2 or Mval.shape[1] != 3:
        raise ValueError("Mval må ha formen (antall elementer, 3)")
    if len(tvsnitt) != len(Mval):
        raise ValueError("tvsnitt må ha én gruppe-ID per element")
    if geom.ndim != 2 or geom.shape[1] != 10:
        raise ValueError("geom må ha nøyaktig ti kolonner")
    if np.any(tvsnitt < 0) or np.any(tvsnitt >= len(geom)):
        raise ValueError("tvsnitt inneholder en ugyldig gruppe-ID")

    spenning = np.zeros_like(Mval)
    for ielem, gruppe in enumerate(tvsnitt):
        profiltype = int(geom[gruppe, 7])
        flenshoyde, flensbredde, steghoyde, stegtykkelse = geom[gruppe, 1:5]
        radius, tykkelse = geom[gruppe, 5:7]
        oppgitt_I = geom[gruppe, 9]
        if profiltype == 0:
            treghetsmoment = (
                oppgitt_I if oppgitt_I > 0
                else np.pi * radius**3 * tykkelse
            )
            ytterfiber = radius
        elif profiltype == 1:
            treghetsmoment = oppgitt_I if oppgitt_I > 0 else (
                2 * (
                    flensbredde * flenshoyde**3 / 12
                    + flensbredde * flenshoyde
                    * (steghoyde / 2 + flenshoyde / 2) ** 2
                )
                + stegtykkelse * steghoyde**3 / 12
            )
            ytterfiber = steghoyde / 2 + flenshoyde
        else:
            raise ValueError(f"Ukjent tverrsnittstype {profiltype}")
        if treghetsmoment <= 0:
            raise ValueError(f"Gruppe {gruppe} har ugyldig treghetsmoment")
        spenning[ielem] = Mval[ielem] * ytterfiber / treghetsmoment
    return spenning