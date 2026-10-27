"""
lesinput.py - leser inputfilen som beskriver rammen
====================================================

OVERORDNET TENKEMÅTE
--------------------
All informasjon om konstruksjonen (geometri, opplager, materialer, tverrsnitt,
elementer og laster) ligger i en egen tekstfil, slik at samme program kan
analysere en hvilken som helst plan, uforskyvelig ramme uten at koden endres.
Denne modulen gjør om tekstfilen til numpy-tabeller som resten av programmet
bruker.

Inputfilen består av fem seksjoner som alltid kommer i denne rekkefølgen:

    1. KNUTEPUNKTER   2. MATERIALER   3. TVERRSNITT   4. ELEMENTER   5. LASTER

Hver seksjon starter med ett heltall som sier hvor mange datalinjer som
følger, og deretter kommer én linje per knutepunkt/material/tverrsnitt/
element/last. Fordi antallet leses fra filen, kan programmet håndtere et
vilkårlig antall av hver type. Alt etter tegnet '#' er kommentar, og tomme
linjer hoppes over.

Kolonner i hver seksjon (enheter i hakeparentes):

    KNUTEPUNKTER : nr  x[m]  y[m]  rot
                   rot = 0: rotasjonen er fastholdt (fast innspent)
                   rot = 1: rotasjonen er fri (ledd eller vanlig rammehjørne)
    MATERIALER   : nr  E[MPa]  fy[MPa]
    TVERRSNITT   : nr  type  D  t  h  b  tf  tw  I
                   type 0 = rørprofil: D = ytre diameter, t = veggtykkelse [mm]
                   type 1 = I-profil : h = total høyde, b = flensbredde,
                                       tf = flenstykkelse, tw = stegtykkelse [mm]
                   Mål som ikke brukes av profiltypen settes lik 0.
                   I [mm^4]: settes normalt lik 0, og da beregnes 2. arealmoment
                   fra målene. Er I > 0 brukes den oppgitte verdien direkte
                   (nyttig i oppgave b der stivhetene er gitt som 10I, 5I, ...).
    ELEMENTER    : nr  knutepunkt1  knutepunkt2  tverrsnittnr  materialnr
                   Elementet kobles til riktig geometri og material via
                   tverrsnittnummeret og materialnummeret.
    LASTER       : type  x1  y1  x2  y2  verdi1  verdi2
                   type 0 = fordelt last [kN/m] som varierer lineært fra verdi1
                            i punktet (x1,y1) til verdi2 i punktet (x2,y2).
                            Jevnt fordelt last: verdi1 = verdi2.
                            Lasten må starte og slutte i knutepunkter, men kan
                            gå over flere elementer som ligger på linje.
                   type 1 = punktlast [kN] med størrelse verdi1 i punktet (x1,y1).
                            Punktet kan ligge hvor som helst på et element.
                   type 2 = ytre knutepunktsmoment [kNm] med størrelse verdi1
                            i knutepunktet (x1,y1). Positivt med klokka.
                   Kolonner som ikke brukes av lasttypen settes lik 0.
                   Fortegn for laster av type 0 og 1: positiv last virker i
                   elementets lokale y-retning, dvs. 90 grader MOT klokka fra
                   elementets retning (knutepunkt1 -> knutepunkt2). For en
                   bjelke definert fra venstre mot høyre er positiv last
                   altså rettet OPPOVER, og for en søyle definert nedenfra og
                   opp er positiv last rettet mot VENSTRE.

Nummeret i første kolonne (nr) må være 0, 1, 2, ... i rekkefølge. Python
teller fra 0, så nummeret er samtidig radindeksen som brukes i resten av
programmet. Nummeret står i filen for at den skal være lett å lese og
kontrollere mot skissen av diskretiseringen.

Alle tabellene beholder samme kolonnerekkefølge som i inputfilen. For at koden
skal være lett å lese, brukes navngitte kolonneindekser (P_X, G_TYPE, osv.)
i stedet for "magiske" tall. Disse importeres av de andre modulene.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Navngitte kolonneindekser for tabellene som returneres av lesinput()
# ---------------------------------------------------------------------------
# punkt: nr, x, y, rot
P_X, P_Y, P_ROT = 1, 2, 3
# mat: nr, E, fy
M_E, M_FY = 1, 2
# geom: nr, type, D, t, h, b, tf, tw, I
G_TYPE, G_D, G_T, G_H, G_B, G_TF, G_TW, G_I = 1, 2, 3, 4, 5, 6, 7, 8
# elem: nr, knutepunkt1, knutepunkt2, tverrsnittnr, materialnr
E_N1, E_N2, E_GEOM, E_MAT = 1, 2, 3, 4
# last: type, x1, y1, x2, y2, verdi1, verdi2
L_TYPE, L_X1, L_Y1, L_X2, L_Y2, L_V1, L_V2 = 0, 1, 2, 3, 4, 5, 6

# Profiltyper og lasttyper (samme tall som i inputfilen)
ROR, IPROFIL = 0, 1
FORDELT, PUNKT, MOMENT = 0, 1, 2


def _les_datalinjer(filnavn):
    """Leser filen og returnerer en liste med datalinjer, der hver datalinje
    er en liste med tekstbiter (tall). Kommentarer og tomme linjer fjernes."""
    datalinjer = []
    with open(filnavn, "r", encoding="utf-8") as fil:
        for linjenr, linje in enumerate(fil, start=1):
            tekst = linje.split("#", 1)[0]        # fjern alt etter '#'
            biter = tekst.split()                 # del opp på mellomrom/tab
            if biter:                             # hopp over tomme linjer
                datalinjer.append((linjenr, biter))
    return datalinjer


def lesinput(filnavn):
    """
    Leser inputfilen og returnerer modellen som numpy-tabeller.

    Returnerer (i samme rekkefølge som seksjonene i filen):
        npunkt, punkt : antall knutepunkter og tabell (npunkt x 4)
        nmat, mat     : antall materialer og tabell (nmat x 3)
        ngeom, geom   : antall tverrsnitt og tabell (ngeom x 9)
        nelem, elem   : antall elementer og heltallstabell (nelem x 5)
        nlast, last   : antall laster og tabell (nlast x 7)
    """
    datalinjer = _les_datalinjer(filnavn)
    posisjon = 0   # hvilken datalinje vi har kommet til

    def les_seksjon(navn, nkolonner, har_nummer=True):
        """Leser én seksjon: først antallet, deretter like mange datalinjer.
        Kontrollerer antall kolonner, at alt er tall og at nr = 0, 1, 2, ..."""
        nonlocal posisjon
        if posisjon >= len(datalinjer):
            raise ValueError(f"Inputfilen slutter før seksjonen '{navn}'.")
        linjenr, biter = datalinjer[posisjon]
        try:
            antall = int(biter[0])
        except ValueError:
            raise ValueError(f"Linje {linjenr}: forventet antall {navn}, fant '{biter[0]}'")
        posisjon += 1

        rader = []
        for _ in range(antall):
            if posisjon >= len(datalinjer):
                raise ValueError(f"Inputfilen mangler datalinjer i seksjonen '{navn}'")
            linjenr, biter = datalinjer[posisjon]
            if len(biter) != nkolonner:
                raise ValueError(f"Linje {linjenr} ({navn}): forventet {nkolonner} "
                                 f"tall, fant {len(biter)}.")
            try:
                rader.append([float(b) for b in biter])
            except ValueError:
                raise ValueError(f"Linje {linjenr} ({navn}): inneholder noe som ikke er et tall")
            posisjon += 1

        tabell = np.array(rader, dtype=float).reshape(antall, nkolonner)
        # Nummeret i første kolonne skal være 0, 1, 2, ... slik at det er lik radindeksen
        if har_nummer and not np.array_equal(tabell[:, 0], np.arange(antall)):
            raise ValueError(f"Seksjonen '{navn}' må nummereres 0, 1, 2, ... i rekkefølge")
        return antall, tabell

    # --- De fem seksjonene leses i fast rekkefølge -------------------------
    npunkt, punkt = les_seksjon("knutepunkter", 4)
    nmat, mat = les_seksjon("materialer", 3)
    ngeom, geom = les_seksjon("tverrsnitt", 9)
    nelem, elem = les_seksjon("elementer", 5)
    nlast, last = les_seksjon("laster", 7, har_nummer=False)

    if posisjon != len(datalinjer):
        linjenr = datalinjer[posisjon][0]
        raise ValueError(f"Linje {linjenr}: inputfilen inneholder flere datalinjer enn "f"antallene i seksjonene tilsier")

    # --- Kontroller som fanger vanlige tastefeil i inputfilen ---------------
    if not np.all(np.isin(punkt[:, P_ROT], (0, 1))):
        raise ValueError("Rotasjonsbetingelsen for knutepunkter må være 0 (fast) eller 1 (fri)")
    if not np.all(np.isin(geom[:, G_TYPE], (ROR, IPROFIL))):
        raise ValueError("Tverrsnittstypen må være 0 (rør) eller 1 (I-profil)")
    if nlast and not np.all(np.isin(last[:, L_TYPE], (FORDELT, PUNKT, MOMENT))):
        raise ValueError("Lasttypen må være 0 (fordelt), 1 (punktlast) eller 2 (moment)")

    # Elementtabellen inneholder bare heltall (nummer), så den lagres som int.
    # Da kan den brukes direkte som indekser i de andre tabellene.
    if not np.all(elem == np.round(elem)):
        raise ValueError("Elementtabellen kan bare inneholde heltall")
    elem = elem.astype(int)
    for kolonne, antall, navn in ((E_N1, npunkt, "knutepunkt"), (E_N2, npunkt, "knutepunkt"),
                                  (E_GEOM, ngeom, "tverrsnitt"), (E_MAT, nmat, "material")):
        if np.any(elem[:, kolonne] < 0) or np.any(elem[:, kolonne] >= antall):
            raise ValueError(f"Et element refererer til et {navn}nummer som ikke finnes")

    return npunkt, punkt, nmat, mat, ngeom, geom, nelem, elem, nlast, last
