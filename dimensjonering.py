"""
dimensjonering.py - valg av standardprofiler (oppgave c)
========================================================

KRAVET
------
For hver elementgruppe (alle elementer med samme tverrsnittnummer) skal den
største bøyespenningen i gruppen ikke overstige 70 % av flytspenningen, kontrollert
ved begge elementender, under punktlaster og midt på elementer med fordelt
last. For rør skal i tillegg D/t < 30.

HVORFOR ITERASJON
-----------------
Rammen er statisk ubestemt: momentfordelingen avhenger av forholdet mellom
stivhetene. Gjør vi en søyle stivere, trekker den til seg mer moment fra
bjelkene, og omvendt. Når ett tverrsnitt endres, endres altså momentene i hele
rammen, og alle grupper må kontrolleres på nytt.

HVORFOR IKKE BARE VELGE DET MINSTE PROFILET SOM HOLDER?
------------------------------------------------------
Et naturlig første forsøk er å gå gjennom gruppene én og én og velge det
første profilet i en liste (fra minst til størst) som tilfredsstiller kravet.
Det gir to uheldige effekter:

  1) For en søyle som er slakk i forhold til bjelkene, er momentet omtrent
     proporsjonalt med I, slik at spenningen sigma = M*c/I blir omtrent
     proporsjonal med c. Jo mindre røret er, jo lavere blir spenningen. Et søk
     som starter på det minste profilet finner derfor alltid et bitte lite rør
     (f.eks. 21,3 x 2,3 mm) som "består" fordi det er så slakt at det nesten
     ikke tar moment. Da må resten av rammen bære alt, og resultatet blir én
     massiv søyle og overdimensjonerte bjelker.
  2) Hvis listen er sortert på diameter og ikke på vekt, blir "første rør som
     holder" et lite rør med svært tykk vegg (f.eks. 273 x 40), som er et lite
     effektivt tverrsnitt. Et rør med større diameter og tynnere vegg har mye
     større I og W per kg stål. Her velges derfor alltid det LETTESTE profilet.

METODEN SOM BRUKES HER: FULLT UTNYTTET DIMENSJONERING ("fully stressed design")
------------------------------------------------------------------------------
  1. Start konservativt med det stiveste profilet i hver gruppe. Da er alle
     søylene og bjelkene stive nok til å ta sin naturlige andel av momentene.
  2. Analyser rammen med de nåværende profilene.
  3. For hver gruppe: finn største |M| i gruppen, regn ut nødvendig
     motstandsmoment  W_krav = |M|maks / (0,7*fy), og velg det LETTESTE
     profilet (minst areal = minst vekt per meter) med W >= W_krav.
     Alle grupper oppdateres SAMTIDIG ut fra den samme analysen.
  4. Gjenta 2-3 til ingen profiler endres. Da er hvert profil valgt ut fra
     momentene i akkurat denne rammen, og kravet er oppfylt i alle grupper.
  Hvis iterasjonen går i ring mellom noen få profiler (kan skje fordi
  profilene er diskrete), tillates bare økning av profilene videre. Da må
  iterasjonen stoppe, og resultatet blir det minste profilsettet i ringen
  som tilfredsstiller kravet.

Fordi vi starter med stive profiler og bare reduserer et profil når det
virkelig har overskuddskapasitet, får alle søylene en realistisk andel av
lasten, i stedet for at én søyle bærer nesten alt.

PRAKTISK MINSTEKRAV: SLANKHET
-----------------------------
En søyle som bare får et lite moment (her: høyre søyle, som bare holder
enden av den lett belastede bjelken ved A) vil alltid "bestå" spenningskravet,
uansett hvor liten den er. Oppgaven sier at knekking ikke skal kontrolleres,
men et 21 mm rør som 12 m høy søyle er ikke en fornuftig dimensjon. Derfor
kan det legges inn et praktisk krav til slankheten lambda = L/i <= lambda_maks,
der i = sqrt(I/A) er treghetsradien og L den lengste elementlengden i gruppen.
lambda_maks = 200 er en vanlig praktisk øvre grense for trykkstaver. Dette er
IKKE en knekkontroll, bare en nedre grense for hvor slanke profiler som
vurderes. Sett lambda_maks = None for å slå kravet av.
"""

from collections import namedtuple

import numpy as np

from beregne_boyestivhet import tverrsnittsdata
from lesinput import (G_TYPE, G_D, G_T, G_H, G_B, G_TF, G_TW, G_I,
                      E_GEOM, E_MAT, M_FY, ROR, IPROFIL)

# ---------------------------------------------------------------------------
# Profilkataloger
# ---------------------------------------------------------------------------
# HEA-profiler (EN 10365): (navn, h, b, tw, tf) i mm.
HEA_PROFILER = (
    ("HEA100", 96, 100, 5.0, 8.0),     ("HEA120", 114, 120, 5.0, 8.0),
    ("HEA140", 133, 140, 5.5, 8.5),    ("HEA160", 152, 160, 6.0, 9.0),
    ("HEA180", 171, 180, 6.0, 9.5),    ("HEA200", 190, 200, 6.5, 10.0),
    ("HEA220", 210, 220, 7.0, 11.0),   ("HEA240", 230, 240, 7.5, 12.0),
    ("HEA260", 250, 260, 7.5, 12.5),   ("HEA280", 270, 280, 8.0, 13.0),
    ("HEA300", 290, 300, 8.5, 14.0),   ("HEA320", 310, 300, 9.0, 15.5),
    ("HEA340", 330, 300, 9.5, 16.5),   ("HEA360", 350, 300, 10.0, 17.5),
    ("HEA400", 390, 300, 11.0, 19.0),  ("HEA450", 440, 300, 11.5, 21.0),
    ("HEA500", 490, 300, 12.0, 23.0),  ("HEA550", 540, 300, 12.5, 24.0),
    ("HEA600", 590, 300, 13.0, 25.0),  ("HEA650", 640, 300, 13.5, 26.0),
    ("HEA700", 690, 300, 14.5, 27.0),  ("HEA800", 790, 300, 15.0, 28.0),
    ("HEA900", 890, 300, 16.0, 30.0),  ("HEA1000", 990, 300, 16.5, 31.0),
)

# Rørprofiler: standard ytterdiametre og veggtykkelser (mm). Bare kombinasjoner
# med D/t < 30 tas med (krav mot lokal knekking på trykksiden).
ROR_DIAMETRE = (21.3, 26.9, 33.7, 42.4, 48.3, 60.3, 76.1, 88.9, 101.6, 114.3, 139.7,
                168.3, 177.8, 193.7, 219.1, 244.5, 273.0, 323.9, 355.6, 406.4, 457.0,
                508.0, 609.6)
ROR_TYKKELSER = (2.3, 2.6, 3.2, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 14.2, 16.0, 20.0,
                 25.0, 30.0, 40.0)
D_T_MAKS = 30.0

# Et profil i katalogen: navn, en ferdig utfylt tverrsnittsrad (samme format
# som i inputfilen) og egenskapene A [mm^2], I [mm^4], W = I/c [mm^3] og
# treghetsradius i = sqrt(I/A) [mm].
Profil = namedtuple("Profil", "navn rad A I W i")


def _lag_profil(navn, profiltype, D=0.0, t=0.0, h=0.0, b=0.0, tf=0.0, tw=0.0):
    #Lager en tverrsnittsrad og beregner egenskapene med samme funksjon som resten av programmet (tverrsnittsdata),
    #så alt blir konsistent
    rad = np.zeros(9)
    rad[G_TYPE], rad[G_D], rad[G_T] = profiltype, D, t
    rad[G_H], rad[G_B], rad[G_TF], rad[G_TW] = h, b, tf, tw
    A, I, c = tverrsnittsdata(rad)
    return Profil(navn, rad, A, I, I / c, np.sqrt(I / A))


def profilkatalog(profiltype):
    # Returnerer listen med tilgjengelige profiler for rør (0) eller I-profil (1)
    if profiltype == IPROFIL:
        return [_lag_profil(navn, IPROFIL, h=h, b=b, tf=tf, tw=tw)
                for navn, h, b, tw, tf in HEA_PROFILER]
    return [_lag_profil(f"Rør {D:g}x{t:g}", ROR, D=D, t=t)
            for D in ROR_DIAMETRE for t in ROR_TYKKELSER
            if 2 * t < D and D / t < D_T_MAKS]


def sett_profiler(geom, profilvalg):
    """Returnerer en kopi av geom der radene for de dimensjonerte gruppene er
    erstattet med de valgte profilene. profilvalg: {tverrsnittnr: Profil}."""
    ny_geom = geom.copy()
    for igeom, profil in profilvalg.items():
        ny_geom[igeom, 1:] = profil.rad[1:]     # behold nr i kolonne 0
        ny_geom[igeom, G_I] = 0.0               # I beregnes fra målene
    return ny_geom


def dimensjoner(geom, mat, elem, elemlen, analyser, utnyttelse_maks=0.7,
                lambda_maks=200, maks_iterasjoner=50):
    """
    Velger profiler for alle tverrsnitt (elementgrupper) med metoden beskrevet
    øverst i filen.

    analyser(geom) er en funksjon (fra main.py) som kjører hele analysen for
    et gitt sett med tverrsnitt og returnerer Mval (nelem x 3) i kNm.

    Returnerer:
        ny_geom   : tverrsnittstabellen med de valgte profilene
        valg      : {tverrsnittnr: Profil}
        historikk : liste med profilnavn per iterasjon (til rapporten)
    """
    grupper = sorted(set(elem[:, E_GEOM]))                 # tverrsnitt som brukes
    katalog = {g: profilkatalog(int(geom[g, G_TYPE])) for g in grupper}
    elementer = {g: np.flatnonzero(elem[:, E_GEOM] == g) for g in grupper}
    fy = mat[elem[:, E_MAT], M_FY]                          # flytspenning per element
    if np.any(~(fy > 0)):
        raise ValueError("Alle elementer må ha positiv flytspenning for å kunne dimensjoneres")

    # Steg 1: konservativ start med det stiveste profilet i hver gruppe
    valg = {g: max(katalog[g], key=lambda p: p.I) for g in grupper}
    historikk, sett_tidligere = [], set()
    bare_okning = False

    for _ in range(maks_iterasjoner):
        historikk.append([valg[g].navn for g in grupper])
        sett_tidligere.add(tuple(valg[g].navn for g in grupper))

        # Steg 2: analyse med nåværende profiler
        Mval = analyser(sett_profiler(geom, valg))

        # Steg 3: nytt profil for hver gruppe ut fra momentene i denne analysen
        nytt = {}
        for g in grupper:
            e = elementer[g]
            # Nødvendig W for hvert element (elementene kan ha ulikt material),
            # og den største av disse gjelder for gruppen. 1e6: kNm -> Nmm.
            W_krav = np.max(np.max(np.abs(Mval[e]), axis=1) * 1e6 / (utnyttelse_maks * fy[e]))
            L_maks = np.max(elemlen[e]) * 1000                       # [mm]
            kandidater = [p for p in katalog[g] if p.W >= W_krav and
                          (lambda_maks is None or L_maks / p.i <= lambda_maks)]
            if not kandidater:
                raise ValueError(f"Ingen profil i katalogen er stort nok for tverrsnitt {g} "
                                 f"(W_krav = {W_krav:.3g} mm^3).")
            lettest = min(kandidater, key=lambda p: p.A)
            # Når iterasjonen har gått i ring, beholdes et profil som allerede
            # er stort nok i stedet for å gå ned igjen (sikrer at den stopper).
            if bare_okning and lettest.W < valg[g].W:
                lettest = valg[g]
            nytt[g] = lettest

        # Steg 4: ferdig når ingenting endres
        if all(nytt[g].navn == valg[g].navn for g in grupper):
            return sett_profiler(geom, valg), valg, historikk
        if tuple(nytt[g].navn for g in grupper) in sett_tidligere:
            bare_okning = True      # iterasjonen har gått i ring
        valg = nytt

    raise RuntimeError(f"Dimensjoneringen konvergerte ikke på {maks_iterasjoner} iterasjoner.")
