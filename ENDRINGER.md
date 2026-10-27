# Endringer i Python-programmet

Kjøres med `python main.py`. Da kjøres oppgave b (plattformen) og oppgave c (dimensjonering av rammen i figur 1).


## 1. Avvik mot Nauticus 3D Beam

Med nøyaktig de samme profilene som i 3D Beam, og evaluert i de samme punktene (3D Beam rapporterer noen verdier 12–20 mm inn på elementet), stemmer alle momentene innenfor 0,26 kNm (< 0,1 %). Det er et ubetydelig avvik. Med fysisk riktige opplagre og reell aksialstivhet (ikke alle knutepunkter fastholdt) endres momentene med maks ca. 0,5 %, så antagelsen om uforskyvelig ramme er god.


## 2. Feil som er rettet

- **Programmet krasjet på NumPy versjon 2.x**: `np.cross` med 2D-vektorer er fjernet. Koden bruker nå prikkprodukt med normalvektoren.
- **Feil i HEA-tabellen**: HEA700 hadde tf = 32 (riktig 27) og HEA800 hadde tf = 35,5 (riktig 28). Skrev også inn HEA550, 650, 900 og 1000 i tabellen.
- **Rørlisten var sortert på diameter, ikke vekt**. Derfor ble «første rør som holder» et tykkvegget og ineffektivt rør (273x40). 
- `main()` ble kjørt ved import (manglet `if __name__ == "__main__":`).
- Formelen for I-profilets I sto tre steder og formelen for laster to steder. Nå beregnes tverrsnittsdata ett sted (`tverrsnittsdata`), og lastene knyttes til elementene ett sted (`fordel_laster`).

## 3. Mangler mot oppgaveteksten som er lagt til

- **Materialnummer**: oppgaven krever at elementene kobles til geometri OG material via nummer i inputfilen. Nå finnes det en egen materialseksjon (E, fy), og hvert element har både tverrsnittnummer og materialnummer.
- **Generell lineært fordelt last**: én lasttype med q1 i ende 1 og q2 i ende 2 (to trekantbidrag i FIM-formelen). Den erstatter de to gamle typene jevnt fordelt last og trekantlast. Vindlasten er nå én linje (8 → 13 kN/m).
- **Endemomenter i matrisemetodens konvensjon** (med klokka positiv) skrives ut. Det er disse som sammenlignes med fasit i oppgave b.
- **Kontroll av momentlikevekt** i alle frie knutepunkter skrives ut.


## 4. Ny dimensjoneringsmetode (oppgave c)

Den gamle metoden valgte for hver gruppe det minste profilet som besto, gruppe for gruppe. En slakk søyle tar nesten ikke moment, så et bitte lite rør «består» alltid. Resultatet ble én massiv søyle og to søyler på 21 mm.

Ny metode («fully stressed design», se `dimensjonering.py`):
1. Start med det stiveste profilet i alle grupper.
2. Analyser rammen.
3. For hver gruppe: velg det **letteste** profilet med W ≥ |M|maks / (0,7·fy). Alle gruppene oppdateres samtidig.
4. Gjenta til ingenting endres. Hvis iterasjonen går i ring, tillates bare økning.

I tillegg brukes en praktisk slankhetsgrense L/i ≤ 200, slik at en søyle med nesten null moment ikke blir et 21 mm rør. Dette er ikke en knekkontroll. Grensen kan slås av med `lambda_maks=None`.

| Tverrsnitt | Elementer | Profil | Utnyttelse |
|---|---|---|---|
| 0 venstre søyle | 0, 1, 2 | Rør 355,6x16 | 67,6 % |
| 1 midtre søyle | 4, 5, 8 | Rør 219,1x8 | 58,5 % |
| 2 høyre søyle | 7 | Rør 177,8x6,3 | 25,6 % (styrt av slankhet) |
| 3 øverste bjelke | 3 | HEA340 | 69,5 % |
| 4 bjelker y = 12 m | 6, 9 | HEA320 | 62,4 % |
| 5 bjelke y = 24 m | 10 | HEA320 | 69,0 % |

Totalvekten er ca. 13,6 tonn (gammel løsning ca. 15,2 tonn). Alle søylene bærer nå en reell andel av lasten.


## 5. Nytt inputformat

Formatet er beskrevet øverst i `lesinput.py` og i kommentarene i inputfilene. Kort oppsummert:

- Seksjonene kommer i rekkefølgen knutepunkter, materialer, tverrsnitt, elementer og laster.
- Den nye utskriften viser alle mål (h, b, tf, tw og steghøyde mellom flensene, eller D og t) og hvilke elementer som hører til hvert tverrsnitt. 
- Hver seksjon starter med antall linjer, og første kolonne er nummeret (0, 1, 2, …).
- Alle søyler er nå definert nedenfra og opp (elementene 4, 5, 7 og 8 er snudd). Da betyr positivt moment strekk på høyre side for alle søyler og på undersiden for alle bjelker.
- **Nummereringen av knutepunkter og elementer er uendret**, så diskretiseringsskissen i rapporten gjelder fortsatt.