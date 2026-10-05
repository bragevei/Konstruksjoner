#f=flensbredde,s=steghøyde,t=tykkelse,E=elastisitetsmodul, R=radius 
#hvis geom = "O" så er det en O-profil, hvis geom = "I" så er det en I-profil
#antar tynnvegget  O-profil.
import numpy as np
def beregne_boyestivhet(f,s,t,E,R,geom):
    if geom == "I":
        I = f*s**3/12 + 2*(f*t**3/12 + f*t*(s/2+t/2)**2)
    elif geom == "O":
        I = np.pi*R**3*t
    else:
        print("Feil i inputdata for beregning av bøyestivhet. Geometri må være 'I' eller 'O'.")
        return None
    EI = E*I
    return EI


