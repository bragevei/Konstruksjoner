#f=flensbredde,s=steghøyde,t=tykkelse,E=elastisitetsmodul, R=radius 
#hvis geom = 0 så er det en O-profil, hvis geom = 1 så er det en I-profil
#antar tynnvegget  O-profil.
import numpy as np
def beregne_boyestivhet(f,s,t,E,R,geom):
    if geom == 1:
        I = f*s**3/12 + 2*(f*t**3/12 + f*t*(s/2+t/2)**2)
    elif geom == 0:
        I = np.pi*R**3*t
    else:
        print("Feil i inputdata for beregning av bøyestivhet. Geometri må være 1 eller 0.")
        return None
    EI = E*I
    return EI


