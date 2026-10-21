import numpy as np

def lengder(punkt, elemkonn):
    punkt = np.asarray(punkt, dtype=float)
    elemkonn = np.asarray(elemkonn, dtype=int)
    if punkt.ndim != 2 or punkt.shape[1] < 2:
        raise ValueError("punkt må ha minst to kolonner")
    if elemkonn.ndim != 2 or elemkonn.shape[1] != 2:
        raise ValueError("elemkonn må ha formen (antall elementer, 2)")
    if np.any(elemkonn < 0) or np.any(elemkonn >= len(punkt)):
        raise ValueError("elemkonn inneholder en ugyldig knutepunktindeks")

    vektorer = punkt[elemkonn[:, 1], :2] - punkt[elemkonn[:, 0], :2]
    elemlen = np.linalg.norm(vektorer, axis=1)
    if np.any(elemlen <= 0):
        raise ValueError("Elementer kan ikke ha lengde lik null")
    return elemlen