import numpy as np


def bc(punkt, K, R):
    #Innfører fastholdte rotasjoner i systemets stivhetsligning.

    # Parametere
    # ----------
    # punkt : array_like, shape (npunkt, 4)
    #     Knutepunktdata. Tredje kolonne er hvorvidt knutepunktet er fastholdt:
    #     0 betyr fastholdt rotasjon og 1 betyr fri rotasjon.
    #     Fjerde kolonne er knutepunktets systemindeks.
    # K : array_like, shape (npunkt, npunkt)
    #     Systemstivhetsmatrisen.
    # R : array_like, shape (npunkt,)
    #     Systemlastvektoren.

    # Returnerer
    # ----------
    # tuple[numpy.ndarray, numpy.ndarray]
    #     Kn og Rn etter at grensebetingelsene er innført.
    #     Inputdataene endres ikke.

    punkt = np.asarray(punkt, dtype=float)
    K = np.asarray(K, dtype=float)
    R = np.asarray(R, dtype=float)

    if punkt.ndim != 2 or punkt.shape[1] < 4:
        raise ValueError("punkt må ha minst fire kolonner")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K må være en kvadratisk matrise")
    if R.ndim != 1 or len(R) != K.shape[0]:
        raise ValueError("R må ha samme dimensjon som K")
    if len(punkt) != K.shape[0]:
        raise ValueError("punkt må ha ett element per systemfrihetsgrad")

    Kn = K.copy()
    Rn = R.copy()
    systemindekser = punkt[:, 3].astype(int)

    if np.any(systemindekser < 0) or np.any(systemindekser >= len(R)):
        raise ValueError("punkt inneholder en ugyldig systemindeks")
    if len(np.unique(systemindekser)) != len(systemindekser):
        raise ValueError("punkt inneholder dupliserte systemindekser")

    for fastholdt, indeks in zip(punkt[:, 2], systemindekser):
        if fastholdt == 0:
            Kn[indeks, :] = 0.0
            Kn[:, indeks] = 0.0
            Kn[indeks, indeks] = 1.0
            Rn[indeks] = 0.0
        elif fastholdt != 1:
            raise ValueError(f"Ugyldig fastholdingsflagget {fastholdt:g} "f"for knutepunkt {indeks}")
    return Kn, Rn