import numpy as np

def lengder(punkt, elemkonn):

    elemlen = np.array([])
    # Beregner elementlengder vha. Pythagoras
    for ielemkonn in elemkonn:
        dx = punkt[ielemkonn[1], 0] - punkt[ielemkonn[0], 0]
        dy = punkt[ielemkonn[1], 1] - punkt[ielemkonn[0], 1]
        elemlen = np.append(elemlen, np.sqrt(dx*dx + dy*dy))

    return elemlen