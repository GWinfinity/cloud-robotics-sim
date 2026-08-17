import numpy as np  # noqa: E402, I001

nx = ny = 16
dx = 0.01
T = np.zeros((nx, ny), dtype=np.float32)
sources = [
    {"pos": np.array([0.06, 0.06]), "T": 1.0, "radius": 0.04, "C": 0.01},
    {"pos": np.array([0.10, 0.10]), "T": 0.0, "radius": 0.04, "C": 0.01},
]
Ccell = 1e-4


def energy():
    """Return the total thermal energy of bodies and grid."""
    bodies = sum(s["C"] * s["T"] for s in sources)
    return bodies + float(Ccell * T.sum())


mask = []
for s in sources:
    m = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            px = i * dx
            py = j * dx
            if (px - s["pos"][0]) ** 2 + (py - s["pos"][1]) ** 2 <= s["radius"] ** 2:
                m[i, j] = True
    mask.append(m)

E0 = energy()
print("E0", E0)
for step in range(20):
    # source coupling
    t_eqs = []
    for idx, s in enumerate(sources):
        cells = T[mask[idx]]
        n = cells.size
        if n == 0:
            t_eqs.append(s["T"])
            continue
        t_eq = (s["C"] * s["T"] + Ccell * cells.sum()) / (s["C"] + Ccell * n)
        t_eqs.append(t_eq)
        s["T"] = t_eq  # alpha=1
    # update cells
    delta = np.zeros_like(T)
    for idx, s in enumerate(sources):
        delta[mask[idx]] += t_eqs[idx] - T[mask[idx]]
    T += delta
    print(step + 1, energy(), (energy() - E0) / E0)
