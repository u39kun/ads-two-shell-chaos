"""Critical scaling at the upper edge of BCR's three-crossing band (type (ii) data; see bcr_validation.py).
Bisects delta3* = the largest delta with exactly 3 crossings before collapse (between 0.0558 and 0.0560), then runs
delta3* - eps for 44 geometric eps in [1e-9, 3e-5], records crossings, trapped shell, and black-hole mass, and fits
M_BH = M_c + A eps^gamma on runs with 3 crossings where the inner shell (rank 0) is trapped, for several eps ranges.
usage: python bcr_critical.py OUT.json"""

import sys, os, json

os.environ.setdefault("SOLVER", "fast")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import curve_fit
import bcr_validation as B


def main():
    lo, hi = 0.0558, 0.0560
    assert B.ncross(lo) == 3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if B.ncross(mid) == 3:
            lo = mid
        else:
            hi = mid
    ds = lo
    print("delta3* =", repr(ds))
    eps = np.geomspace(1e-9, 3e-5, 44)
    R = [B.run(ds - e) for e in eps]
    sel = [(e, r["M_BH"]) for e, r in zip(eps, R) if r.get("crossings") == 3 and r.get("trapped_rank") == 0]
    e = np.array([q[0] for q in sel])
    M = np.array([q[1] for q in sel])
    f = lambda x, M0, A, g: M0 + A * x**g
    fits = {}
    for a_, b_ in ((1e-9, 3e-6), (1.7e-7, 3e-6), (1e-9, 1.1e-5), (1e-9, 3e-5)):
        s = (e >= a_) & (e <= b_)
        p, cv = curve_fit(f, e[s], M[s], p0=[M[s].min() - 1e-6, 1.0, 0.9], maxfev=50000)
        fits[f"{a_:.0e}-{b_:.0e}"] = dict(
            n=int(s.sum()), gamma=float(p[2]), gamma_err=float(np.sqrt(cv[2, 2])), M0=float(p[0]), A=float(p[1])
        )
        print(f"eps {a_:.0e}..{b_:.0e}: n {s.sum()}, gamma {p[2]:.3f} +- {np.sqrt(cv[2, 2]):.3f}")
    json.dump(dict(delta3_star=ds, eps=eps.tolist(), runs=R, fits=fits), open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
