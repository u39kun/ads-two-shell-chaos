"""Smoothness of the section curve of BCR's Fig. 5 example (Sec. 3.3 and App. B of the paper).

Reads the example's crossings to t = 1e5 (long_points and long_times in results/bcr_section/bcr_checks.json) and,
as for Fig. 2b, takes those with 14.6e-3 < M_mid < 15.6e-3 and R > 0.9 near the waist of the curve. In strips of
width 2.5e-5, 5e-5, and 1e-4 in M_mid centered on their median M_mid, it fits R as a polynomial in M_mid of degree
6 to 10 and records the rms residual. For degree 8 it also fits the residual linearly in time (the slow drift of the
crossings across the curve, quoted per 1e5 time units) and records the rms residual with that drift removed.
usage: python bcr_curve_smoothness.py CHECKS.json OUT.json"""

import sys, json
import numpy as np


def main():
    C = json.load(open(sys.argv[1]))
    L = np.array(C["long_points"])
    T = np.array(C["long_times"])
    k = (L[:, 1] > 14.6e-3) & (L[:, 1] < 15.6e-3) & (L[:, 0] > 0.9)
    x = L[k, 1] * 1e3
    y = L[k, 0]
    t = T[k]
    c0 = float(np.median(x))  # x: M_mid in units of 1e-3
    strips = []
    for width in (2.5e-5, 5e-5, 1e-4):
        s = np.abs(x - c0) < width * 1e3 / 2
        rms = {}
        for deg in range(6, 11):
            pf = np.polyfit(x[s] - c0, y[s], deg)
            rms[str(deg)] = float(np.sqrt(np.mean((y[s] - np.polyval(pf, x[s] - c0)) ** 2)))
        r = y[s] - np.polyval(np.polyfit(x[s] - c0, y[s], 8), x[s] - c0)
        slope, icept = np.polyfit(t[s], r, 1)
        strips.append(
            dict(
                width=width,
                n=int(s.sum()),
                rms_by_degree=rms,
                drift_per_1e5=float(slope * 1e5),
                rms_drift_removed=float(np.sqrt(np.mean((r - slope * t[s] - icept) ** 2))),
            )
        )
        print(
            f"strip {width:.1e}: {int(s.sum())} crossings; rms (degree 6-10) "
            + " ".join(f"{v:.2e}" for v in rms.values())
            + f"; drift {slope*1e5:.2e} per 1e5; rms with drift removed {strips[-1]['rms_drift_removed']:.1e}",
            flush=True,
        )
    json.dump(dict(center_M_mid=c0 * 1e-3, t_max=float(T[-1]), strips=strips), open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
