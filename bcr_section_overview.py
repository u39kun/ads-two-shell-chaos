"""The whole head-on section at BCR's Fig. 5 parameters (Fig. 7 and Sec. 4.3 of the paper).

  edges    at each R, the allowed head-on crossings form an interval of M_mid: its lower edge is where shell A crosses
           at a turning point (A's effective potential vanishes) and its upper edge where shell B does; both are
           located by bisection;
  ends     the section's two pointed ends, where the interval closes: the smallest and largest R at which
           max over M_mid of min(-V_A, -V_B) is still positive (bisection in R, bounded maximization in M_mid);
  orbits   a strongly chaotic orbit started near the small-R end, (R, 100 M_mid) = (0.19, 1.575), and a start nearby
           that stays at that end, (0.185, 1.570): their head-on crossings to t = 20000 (event-located solver), how
           many fall near each end (R < 0.4, R > 2.0) or in between, the number of switches between the ends, and
           the growth rate per unit time of the variational tangent (chart (R, 100 M_mid)) over [5000, 20000].
usage: python bcr_section_overview.py OUT.json"""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from scipy.optimize import brentq, minimize_scalar
import tangent_map as TM
import bcr_section as BS
from common import N

TOT, T = BS.TOT, 20000.0
ORBITS = [(0.19, 0.01575, "chaotic, near the small-R end"), (0.185, 0.01570, "stays at the small-R end")]
NEAR_SMALL, NEAR_LARGE = 0.4, 2.0


def va(R, M):
    return N.potential(R, 0.0, M, BS.A)


def vb(R, M):
    return N.potential(R, M, TOT, BS.B)


def margin(R, M):
    """Positive where a head-on crossing at (R, M) is allowed: both shells inside their allowed ranges."""
    return min(-va(R, M), -vb(R, M))


def best(R):
    r = minimize_scalar(
        lambda M: -margin(R, M), bounds=(0.3 * TOT, 0.7 * TOT), method="bounded", options=dict(xatol=1e-13)
    )
    return -r.fun, r.x


def edges(R):
    """Lower and upper M_mid of the allowed interval at R, or None if it is empty."""
    g, M0 = best(R)
    if g <= 0:
        return None
    lo = brentq(lambda M: va(R, M), 0.3 * TOT, M0, xtol=1e-15) if va(R, 0.3 * TOT) > 0 else 0.3 * TOT
    hi = brentq(lambda M: vb(R, M), M0, 0.7 * TOT, xtol=1e-15) if vb(R, 0.7 * TOT) > 0 else 0.7 * TOT
    return lo, hi


def end(r_in, r_out):
    """Bisect between an R inside the section and one outside it; returns the end point (R, M_mid)."""
    R = brentq(lambda R: best(R)[0], r_in, r_out, xtol=1e-12)
    return [float(R), float(best(R)[1])]


def orbit(args):
    R0, M0, label = args
    va0, vb0 = np.sqrt(-va(R0, M0)), -np.sqrt(-vb(R0, M0))
    mid, na, nb, _, _ = N.crossing(R0, va0, vb0, 0.0, M0, TOT, BS.A, BS.B)
    s = N.simulate(
        [BS.A, BS.B],
        [0.0, mid, TOT],
        [R0, na, R0, nb],
        order=[1, 0],
        tmax=T,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=10**7,
    )
    q = np.array(
        [
            [e["r"], e["masses_before"][1]]
            for e in s["events"]
            if e["loser"] == "A" and e["v_before"][0] > 0 > e["v_before"][1]
        ]
    )
    zone = np.where(q[:, 0] < NEAR_SMALL, 0, np.where(q[:, 0] > NEAR_LARGE, 2, 1))
    ends = zone[zone != 1]
    # variational tangent, chart (R, 100 M_mid), rate over [T/4, T]
    sc = np.array([1.0, 100.0])
    x = np.array([R0, M0])
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S, clock, cum = 0.0, 0.0, [(0.0, 0.0)]
    P = TM.TwoShellMap(m=BS.M, w=BS.W, Min=0.0, Mt=TOT, rtol=1e-12, atol=1e-14, max_step=0.02, tmax=5000.0)
    while clock < T:
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            break
        v = DF @ v
        g = np.linalg.norm(v * sc)
        S += np.log(g)
        v /= g
        clock += tr
        cum.append((clock, S))
        x = x1
    c = np.array(cum)
    i = int(np.searchsorted(c[:, 0], T / 4))
    return dict(
        x0=[R0, M0],
        label=label,
        status=s["status"],
        headon=q.tolist(),
        R_range=[float(q[:, 0].min()), float(q[:, 0].max())],
        near_small_end=int((zone == 0).sum()),
        middle=int((zone == 1).sum()),
        near_large_end=int((zone == 2).sum()),
        end_switches=int((np.diff(ends) != 0).sum()),
        tangent_status="ok" if clock >= T else f"stopped ({st})",
        rate_per_time=float((c[-1, 1] - c[i, 1]) / (c[-1, 0] - c[i, 0])),
    )


def main():
    t0 = time.time()
    out_path = sys.argv[1]
    Rs = np.linspace(0.1, 2.6, 1251)
    with Pool(min(len(ORBITS), os.cpu_count())) as pool:
        orb = pool.map_async(orbit, ORBITS)
        e = [edges(R) for R in Rs]
        inside = [R for R, x in zip(Rs, e) if x is not None]
        ends = dict(small_R=end(min(inside), min(inside) - 0.002), large_R=end(max(inside), max(inside) + 0.002))
        orb = orb.get()
    rows = [[float(R), float(x[0]), float(x[1])] for R, x in zip(Rs, e) if x is not None]
    for o in orb:
        o["distance_to_large_end"] = ends["large_R"][0] - o["R_range"][1]
    res = dict(
        parameters=dict(m=BS.M, w=BS.W, masses=BS.MASSES),
        edges=rows,
        ends=ends,
        near_end_cuts=[NEAR_SMALL, NEAR_LARGE],
        orbits=orb,
        seconds=time.time() - t0,
    )
    json.dump(res, open(out_path, "w"))
    print(f"ends: small R {ends['small_R']}, large R {ends['large_R']}")
    for o in orb:
        print(
            f"{o['label']}: x0 {o['x0']}, {len(o['headon'])} head-on crossings, R {o['R_range']}, near ends "
            f"{o['near_small_end']}/{o['middle']}/{o['near_large_end']}, switches {o['end_switches']}, "
            f"to the large-R end {o['distance_to_large_end']:.4f}; tangent rate {o['rate_per_time']:.3g} per unit time"
        )
    print(f"wrote {out_path}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
