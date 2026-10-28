"""Code validation against Brito, Cardoso & Rocha, PRD 94, 024003 (2016), arXiv:1602.03535, type (ii) data:
two identical shells, w = 0.2, rest-mass parameter m = 0.9*20^(-2w)*delta = 0.2715*delta, region masses
0 (inside), M2 = delta/2 (between), M1 = delta (outside), both at R_i = 1.5 at t = 0, inner contracting and outer
expanding (convention D: the shells separate at t = 0). Exterior (boundary) time, runs to t = 1000.
Outputs per delta: collapse or not, crossings before collapse, black-hole mass (mass just outside the trapped shell).
Then the 1 -> 3 transition: bisection for delta*, and M_BH(delta) near it on both sides for the critical exponent.
BCR: no collapse for delta <~ 0.0370; ladder 1, 3, 5, ..., 13 from right to left; gamma = 0.95 +- 0.05 at 1 -> 3.
usage: python bcr_validation.py OUT.json [--n 301]"""

import sys, os, json, time, argparse

os.environ.setdefault("SOLVER", "fast")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from common import N

W, RI, TMAX = 0.2, 1.5, 1000.0
MFAC = 0.9 * 20 ** (-2 * W)


def run(delta):
    A, B = N.Shell("A", MFAC * delta, W), N.Shell("B", MFAC * delta, W)
    masses = [0.0, delta / 2, delta]
    vin = -np.sqrt(-N.potential(RI, 0.0, delta / 2, A))
    vout = np.sqrt(-N.potential(RI, delta / 2, delta, B))
    try:
        r = N.simulate(
            [A, B],
            masses,
            [RI, vin, RI, vout],
            order=[0, 1],
            tmax=TMAX,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.02,
            max_events=100000,
        )
    except Exception as e:
        return dict(delta=delta, status=f"error {e}")
    out = dict(delta=delta, status=r["status"], crossings=len(r["events"]), t_end=r["t_end"])
    if r["status"] == "horizon_approach":
        st, ms = r["state_end"], r["masses_end"]
        order = [0 if n == "A" else 1 for n in r["order_end"]]
        fs = [float(N.f(st[2 * i], ms[k + 1])) for k, i in enumerate(order)]
        k = int(np.argmin(fs))
        out.update(M_BH=float(ms[k + 1]), trapped_rank=k)
    return out


def ncross(delta):
    r = run(delta)
    return r.get("crossings") if r["status"] == "horizon_approach" else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--n", type=int, default=301)
    a = p.parse_args()
    t0 = time.time()
    grid = np.linspace(0.030, 0.090, a.n)
    with Pool(os.cpu_count()) as pool:
        S = pool.map(run, grid.tolist())
    nc = [s for s in S if s["status"] != "horizon_approach"]
    thr = max(s["delta"] for s in nc) if nc else None
    print(f"scan {len(S)} deltas in {time.time() - t0:.0f}s; largest delta without collapse by t = {TMAX:.0f}: {thr}")
    # threshold by bisection between the largest non-collapsing and the next collapsing grid point
    lo = thr
    hi = min(s["delta"] for s in S if s["status"] == "horizon_approach" and s["delta"] > thr)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if run(mid)["status"] == "horizon_approach":
            hi = mid
        else:
            lo = mid
    print(f"no-collapse threshold: {lo:.6f} .. {hi:.6f}")
    threshold = [lo, hi]
    ladder = {}
    for s in S:
        if s["status"] == "horizon_approach":
            ladder.setdefault(s["crossings"], []).append(s["delta"])
    for k in sorted(ladder):
        print(
            f"  {k:3d} crossings before collapse: delta {min(ladder[k]):.4f} .. {max(ladder[k]):.4f} ({len(ladder[k])} grid points)"
        )
    # 1 -> 3 transition
    one = [s["delta"] for s in S if s.get("crossings") == 1 and s["status"] == "horizon_approach"]
    three = [s["delta"] for s in S if s.get("crossings") == 3 and s["status"] == "horizon_approach"]
    lo, hi = max(three), min(one)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        c = ncross(mid)
        if c == 1:
            hi = mid
        else:
            lo = mid
    dstar = 0.5 * (lo + hi)
    print(f"delta* (1 -> 3) = {dstar:.10f} (bracket {hi - lo:.1e})")
    eps = np.geomspace(1e-7, 3e-5, 24)
    with Pool(os.cpu_count()) as pool:
        up = pool.map(run, (dstar + eps).tolist())
        dn = pool.map(run, (dstar - eps).tolist())
    fits = {}
    for name, R in (("above (1 crossing)", up), ("below (3 crossings)", dn)):
        e = np.array([eps[i] for i, r in enumerate(R) if "M_BH" in r])
        M = np.array([r["M_BH"] for r in R if "M_BH" in r])
        best = None
        for M0 in np.linspace(M.min() - 2e-3, M.min() - 1e-9, 4000):
            d = M - M0
            if np.any(d <= 0):
                continue
            g, la = np.polyfit(np.log(e), np.log(d), 1)
            res = np.sum((np.log(d) - (g * np.log(e) + la)) ** 2)
            if best is None or res < best[0]:
                best = (res, M0, g)
        # three-parameter fit (M0, gamma, amplitude); the slope of log|M - M(eps_min)| alone would bias gamma
        fits[name] = dict(
            eps=e.tolist(),
            M_BH=M.tolist(),
            gamma=float(best[2]) if best else None,
            M0=float(best[1]) if best else None,
            crossings=[r.get("crossings") for r in R],
        )
        print(
            f'  {name}: crossings {sorted(set(fits[name]["crossings"]), key=str)}, M_BH {M.min():.6f}..{M.max():.6f}, '
            f'fit gamma = {fits[name]["gamma"]}, M0 = {fits[name]["M0"]}'
        )
    json.dump(
        dict(
            scan=S,
            threshold=threshold,
            delta_star_bracket=[lo, hi],
            ladder={str(k): v for k, v in ladder.items()},
            delta_star=dstar,
            fits=fits,
            seconds=time.time() - t0,
        ),
        open(a.out, "w"),
        indent=1,
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
