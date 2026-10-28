"""Poincare section of BCR's Fig. 5 orbit (Brito, Cardoso & Rocha, PRD 94, 024003; arXiv:1602.03535).

Two identical shells, m = 0.00815, w = 0.2, region masses 0, 0.015, 0.03, both starting at R = 1 (the two velocity
assignments D and C of bcr_lyapunov.py). Section: head-on crossings with shell A inside, A moving out and B moving
in (the section of the two-shell map in Sec. 2 of the paper), in the chart (R, M_mid) with M_mid the middle-region
mass just before the crossing. Background: orbits of the same map started along a line through its fixed points.
Fixed points of the return map are located by Newton iteration.
Regularity: digits of agreement between weighted Birkhoff averages of M_mid over the two halves of the section points.
usage: python bcr_section.py OUT.json [--tmax 20000] [--background 25] [--bg-returns 1500]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from common import N

M, W, MASSES = 0.00815, 0.2, [0.0, 0.015, 0.03]
A, B = N.Shell("A", M, W), N.Shell("B", M, W)
TOT = MASSES[2]


def initial(start):
    sa = np.sqrt(-N.potential(1.0, MASSES[0], MASSES[1], A))
    sb = np.sqrt(-N.potential(1.0, MASSES[1], MASSES[2], B))
    if start == "D":
        return list(MASSES), [1.0, -sa, 1.0, sb], [0, 1]
    mid, va, vb, _, _ = N.crossing(1.0, sa, -sb, MASSES[0], MASSES[1], MASSES[2], A, B)
    return [MASSES[0], mid, MASSES[2]], [1.0, va, 1.0, vb], [1, 0]


def section_points(events):
    pts, ts = [], []
    for e in events:
        if e["loser"] == "A" and e["v_before"][0] > 0 > e["v_before"][1]:
            pts.append([e["r"], e["masses_before"][1]])
            ts.append(e["t"])
    return np.array(pts), np.array(ts)


def run_start(args):
    start, tmax = args
    masses, y, order = initial(start)
    r = N.simulate(
        [A, B], masses, y, order=order, tmax=tmax, anchor=1, rtol=1e-12, atol=1e-14, max_step=0.02, max_events=1_000_000
    )
    p, t = section_points(r["events"])
    return dict(
        start=start,
        status=r["status"],
        t_end=r["t_end"],
        crossings=len(r["events"]),
        points=p.tolist(),
        times=t.tolist(),
    )


def velocities(R, mid):
    pa, pb = N.potential(R, 0.0, mid, A), N.potential(R, mid, TOT, B)
    if pa >= 0 or pb >= 0:
        return None
    return np.sqrt(-pa), -np.sqrt(-pb)


def F(x):
    v = velocities(*x)
    if v is None:
        raise ValueError("forbidden")
    newmid, na, nb, _, _ = N.crossing(x[0], v[0], v[1], 0.0, x[1], TOT, A, B)
    r = N.simulate(
        [A, B],
        [0.0, newmid, TOT],
        [x[0], na, x[0], nb],
        order=[1, 0],
        tmax=80,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=40,
    )
    p, _ = section_points(r["events"])
    if len(p) == 0:
        raise RuntimeError("no return")
    return p[0]


def newton(x, it=30):
    x = np.array(x, float)
    for _ in range(it):
        Fx = F(x)
        r = Fx - x
        J = np.zeros((2, 2))
        for j, h in enumerate((1e-6, 1e-8)):
            d = np.zeros(2)
            d[j] = h
            J[:, j] = (F(x + d) - F(x - d)) / (2 * h)
        dx = np.linalg.solve(J - np.eye(2), -r)
        x = x + dx
        if np.linalg.norm(dx / np.array([1, 0.01])) < 1e-12:
            break
    J = np.zeros((2, 2))
    for j, h in enumerate((1e-6, 1e-8)):
        d = np.zeros(2)
        d[j] = h
        J[:, j] = (F(x + d) - F(x - d)) / (2 * h)
    ev = np.linalg.eigvals(J)
    kind = "elliptic" if np.all(np.abs(np.abs(ev) - 1) < 1e-5) and np.any(np.abs(ev.imag) > 1e-8) else "hyperbolic"
    rho = float(np.abs(np.angle(ev[0])) / (2 * np.pi)) if kind == "elliptic" else None
    return dict(
        x=x.tolist(),
        residual=float(np.linalg.norm(F(x) - x)),
        eigenvalues=[complex(e).__repr__() for e in ev],
        abs_eigenvalues=np.abs(ev).tolist(),
        trace=float(np.trace(J)),
        kind=kind,
        rotation_number=rho,
    )


def run_bg(args):
    x0, n = args
    v = velocities(*x0)
    if v is None:
        return dict(x0=list(x0), points=[])
    newmid, na, nb, _, _ = N.crossing(x0[0], v[0], v[1], 0.0, x0[1], TOT, A, B)
    r = N.simulate(
        [A, B],
        [0.0, newmid, TOT],
        [x0[0], na, x0[0], nb],
        order=[1, 0],
        tmax=5.0 * n,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=6 * n,
    )
    p, _ = section_points(r["events"])
    return dict(x0=list(x0), points=[list(x0)] + p[:n].tolist(), status=r["status"])


def wb(P):
    n = len(P)
    t = (np.arange(n) + 0.5) / n
    w = np.exp(-1.0 / (t * (1 - t)))
    return (w @ P) / w.sum()


def regularity(points):
    m = np.array(points)[:, 1]
    n = len(m) // 2
    a, b = wb(m[:n]), wb(m[n : 2 * n])
    return float(-np.log10(abs(a - b) / abs(a))) if a != b else 16.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=20000.0)
    p.add_argument("--background", type=int, default=25)
    p.add_argument("--bg-returns", type=int, default=1500)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    out = dict(params=dict(m=M, w=W, masses=MASSES, tmax=a.tmax))
    with Pool(a.workers) as pool:
        runs = pool.map(run_start, [("D", a.tmax), ("C", a.tmax)])
    for r in runs:
        r["regularity_digits"] = regularity(r["points"]) if len(r["points"]) > 20 else None
        print(
            f"start {r['start']}: {r['status']} t_end {r['t_end']:.0f}, {r['crossings']} crossings, "
            f"{len(r['points'])} section points, regularity {r['regularity_digits']}"
        )
    out["bcr_orbit"] = runs
    # fixed points of the return map. Newton starts from guesses read off the example's section curve, an
    # hourglass: its waist (the saddle) and its upper and lower halves (the two centers). A blind grid of 84 starts
    # (R from 0.45 to 1.3, M_mid from 0.012 to 0.018) finds the same three fixed points and no others.
    seeds = [(0.79, 0.0186), (0.79, 0.0116), (0.825, 0.0151)]
    fps = []
    for s in seeds:
        try:
            fp = newton(s)
            fps.append(fp)
            print(
                "fixed point", np.round(fp["x"], 8), fp["kind"], "rho", fp["rotation_number"], "resid", fp["residual"]
            )
        except Exception as ex:
            print("newton failed from", s, ex)
    out["fixed_points"] = fps
    # background orbits: start points along the segment through the fixed points, and through the BCR orbit's range
    allp = np.vstack([np.array(r["points"]) for r in runs if r["points"]])
    lo, hi = allp[:, 1].min(), allp[:, 1].max()
    xs = [fp["x"] for fp in fps]
    ell = sorted([fp["x"] for fp in fps if fp["kind"] == "elliptic"], key=lambda x: x[1])
    starts = []
    if len(ell) == 2:
        for s in np.linspace(0.02, 0.98, a.background):
            starts.append(tuple(np.array(ell[0]) + s * (np.array(ell[1]) - np.array(ell[0]))))
    Rm = float(np.median(allp[:, 0]))
    for mid in np.linspace(lo, hi, 9):
        starts.append((Rm, float(mid)))
    with Pool(a.workers) as pool:
        bg = pool.map(run_bg, [(s, a.bg_returns) for s in starts])
    for b in bg:
        b["regularity_digits"] = regularity(b["points"]) if len(b["points"]) > 40 else None
    out["background"] = bg
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"))
    print(
        f'done in {out["seconds"]:.0f}s; background regularity digits:',
        [None if b["regularity_digits"] is None else round(b["regularity_digits"], 1) for b in bg],
    )


if __name__ == "__main__":
    main()
