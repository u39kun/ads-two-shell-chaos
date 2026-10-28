"""Chaotic layer against delta for BCR's type (ii) parameter family, and where the type (ii) orbit sits relative to it.

At each delta the two-shell section map (head-on crossings with A inside, chart (R, M_mid)) is that of
bcr_delta_scan.py: identical shells with m = 0.9*20^(-0.4)*delta, w = 0.2, inner mass 0, total mass delta.
  1. Fixed points by Newton iteration from a grid of seeds; the saddle and the two elliptic points are kept.
  2. Orbits are started at distances dist from the saddle along R (both sides, M_mid fixed at the saddle's value),
     and a renormalized two-orbit exponent is measured (blocks of 10 returns, companion d0 = 1e-10).
     Distances are in the chart (R, 3 M_mid/delta), which reduces to (R, 100 M_mid) at delta = 0.03.
  3. The type (ii) orbit's distance from the saddle along the same line is read from its section points
     (results of bcr_delta_scan.py, start D): the crossings of the curve with M_mid = M_saddle, interpolated.
SCAN is the scan's result file (delta_scan.json) or its per-delta directory (delta_scan_parts, which must then hold
exactly one file per delta). Each exponent run is cached in OUT_parts/ under a name built from all of its parameters
and reused only if its stored parameters match; the summary collects exactly the runs requested.
usage: python bcr_layer_delta.py SCAN OUT.json [--deltas ...] [--dists ...] [--returns 4000]"""

import os, sys, json, time, argparse, hashlib, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_delta_scan as DS
from common import N

BLOCK = 10


def advance(delta, x, n):
    A, B = DS.shells(delta)
    pa, pb = N.potential(x[0], 0.0, x[1], A), N.potential(x[0], x[1], delta, B)
    if pa >= 0 or pb >= 0:
        raise ValueError("forbidden start")
    va, vb = np.sqrt(-pa), -np.sqrt(-pb)
    newmid, na, nb, _, _ = N.crossing(x[0], va, vb, 0.0, x[1], delta, A, B)
    r = N.simulate(
        [A, B],
        [0.0, newmid, delta],
        [x[0], na, x[0], nb],
        order=[1, 0],
        tmax=80.0 * n,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * delta / 0.03,
        max_step=0.02,
        max_events=40 * n,
    )
    p, t = DS.section_points(r["events"])
    if len(p) < n:
        raise RuntimeError(f'only {len(p)} of {n} returns ({r["status"]})')
    return p[:n], float(t[n - 1])


def fixed_points(delta):
    found = []
    for R0 in np.linspace(0.45, 1.3, 18):
        for f in (0.38, 0.5, 0.62):
            x = np.array([R0, f * delta])
            try:
                for _ in range(25):
                    Fx = DS.map_F(delta, x)
                    r = Fx - x
                    J = np.zeros((2, 2))
                    h = np.array([1e-6, 1e-8 * delta / 0.03])
                    for j in range(2):
                        d = np.zeros(2)
                        d[j] = h[j]
                        J[:, j] = (DS.map_F(delta, x + d) - DS.map_F(delta, x - d)) / (2 * h[j])
                    dx = np.linalg.solve(J - np.eye(2), -r)
                    x = x + dx
                    if abs(dx[0]) < 1e-12 and abs(dx[1]) < 1e-14:
                        break
                if np.linalg.norm(DS.map_F(delta, x) - x) > 1e-9:
                    continue
                ev = np.linalg.eigvals(J)
                kind = (
                    "elliptic"
                    if np.all(np.abs(np.abs(ev) - 1) < 1e-5) and np.any(np.abs(ev.imag) > 1e-8)
                    else "hyperbolic"
                )
                if all(np.hypot(x[0] - y["x"][0], (x[1] - y["x"][1]) * 3 / delta) > 1e-6 for y in found):
                    found.append(dict(x=x.tolist(), kind=kind, trace=float(np.trace(J))))
            except Exception:
                continue
    return found


def exp_params(delta, saddle, side, dist, n, d0):
    return dict(
        delta=delta,
        saddle=list(map(float, saddle)),
        side=side,
        dist=dist,
        returns=n,
        d0=d0,
        block=BLOCK,
        rtol=1e-12,
        max_step=0.02,
    )


def exp_path(outdir, params):
    tag = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(outdir, f"exp_{params['delta']:.4f}_{params['side']:+d}_{params['dist']:.1e}_{tag}.json")


def load_exp(outdir, params):
    path = exp_path(outdir, params)
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("params") == params:
            return r
    return None


def exponent(args):
    delta, saddle, side, dist, n, d0 = args
    sc = np.array([1.0, 3.0 / delta])
    x = np.array(saddle) + np.array([side * dist, 0.0])
    e = np.array([1.0, 1.0]) / sc
    e /= np.linalg.norm(e * sc)
    y = x + d0 * e
    S = 0.0
    out = []
    clock = 0.0
    t0 = time.time()
    status = "ok"
    try:
        for b in range(1, n // BLOCK + 1):
            (px, tb), (py, _) = advance(delta, x, BLOCK), advance(delta, y, BLOCK)
            clock += tb
            x, yy = px[-1], py[-1]
            d = np.linalg.norm((yy - x) * sc)
            S += np.log(d / d0)
            y = x + (yy - x) * (d0 / d)
            k = b * BLOCK
            if k in (300, 1000, 2000, 4000) or k == n:
                out.append([k, S / k])
    except Exception as ex:
        status = f"stopped: {ex}"
    res = dict(
        delta=delta,
        side=side,
        dist=dist,
        status=status,
        lambda_N=out,
        return_time=clock / max(len(out) and out[-1][0], 1),
        params=exp_params(*args),
        seconds=time.time() - t0,
    )
    print(f"delta {delta} side {side:+d} dist {dist:.0e}: {status}; {[(k, f'{l:.2e}') for k, l in out]}", flush=True)
    return res


def orbit_distance(points, saddle, delta):
    p = np.array(points)
    Ms = saddle[1]
    out = {}
    for side in (-1, 1):
        q = p[np.sign(p[:, 0] - saddle[0]) == side]
        if len(q) < 10:
            continue
        k = np.argsort(np.abs(q[:, 1] - Ms))[:6]  # nearest points to the line M = M_saddle
        a, b = q[k, 1] - Ms, q[k, 0]
        c = np.polyfit(a, b, 2) if len(k) >= 3 else [0, 0, b.mean()]
        out[side] = float(abs(np.polyval(c, 0.0) - saddle[0]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parts")
    p.add_argument("out")
    p.add_argument("--deltas", default="0.006,0.012,0.020,0.030,0.037")
    p.add_argument("--dists", default="1e-3,1e-2,3e-2,5e-2,1e-1,1.5e-1")
    p.add_argument("--returns", type=int, default=4000)
    p.add_argument("--d0", type=float, default=1e-10)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)
    deltas = [float(x) for x in a.deltas.split(",")]
    dists = [float(x) for x in a.dists.split(",")]
    summary = []
    for dl in deltas:
        fpath = os.path.join(outdir, f"fixed_{dl:.4f}.json")
        if os.path.exists(fpath):
            fps = json.load(open(fpath))
        else:
            fps = fixed_points(dl)
            json.dump(fps, open(fpath, "w"))
        sad = [f for f in fps if f["kind"] == "hyperbolic"]
        print(f"delta {dl}: fixed points", [(np.round(f["x"], 6).tolist(), f["kind"]) for f in fps], flush=True)
        if not sad:
            summary.append(dict(delta=dl, fixed_points=fps, note="no saddle found"))
            continue
        saddle = min(sad, key=lambda f: abs(f["x"][1] / dl - 0.5))["x"]
        if os.path.isdir(a.parts):
            found = sorted(glob.glob(os.path.join(a.parts, f"delta_{dl:.4f}*.json")))
            if len(found) != 1:
                raise SystemExit(f"expected one scan file for delta {dl:.4f} in {a.parts}, found {len(found)}")
            orb = json.load(open(found[0]))
        else:
            orb = next(r for r in json.load(open(a.parts))["runs"] if abs(r["delta"] - dl) < 1e-9)
        od = orbit_distance(orb["points"], saddle, dl)
        summary.append(dict(delta=dl, fixed_points=fps, saddle=saddle, orbit_distance=od))
    tasks = []
    for s in summary:
        if "saddle" not in s:
            continue
        for side in (1, -1):
            for ds in dists:
                if load_exp(outdir, exp_params(s["delta"], s["saddle"], side, ds, a.returns, a.d0)) is None:
                    tasks.append((s["delta"], s["saddle"], side, ds, a.returns, a.d0))
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(exponent, tasks):
            json.dump(r, open(exp_path(outdir, r["params"]), "w"))
    for s in summary:
        if "saddle" not in s:
            continue
        s["exponents"] = [
            load_exp(outdir, exp_params(s["delta"], s["saddle"], side, ds, a.returns, a.d0))
            for side in (1, -1)
            for ds in dists
        ]
    json.dump(
        dict(dists=dists, returns=a.returns, d0=a.d0, summary=summary, seconds=time.time() - t0), open(a.out, "w")
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
