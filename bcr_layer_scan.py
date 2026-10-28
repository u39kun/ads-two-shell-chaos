"""Where is the chaos? Orbits started near the separatrix of the two-shell map at BCR's Fig. 5 parameters.

The two-shell section map (bcr_section.py: head-on crossings with A inside, chart (R, M_mid)) has a saddle whose
figure-eight separatrix separates motion around each elliptic point (inside a lobe) from motion around both (outside).
A nonintegrable area-preserving map typically has a chaotic layer along such a separatrix (not shown here). We start
orbits at distances dist from the saddle along four directions (R+ and R-: outside the lobes; M+ and M-: inside the
upper and lower lobes; distances in the chart with M_mid scaled by 100) and measure a renormalized two-orbit exponent:
blocks of 10 returns are integrated for the orbit and for a companion d0 away, after which the companion is pulled
back to distance d0 along the separation.
Regular motion gives lambda_N ~ ln(C N)/N, which keeps falling as N grows; chaotic motion gives a plateau.
Each case is cached in OUT_parts/ under a name built from all of its parameters (direction, distance, returns, d0,
block, tolerances, saddle) and reused only if its stored parameters match.
usage: python bcr_layer_scan.py SECTION.json OUT.json [--returns 4000] [--dists 1e-1,...]"""

import os, sys, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS
from common import N

SC = np.array([1.0, 100.0])
BLOCK = 10


def advance(x, n):
    """n returns of the section map from x; returns the n section points."""
    v = BS.velocities(*x)
    if v is None:
        raise ValueError("forbidden start")
    newmid, na, nb, _, _ = N.crossing(x[0], v[0], v[1], 0.0, x[1], BS.TOT, BS.A, BS.B)
    r = N.simulate(
        [BS.A, BS.B],
        [0.0, newmid, BS.TOT],
        [x[0], na, x[0], nb],
        order=[1, 0],
        tmax=60.0 * n,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=40 * n,
    )
    p, t = BS.section_points(r["events"])
    if len(p) < n:
        raise RuntimeError(f'only {len(p)} of {n} returns ({r["status"]})')
    return p[:n], float(t[n - 1])


def case_params(direction, dist, n, d0, saddle):
    return dict(
        direction=direction,
        dist=dist,
        returns=n,
        d0=d0,
        block=BLOCK,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        saddle=list(map(float, saddle)),
    )


def part_path(outdir, params):
    tag = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(outdir, f"{params['direction']}_{params['dist']:.1e}_{tag}.json")


def load_part(outdir, params):
    path = part_path(outdir, params)
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("params") == params:
            return r
    return None


def run(args):
    direction, dist, n, d0 = args
    saddle = np.array(SADDLE)
    step = {"R+": [1, 0], "R-": [-1, 0], "M+": [0, 1], "M-": [0, -1]}[direction]
    x = saddle + dist * np.array(step) / SC
    e = np.array([1.0, 1.0]) / SC
    e /= np.linalg.norm(e * SC)
    y = x + d0 * e
    S = 0.0
    out = []
    pts = [x.tolist()]
    t0 = time.time()
    clock = 0.0
    try:
        for b in range(1, n // BLOCK + 1):
            (px, tb), (py, _) = advance(x, BLOCK), advance(y, BLOCK)
            clock += tb
            x, yy = px[-1], py[-1]
            d = np.linalg.norm((yy - x) * SC)
            S += np.log(d / d0)
            y = x + (yy - x) * (d0 / d)
            pts += px.tolist()
            k = b * BLOCK
            if k in (100, 300, 1000, 2000, 4000, 6000, 8000) or k == n:
                out.append([k, S / k])
        status = "ok"
    except Exception as ex:
        status = f"stopped: {ex}"
    p = np.array(pts)
    res = dict(
        direction=direction,
        dist=dist,
        d0=d0,
        status=status,
        lambda_N=out,
        n_points=len(p),
        params=case_params(direction, dist, n, d0, SADDLE),
        return_time=clock / max(len(p) - 1, 1),
        R_range=[float(p[:, 0].min()), float(p[:, 0].max())],
        M_range=[float(p[:, 1].min()), float(p[:, 1].max())],
        points=p[::2].tolist(),
        seconds=time.time() - t0,
    )
    print(
        f"{direction} dist {dist:.1e}: {status}; lambda_N {[(k, f'{l:.2e}') for k, l in out]}; {res['seconds']:.0f}s",
        flush=True,
    )
    return res


def init(saddle):
    global SADDLE
    SADDLE = saddle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=4000)
    p.add_argument("--dists", default=",".join(f"{x:.0e}" for x in np.logspace(-1, -9, 9)))
    p.add_argument("--dirs", default="R+,R-,M+,M-")
    p.add_argument("--d0", type=float, default=1e-10)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    S = json.load(open(a.section))
    saddle = [fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0]
    tasks = [(dr, float(ds), a.returns, a.d0) for ds in a.dists.split(",") for dr in a.dirs.split(",")]
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)
    todo = [t for t in tasks if load_part(outdir, case_params(*t, saddle)) is None]
    with Pool(a.workers, initializer=init, initargs=(saddle,)) as pool:
        for r in pool.imap_unordered(run, todo):
            json.dump(r, open(part_path(outdir, r["params"]), "w"))
    res = [load_part(outdir, case_params(*t, saddle)) for t in tasks]
    json.dump(dict(saddle=saddle, returns=a.returns, d0=a.d0, block=BLOCK, runs=res), open(a.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
