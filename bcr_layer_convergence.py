"""Convergence of the two-orbit exponents in and near the chaotic layer at BCR's Fig. 5 parameters
(App. C and Table 3 of the paper).

The layer scans (bcr_layer_scan.py, bcr_layer_delta.py) use one setting: companion distance d0 = 1e-10 in the chart
(R, 100 M_mid), renormalization every 10 returns, rtol = 1e-12 (atol = rtol/100), max_step = 0.02, and initial
separation along (1, 1). Here representative orbits are rerun with each of these changed in turn:
    d0 = 1e-8, 1e-12;  renormalization every 1 or 50 returns;  rtol = 1e-11, 1e-13;  max_step = 0.01;  and
    initial separation along (1, -1).
For every orbit, d0 is also set to 1e-6, 1e-7, 1e-9, and 1e-11.
Orbits (start distance from the saddle along a direction, as in bcr_layer_scan.py): in the layer (R+ and M- at 1e-4),
near the saddle but regular (R+ at 1e-2, R- at 2e-2), farther out (R+ at 0.07), and BCR's example. With the baseline
setting, the orbits in LONG are continued to --long returns (long runs with the variational tangent:
bcr_tangent_long.py).
The cumulative log growth is stored at every renormalization, so exponents over any late block of returns [N1, N2]
can be formed: lambda[N1, N2] = (S(N2) - S(N1))/(N2 - N1).
The map is integrated only as far as the returns needed (tmax grown on demand); this gives the same section points,
bit for bit, as the fixed tmax = 60 n of bcr_layer_scan.py, which the baseline run at R+ 1e-4 reproduces.
Each case is cached in OUT_parts/ under a key built from all of its parameters, and a cached file is used only if its
stored parameters match.
usage: python bcr_layer_convergence.py SECTION.json OUT.json [--returns 6000] [--long 50000] [--workers N]"""

import os, sys, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS
from common import N

SC = np.array([1.0, 100.0])
BASE = dict(d0=1e-10, renorm=10, rtol=1e-12, max_step=0.02, sep=[1.0, 1.0])
VARIANTS = [
    dict(d0=1e-8),
    dict(d0=1e-12),
    dict(renorm=1),
    dict(renorm=50),
    dict(rtol=1e-11),
    dict(rtol=1e-13),
    dict(max_step=0.01),
    dict(sep=[1.0, -1.0]),
]
D0_SWEEP = [1e-6, 1e-7, 1e-9, 1e-11]  # extra companion distances, for every orbit
ORBITS = [
    ("layer", "R+", 1e-4),
    ("layer", "M-", 1e-4),
    ("edge", "R+", 1e-2),
    ("edge", "R-", 2e-2),
    ("outside", "R+", 7e-2),
    ("example", "BCR", None),
]
LONG = [("layer", "R+", 1e-4), ("edge", "R+", 1e-2), ("outside", "R+", 7e-2)]


def advance(x, n, rtol, max_step):
    """n returns of the section map from x (same section points as bcr_layer_scan.advance for the same settings)."""
    v = BS.velocities(*x)
    if v is None:
        raise ValueError("forbidden start")
    newmid, na, nb, _, _ = N.crossing(x[0], v[0], v[1], 0.0, x[1], BS.TOT, BS.A, BS.B)
    tmax = 6.0 * n + 20.0
    while True:
        r = N.simulate(
            [BS.A, BS.B],
            [0.0, newmid, BS.TOT],
            [x[0], na, x[0], nb],
            order=[1, 0],
            tmax=tmax,
            anchor=1,
            rtol=rtol,
            atol=rtol * 1e-2,
            max_step=max_step,
            max_events=40 * n,
        )
        p, t = BS.section_points(r["events"])
        if len(p) >= n:
            return p[:n], float(t[n - 1])
        if r["status"] != "time_limit" or tmax >= 60.0 * n:
            raise RuntimeError(f'only {len(p)} of {n} returns ({r["status"]})')
        tmax = min(2 * tmax, 60.0 * n)


def start_point(kind, direction, dist, saddle, bcr0):
    if kind == "example":
        return np.array(bcr0, float)
    step = {"R+": [1, 0], "R-": [-1, 0], "M+": [0, 1], "M-": [0, -1]}[direction]
    return np.array(saddle) + dist * np.array(step) / SC


def key(case):
    s = json.dumps({k: case[k] for k in sorted(case) if k not in ("saddle", "bcr0")}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def run(case):
    x = start_point(case["kind"], case["direction"], case["dist"], case["saddle"], case["bcr0"])
    e = np.array(case["sep"], float) / SC
    e /= np.linalg.norm(e * SC)
    d0, B, n = case["d0"], case["renorm"], case["returns"]
    y = x + d0 * e
    S = 0.0
    cum = []
    t0 = time.time()
    status = "ok"
    clock = 0.0
    try:
        for b in range(1, n // B + 1):
            (px, tb), (py, _) = advance(x, B, case["rtol"], case["max_step"]), advance(
                y, B, case["rtol"], case["max_step"]
            )
            clock += tb
            x, yy = px[-1], py[-1]
            d = np.linalg.norm((yy - x) * SC)
            S += np.log(d / d0)
            y = x + (yy - x) * (d0 / d)
            cum.append(S)
    except Exception as ex:
        status = f"stopped: {ex}"
    res = dict(
        case={k: v for k, v in case.items() if k not in ("saddle", "bcr0")},
        status=status,
        block=B,
        cumulative_log=cum,
        returns_done=len(cum) * B,
        return_time=clock / max(len(cum) * B, 1),
        seconds=time.time() - t0,
    )
    lam = lambda_at(res, [1000, 2000, 4000, 6000, n])
    print(
        f"{case['kind']:8s} {case['direction']} {case['dist']} | {variant_name(case)}: {status}; "
        f"lambda_N {[(k, f'{l:.2e}') for k, l in lam]}; {res['seconds']:.0f}s",
        flush=True,
    )
    return res


def lambda_at(res, Ns):
    S = np.array(res["cumulative_log"])
    B = res["block"]
    return [(k, float(S[k // B - 1] / k)) for k in Ns if k % B == 0 and 0 < k // B <= len(S)]


def late(res, n1, n2):
    S = np.array(res["cumulative_log"])
    B = res["block"]
    if n2 // B > len(S) or n1 % B or n2 % B:
        return None
    return float((S[n2 // B - 1] - (S[n1 // B - 1] if n1 else 0.0)) / (n2 - n1))


def variant_name(case):
    diff = [k for k in BASE if case[k] != BASE[k]]
    return "baseline" if not diff else ", ".join(f"{k}={case[k]}" for k in diff)


def cached(case, outdir):
    path = os.path.join(outdir, f"{key(case)}.json")
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("case") == json.loads(json.dumps({k: v for k, v in case.items() if k not in ("saddle", "bcr0")})):
            return r
    r = run(case)
    json.dump(r, open(path, "w"))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=6000)
    p.add_argument("--long", type=int, default=50000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    S = json.load(open(a.section))
    saddle = [fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0]
    bcr0 = S["bcr_orbit"][0]["points"][0]
    cases = []
    for kind, direction, dist in ORBITS:
        for var in [{}] + VARIANTS:
            c = dict(BASE, **var)
            c["sep"] = list(c["sep"])
            cases.append(
                dict(c, kind=kind, direction=direction, dist=dist, returns=a.returns, saddle=saddle, bcr0=bcr0)
            )
    for kind, direction, dist in ORBITS:
        for d0 in D0_SWEEP:
            c = dict(BASE, d0=d0)
            c["sep"] = list(c["sep"])
            cases.append(
                dict(c, kind=kind, direction=direction, dist=dist, returns=a.returns, saddle=saddle, bcr0=bcr0)
            )
    for kind, direction, dist in LONG:
        c = dict(BASE)
        c["sep"] = list(c["sep"])
        cases.append(dict(c, kind=kind, direction=direction, dist=dist, returns=a.long, saddle=saddle, bcr0=bcr0))
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)
    cases.sort(key=lambda c: -c["returns"] / c["renorm"] ** 0.3)  # long and slow cases first
    with Pool(a.workers) as pool:
        res = pool.starmap(cached, [(c, outdir) for c in cases], chunksize=1)
    summary = []
    for r in res:
        c = r["case"]
        n = r["returns_done"]
        row = dict(
            kind=c["kind"],
            direction=c["direction"],
            dist=c["dist"],
            variant=variant_name(c),
            returns=n,
            status=r["status"],
            lambda_N=lambda_at(r, [1000, 2000, 4000, 6000, 10000, 20000, 50000]),
            late_2000_4000=late(r, 2000, 4000),
            late_2000_6000=late(r, 2000, 6000),
            late_4000_6000=late(r, 4000, 6000),
            late_10000_50000=late(r, 10000, 50000),
            return_time=r["return_time"],
        )
        summary.append(row)
    json.dump(
        dict(saddle=saddle, base=BASE, variants=VARIANTS, summary=summary, runs=res, seconds=time.time() - t0),
        open(a.out, "w"),
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
