"""Layer scans with the variational tangent (tangent_map.py) in place of the two-orbit method.

The two-orbit scans (bcr_layer_scan.py, bcr_layer_delta.py) used a companion d0 = 1e-10 away. Near the saddle the
separation is compressed by up to ~10 within a block of returns before it is stretched again, so integration noise of
~1e-14 enters the separation at the compressed stage and is then stretched; at d0 = 1e-10 this inflates the exponents
of orbits that pass close to the saddle (bcr_layer_convergence.py). The variational tangent has no companion orbit,
so it avoids the noise floor that comes from subtracting nearby trajectories. Starts (similar to, not the same as,
those of the two-orbit scans):
  example  BCR's Fig. 5 parameters: distances along R+, R-, M+, M- from the saddle, chart (R, 100 M_mid);
  delta    type (ii) family: distances along R on both sides of the saddle, chart (R, 3 M_mid/delta), saddles from
           results/bcr_section/layer_delta.json;
  family   the type (ii) orbits themselves (first section point of each run in results/bcr_section/delta_scan.json),
           chart (R, M_mid/delta) as in bcr_delta_scan.py;
  extend   the runs of results/bcr_section/layer_tangent_delta.json whose late-block exponent over returns
           2000-6000 lies between 1e-3 and 1e-2 (neither clearly regular nor clearly chaotic), rerun to --returns
           (run_all.sh passes 20000) so that a later block can decide.
Stored for each orbit: the cumulative log growth of a tangent vector (started along (1, 1) in the chart) every 10
returns, hence lambda_N and late-block exponents; the return time; every 4th section point.
usage: python bcr_layer_tangent.py example|delta|family|extend OUT.json [--returns 6000] [--workers N]"""

import os, sys, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM

HERE = os.path.dirname(os.path.abspath(__file__))
MFAC = 0.9 * 20 ** (-0.4)
EX_DISTS = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1]
DL_DISTS = [1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1, 1.5e-1]


def run(case):
    m, Mt, x0, sc, n = case["m"], case["Mt"], np.array(case["x0"]), np.array(case["scale"]), case["returns"]
    rtol, max_step = case.get("rtol", 1e-12), case.get("max_step", 0.02)  # defaults: the settings of all scans
    P = TM.TwoShellMap(m=m, w=0.2, Min=0.0, Mt=Mt, rtol=rtol, atol=1e-2 * rtol * Mt / 0.03, max_step=max_step)
    x = x0.copy()
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S = 0.0
    cum = []
    pts = [x.tolist()]
    t0 = time.time()
    status = "ok"
    clock = 0.0
    try:
        for k in range(1, n + 1):
            st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
            if st != 0:
                raise RuntimeError({1: "forbidden start", 2: "integrator failure", 3: "no return"}[st])
            clock += tr
            v = DF @ v
            g = np.linalg.norm(v * sc)
            S += np.log(g)
            v /= g
            x = x1
            if k % 10 == 0:
                cum.append(S)
            if k % 4 == 0:
                pts.append(x.tolist())
    except Exception as ex:
        status = f"stopped: {ex}"
    out = dict(
        case={k: v for k, v in case.items()},
        status=status,
        every=10,
        cumulative_log=cum,
        returns_done=10 * len(cum),
        return_time=clock / max(10 * len(cum), 1),
        points=pts,
        seconds=time.time() - t0,
    )
    print(
        f"{case['label']}: {status}; lambda_N {lambda_N(out, [1000, 2000, 4000, 6000])}; "
        f"late[N/3, N] {late(out, n//3 - (n//3) % 10, n)}; {out['seconds']:.0f}s",
        flush=True,
    )
    return out


def lambda_N(res, Ns):
    c = res["cumulative_log"]
    return [[k, float(c[k // 10 - 1] / k)] for k in Ns if 0 < k // 10 <= len(c)]


def late(res, n1, n2):
    c = res["cumulative_log"]
    if n2 // 10 > len(c) or n1 <= 0:
        return None
    return float((c[n2 // 10 - 1] - c[n1 // 10 - 1]) / (n2 - n1))


def key(case):
    return hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]


def cached(case, outdir):
    path = os.path.join(outdir, f"{key(case)}.json")
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("case") == case:
            return r
    r = run(case)
    json.dump(r, open(path, "w"))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("example", "delta", "family", "extend"))
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=6000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    cases = []
    if a.which == "example":
        S = json.load(open(os.path.join(HERE, "results", "bcr_section", "bcr_section.json")))
        saddle = np.array([fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0])
        sc = np.array([1.0, 100.0])
        for dr, step in (("R+", [1, 0]), ("R-", [-1, 0]), ("M+", [0, 1]), ("M-", [0, -1])):
            for d in EX_DISTS:
                cases.append(
                    dict(
                        label=f"{dr} {d:g}",
                        direction=dr,
                        dist=d,
                        m=0.00815,
                        Mt=0.03,
                        x0=(saddle + d * np.array(step) / sc).tolist(),
                        scale=sc.tolist(),
                        returns=a.returns,
                    )
                )
        cases.append(
            dict(
                label="BCR's example",
                direction="example",
                dist=None,
                m=0.00815,
                Mt=0.03,
                x0=S["bcr_orbit"][0]["points"][0],
                scale=sc.tolist(),
                returns=a.returns,
            )
        )
    elif a.which == "extend":
        L = json.load(open(os.path.join(HERE, "results", "bcr_section", "layer_tangent_delta.json")))
        for r in L["runs"]:
            lb = late(r, 2000, 6000)
            if lb is not None and 1e-3 < lb < 1e-2:
                c = dict(r["case"])
                c["returns"] = a.returns
                c["label"] += " (extended)"
                cases.append(c)
    elif a.which == "family":
        D = json.load(open(os.path.join(HERE, "results", "bcr_section", "delta_scan.json")))
        for r in sorted(D["runs"], key=lambda r: r["delta"]):
            dl = r["delta"]
            cases.append(
                dict(
                    label=f"type (ii) delta {dl:g}",
                    delta=dl,
                    m=MFAC * dl,
                    Mt=dl,
                    x0=r["points"][0],
                    scale=[1.0, 1.0 / dl],
                    returns=a.returns,
                )
            )
    else:
        L = json.load(open(os.path.join(HERE, "results", "bcr_section", "layer_delta.json")))
        for s in L["summary"]:
            dl = s["delta"]
            sc = np.array([1.0, 3.0 / dl])
            for side in (-1, 1):
                for d in DL_DISTS:
                    cases.append(
                        dict(
                            label=f"delta {dl:g} side {side:+d} dist {d:g}",
                            delta=dl,
                            side=side,
                            dist=d,
                            m=MFAC * dl,
                            Mt=dl,
                            x0=(np.array(s["saddle"]) + np.array([side * d, 0.0])).tolist(),
                            scale=sc.tolist(),
                            returns=a.returns,
                        )
                    )
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.starmap(cached, [(c, outdir) for c in cases], chunksize=1)
    for r in res:
        n = r["case"]["returns"]
        r["lambda_N"] = lambda_N(r, [300, 1000, 2000, 4000, 6000])
        r["late"] = {
            f"{n1}-{n2}": late(r, n1, n2)
            for n1, n2 in ((2000, 4000), (2000, 6000), (3000, 6000), (6000, 20000), (10000, 20000))
        }
    json.dump(dict(which=a.which, returns=a.returns, runs=res, seconds=time.time() - t0), open(a.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
