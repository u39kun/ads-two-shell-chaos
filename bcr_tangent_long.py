"""Long finite-time exponents from the variational tangent (tangent_map.py) at BCR's Fig. 5 parameters.

Independent of the two-orbit method: no companion orbit, no renormalization distance, and no finite differences.
Orbits are started at distances from the saddle along R+, R-, M+ or M- (chart (R, 100 M_mid), as in
bcr_layer_scan.py) and at BCR's example; the tangent vector starts along (1, 1) in the scaled chart. Stored: the
cumulative log growth every 10 returns, from which exponents over late blocks [N1, N2] follow.
usage: python bcr_tangent_long.py SECTION.json OUT.json [--returns 50000] [--starts R+:1e-4,M-:1e-4,R+:1e-2,...]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM

SC = np.array([1.0, 100.0])


def run(args):
    label, x0, n = args
    P = TM.TwoShellMap(m=0.00815, w=0.2, Min=0.0, Mt=0.03)
    x = np.array(x0, float)
    v = np.array([1.0, 1.0]) / SC
    v /= np.linalg.norm(v * SC)
    S = 0.0
    cum = []
    t0 = time.time()
    status = "ok"
    try:
        for k in range(1, n + 1):
            x, DF = P(x, tangent=True)
            v = DF @ v
            g = np.linalg.norm(v * SC)
            S += np.log(g)
            v /= g
            if k % 10 == 0:
                cum.append(S)
    except Exception as ex:
        status = f"stopped: {ex}"
    out = dict(
        label=label, x0=list(map(float, x0)), status=status, every=10, cumulative_log=cum, seconds=time.time() - t0
    )
    c = np.array(cum)
    late = lambda a, b: float((c[b // 10 - 1] - c[a // 10 - 1]) / (b - a)) if len(c) >= b // 10 else None
    out["lambda_N"] = [
        [k, float(c[k // 10 - 1] / k)] for k in (1000, 2000, 6000, 10000, 20000, 50000) if len(c) >= k // 10
    ]
    out["late"] = {"2000-6000": late(2000, 6000), "10000-50000": late(10000, 50000)}
    print(
        label, status, [(k, f"{l:.2e}") for k, l in out["lambda_N"]], out["late"], f"{out['seconds']:.0f}s", flush=True
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=50000)
    p.add_argument("--starts", default="R+:1e-4,M-:1e-4,R+:1e-2,R-:2e-2,R+:7e-2,R+:0.1,R+:0.15")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    S = json.load(open(a.section))
    saddle = np.array([fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    steps = {"R+": [1, 0], "R-": [-1, 0], "M+": [0, 1], "M-": [0, -1]}
    tasks = []
    for item in a.starts.split(","):
        dr, d = item.split(":")
        tasks.append((f"{dr} {float(d):g}", (saddle + float(d) * np.array(steps[dr]) / SC).tolist(), a.returns))
    tasks.append(("BCR's example", S["bcr_orbit"][0]["points"][0], a.returns))
    with Pool(a.workers) as pool:
        res = pool.map(run, tasks, chunksize=1)
    json.dump(dict(saddle=saddle.tolist(), returns=a.returns, runs=res), open(a.out, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
