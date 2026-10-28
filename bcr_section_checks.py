"""Checks of the BCR-orbit section result: (1) long flow run and local thickness (zoom), (2) two-orbit (Benettin)
exponent on the section map without Jacobians, for the BCR orbit and a regular orbit near the center,
at two initial separations and two integration tolerances.
usage: python bcr_section_checks.py SECTION.json OUT.json"""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS
from common import N

SC = np.array([1.0, 100.0])


def F_tol(x, rtol):
    v = BS.velocities(*x)
    newmid, na, nb, _, _ = N.crossing(x[0], v[0], v[1], 0.0, x[1], BS.TOT, BS.A, BS.B)
    r = N.simulate(
        [BS.A, BS.B],
        [0.0, newmid, BS.TOT],
        [x[0], na, x[0], nb],
        order=[1, 0],
        tmax=80,
        anchor=1,
        rtol=rtol,
        atol=rtol * 1e-2,
        max_step=0.02,
        max_events=40,
    )
    p, _ = BS.section_points(r["events"])
    return p[0]


def benettin(args):
    name, x0, d0, rtol, n, renorm = args
    x = np.array(x0, float)
    dirn = np.array([1.0, 1.0]) / SC
    dirn /= np.linalg.norm(dirn * SC)
    y = x + d0 * dirn
    S = 0.0
    out = []
    for k in range(1, n + 1):
        x = F_tol(x, rtol)
        y = F_tol(y, rtol)
        if k % renorm == 0:
            d = np.linalg.norm((y - x) * SC)
            S += np.log(d / d0)
            y = x + (y - x) * (d0 / d)
        if k in (100, 300, 1000, 3000, 6000, 10000, 15000, 20000) or k == n:
            out.append([k, S / k])
    return dict(name=name, d0=d0, rtol=rtol, renorm=renorm, lambda_N=out)


def main():
    D = json.load(open(sys.argv[1]))
    t0 = time.time()
    # long flow run for local thickness
    r = BS.run_start(("D", 1.0e5))
    pts = np.array(r["points"])
    print("long run:", r["status"], len(pts), "section points", flush=True)
    saddle = np.array([fp["x"] for fp in D["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    heavy = sorted([fp["x"] for fp in D["fixed_points"] if fp["kind"] == "elliptic"], key=lambda x: x[1])[1]
    bcr0 = D["bcr_orbit"][0]["points"][0]
    tasks = []
    for d0 in (1e-8, 1e-6):
        for rtol in (1e-12, 1e-13):
            tasks.append(("BCR", bcr0, d0, rtol, 6000, 10))
    tasks.append(("center", [heavy[0], heavy[1] - 0.0004], 1e-8, 1e-12, 6000, 10))
    tasks.append(("center far", [heavy[0], heavy[1] - 0.0030], 1e-8, 1e-12, 6000, 10))
    with Pool(os.cpu_count()) as pool:
        res = pool.map(benettin, tasks)
    for x in res:
        print(x["name"], "d0", x["d0"], "rtol", x["rtol"], [(k, f"{l:.2e}") for k, l in x["lambda_N"]], flush=True)
    json.dump(
        dict(long_points=pts.tolist(), long_times=r["times"], benettin=res, seconds=time.time() - t0),
        open(sys.argv[2], "w"),
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
