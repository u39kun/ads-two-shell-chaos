"""Finite-time Lyapunov exponent and curve thickness on the two-shell section (companion to bcr_section.py).

For an orbit of the return map F (bcr_section.py), propagate a tangent vector with central-difference
Jacobians of F and accumulate the log growth: lambda_N = (1/N) sum log |J v|. Regular (invariant-circle) motion gives
lambda_N ~ ln(C N)/N (linear growth of |J_N v|); chaotic motion gives a plateau lambda_N -> lambda > 0.
Orbits: the BCR Fig. 5 orbit (first section point of the D run), an orbit near the A-heavy center, and the orbit
started next to the saddle. Also: rotation number about the saddle and the thickness of each section curve
(radial residual from a Fourier fit of radius against angle about the saddle, in scaled coordinates).
usage: python bcr_section_lyapunov.py SECTION.json OUT.json [--returns 3000]"""

import os, sys, json, argparse, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS

SC = np.array([1.0, 1.0 / 0.01])  # scale M_mid (~1e-2) to order one for norms and angles
H = np.array([1e-7, 1e-9])


def jac(x):
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = H[j]
        J[:, j] = (BS.F(x + d) - BS.F(x - d)) / (2 * H[j])
    return J


def ftle(args):
    name, x0, n = args
    x = np.array(x0, float)
    v = np.array([1.0, 0.0]) / SC
    v /= np.linalg.norm(v * SC)
    S = 0.0
    lam = []
    pts = [x.copy()]
    for k in range(1, n + 1):
        J = jac(x)
        v = J @ v
        g = np.linalg.norm(v * SC)
        S += np.log(g)
        v /= g
        x = BS.F(x)
        pts.append(x.copy())
        if k in (10, 30, 100, 300, 1000, 3000, 10000) or k == n:
            lam.append([k, S / k, S])
    return dict(name=name, x0=list(x0), lambda_N=lam, points=np.array(pts).tolist())


def curve_stats(points, saddle):
    p = (np.array(points) - saddle) * SC
    th = np.arctan2(p[:, 1], p[:, 0])
    r = np.hypot(p[:, 0], p[:, 1])
    # rotation number about the saddle: mean angle advance per return / 2pi
    d = np.diff(np.unwrap(th))
    rho = float(np.mean(d) / (2 * np.pi))
    out = dict(rotation_number_about_saddle=rho)
    for K in (8, 16, 32):
        X = np.column_stack([np.ones_like(th)] + [f(k * th) for k in range(1, K + 1) for f in (np.cos, np.sin)])
        c, *_ = np.linalg.lstsq(X, r, rcond=None)
        res = r - X @ c
        out[f"thickness_rms_K{K}"] = float(np.sqrt(np.mean(res**2)) / np.mean(r))
        out[f"thickness_max_K{K}"] = float(np.max(np.abs(res)) / np.mean(r))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=3000)
    a = p.parse_args()
    t0 = time.time()
    D = json.load(open(a.section))
    saddle = np.array([fp["x"] for fp in D["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    heavy = np.array(sorted([fp["x"] for fp in D["fixed_points"] if fp["kind"] == "elliptic"], key=lambda x: x[1])[1])
    bcr0 = D["bcr_orbit"][0]["points"][0]
    near_center = [heavy[0], heavy[1] - 0.0004]
    near_saddle = [saddle[0] - 0.01, saddle[1]]
    tasks = [
        ("BCR Fig. 5 orbit", bcr0, a.returns),
        ("near A-heavy center", near_center, a.returns),
        ("next to the saddle", near_saddle, a.returns),
    ]
    with Pool(min(3, os.cpu_count())) as pool:
        res = pool.map(ftle, tasks)
    for r in res:
        print(r["name"], "lambda_N:", [(k, f"{l:.2e}") for k, l, _ in r["lambda_N"]])
    # curve statistics: BCR orbit from the long flow run; background orbits are not star-shaped about the saddle
    stats = curve_stats(D["bcr_orbit"][0]["points"], saddle)
    print("BCR section curve:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})
    json.dump(dict(ftle=res, bcr_curve=stats, saddle=saddle.tolist(), seconds=time.time() - t0), open(a.out, "w"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
