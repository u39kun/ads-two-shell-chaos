"""Why the finite-difference Jacobian product disagrees with the two-orbit exponent on BCR's example (App. C).

Section map F of BCR's Fig. 5 parameters (bcr_section.py), chart (R, 100 M_mid) for norms. Three ways to follow a
tangent vector along the example's orbit:
  * variational tangent: DF from the variational equations with crossing saltation (tangent_map.py), no finite
    differences;
  * finite-difference Jacobians, central differences with step h in the scaled chart (bcr_section_lyapunov.py used
    h = 1e-7 for both coordinates), for several h and integration tolerances;
  * the two-orbit (Benettin) results already in results/bcr_section (bcr_checks.json, bcr_benettin_long.json).
Checks of the variational tangent: it agrees with central differences at generic points, and det DF(x) = g(x)/g(F(x)).
Diagnosis: the per-return error of the finite-difference Jacobian against the variational one, against the distance
of the section point from the nearest chart fold (a shell at its turning point when the shells cross, where g diverges).
usage: python bcr_jacobian_check.py OUT.json [--returns 3000] [--long 20000] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
from common import N

HERE = os.path.dirname(os.path.abspath(__file__))
SC = np.array([1.0, 100.0])
M5 = dict(m=0.00815, w=0.2, Min=0.0, Mt=0.03)  # BCR Fig. 5 parameters
MEAS = dict(m=0.9 * 20 ** (-0.4) * 0.03, w=0.2, Min=0.0, Mt=0.03)  # parameters of two_shell_measure.py
CHECKS = [(0.800, 0.018), (0.850, 0.014)]


def scaled(J):
    return np.diag(SC) @ J @ np.diag(1 / SC)


def fd_jac(P, x, h):
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = h / SC[j]
        J[:, j] = (P(x + d) - P(x - d)) / (2 * d[j])
    return J


def fold_distance(P, x):
    """Distance in R from x to the nearest fold of the chart: min over the shells of v^2/|V'(R)|."""
    R, M = x
    va, vb = P.speeds(x)
    out = []
    for v, lo, hi in ((va, P.Min, M), (vb, M, P.Mt)):
        dV = float(N.derivative(R, lo, hi, N.Shell("X", P.m[0], P.w[0])))
        out.append(v * v / abs(dV))
    return float(min(out))


def validation():
    out = []
    for name, par, pts in (
        ("measure parameters", MEAS, CHECKS),
        ("Fig. 5 parameters", M5, [(0.6743747653549492, 0.015)]),
    ):
        P = TM.TwoShellMap(**par)
        for x in pts:
            x = np.array(x)
            x1, DF = P(x, tangent=True)
            fd = {
                f"{h:g}": float(np.abs(scaled(fd_jac(P, x, h)) - scaled(DF)).max() / np.abs(scaled(DF)).max())
                for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
            }
            out.append(
                dict(
                    parameters=name,
                    x=x.tolist(),
                    det=float(np.linalg.det(DF)),
                    g_ratio=float(P.g(x) / P.g(x1)),
                    log_residual=float(np.log(np.linalg.det(DF)) - np.log(P.g(x) / P.g(x1))),
                    fd_rel_error=fd,
                )
            )
            print(out[-1], flush=True)
    return out


def exact_orbit(args):  # 'exact' (here and in the saved results) = the variational tangent
    """Variational-tangent exponent along the orbit, plus per-return data for the diagnosis."""
    x0, n, v0 = args
    P = TM.TwoShellMap(**M5)
    x = np.array(x0, float)
    v = np.array(v0, float) / SC
    v /= np.linalg.norm(v * SC)
    S = 0.0
    lam = []
    logdet = 0.0
    resid = []
    rows = []
    for k in range(1, n + 1):
        x1, DF = P(x, tangent=True)
        v = DF @ v
        gk = np.linalg.norm(v * SC)
        S += np.log(gk)
        v /= gk
        r = float(np.log(abs(np.linalg.det(DF))) - np.log(P.g(x) / P.g(x1)))
        resid.append(abs(r))
        if v0 == (1.0, 0.0) and k <= 3000:
            rows.append([x[0], x[1], fold_distance(P, x)] + DF.ravel().tolist())
        x = x1
        if k in (10, 30, 100, 300, 1000, 3000, 6000, 10000, 15000, 20000) or k == n:
            lam.append([k, S / k])
    resid = np.array(resid)
    return dict(
        v0=list(v0),
        lambda_N=lam,
        measure_residual_max=float(resid.max()),
        measure_residual_median=float(np.median(resid)),
        rows=rows,
    )


def fd_orbit(args):
    """Finite-difference Jacobian product along the same orbit (as bcr_section_lyapunov.py), step h, tolerance rtol."""
    x0, n, h, rtol, rows = args
    P = TM.TwoShellMap(**M5, rtol=rtol)
    x = np.array(x0, float)
    v = np.array([1.0, 0.0]) / SC
    v /= np.linalg.norm(v * SC)
    S = 0.0
    lam = []
    err = []
    cum = []
    stopped = None
    for k in range(1, n + 1):
        try:
            J = fd_jac(P, x, h)
        except RuntimeError as ex:  # a stencil point lies beyond a fold of the chart
            stopped = dict(at_return=k, reason=str(ex), fold_distance=fold_distance(P, x))
            break
        if rows is not None:
            Jx = np.array(rows[k - 1][3:]).reshape(2, 2)
            err.append(float(np.abs(scaled(J) - scaled(Jx)).max() / np.abs(scaled(Jx)).max()))
        v = J @ v
        gk = np.linalg.norm(v * SC)
        S += np.log(gk)
        v /= gk
        cum.append(S)
        x = P(x)
        if k in (10, 30, 100, 300, 1000, 3000) or k == n:
            lam.append([k, S / k])
    print(f"FD h {h:g} rtol {rtol:g}: {[(k, round(l, 6)) for k, l in lam]}; stopped {stopped}", flush=True)
    return dict(h=h, rtol=rtol, lambda_N=lam, rel_error=err, cumulative_log=cum[::10], stopped=stopped)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=3000)
    p.add_argument("--long", type=int, default=20000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    S = json.load(open(os.path.join(HERE, "results", "bcr_section", "bcr_section.json")))
    x0 = S["bcr_orbit"][0]["points"][0]
    out = dict(x0=x0, chart_scale=SC.tolist(), validation=validation())
    with Pool(min(a.workers, 2)) as pool:
        ex = pool.map(exact_orbit, [(x0, a.long, (1.0, 0.0)), (x0, a.long, (1.0, 1.0))])
    for r in ex:
        print(
            "variational tangent, v0",
            r["v0"],
            [(k, f"{l:.3e}") for k, l in r["lambda_N"]],
            "measure residual max",
            r["measure_residual_max"],
            flush=True,
        )
    rows = ex[0]["rows"][: a.returns]
    out["exact"] = [{k: v for k, v in r.items() if k != "rows"} for r in ex]
    tasks = [(x0, a.returns, h, 1e-12, rows) for h in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)]
    tasks += [(x0, a.returns, 1e-7, rt, None) for rt in (1e-11, 1e-13)]
    with Pool(a.workers) as pool:
        fd = pool.map(fd_orbit, tasks)
    # diagnosis: error of the h = 1e-7 Jacobians against the fold distance of the section point
    r7 = [r for r in fd if r["h"] == 1e-7 and r["rtol"] == 1e-12][0]
    e7 = np.array(r7["rel_error"])
    dist = np.array([q[2] for q in rows])[: len(e7)]
    bins = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e9]
    out["fold_diagnosis"] = dict(
        bins=bins,
        median_error=[
            float(np.median(e7[(dist >= lo) & (dist < hi)])) if ((dist >= lo) & (dist < hi)).any() else None
            for lo, hi in zip(bins[:-1], bins[1:])
        ],
        count=[int(((dist >= lo) & (dist < hi)).sum()) for lo, hi in zip(bins[:-1], bins[1:])],
        dist=dist.tolist(),
        err_h1e7=e7.tolist(),
    )
    out["fd"] = [{k: v for k, v in r.items()} for r in fd]
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"))
    print(
        "fold diagnosis (h = 1e-7):", list(zip(out["fold_diagnosis"]["count"], out["fold_diagnosis"]["median_error"]))
    )
    print(f'done in {out["seconds"]:.0f}s')


if __name__ == "__main__":
    main()
