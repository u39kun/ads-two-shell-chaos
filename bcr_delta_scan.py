"""Scan of BCR's type (ii) data in delta (cf. their Fig. 7): does irregular motion set in?

Type (ii) data as in bcr_validation.py: two identical shells, w = 0.2, m = 0.9*20^(-0.4)*delta, region masses
0, delta/2, delta, both at R = 1.5, inner contracting and outer expanding (BCR_START=D, default), or inner
expanding and outer contracting (BCR_START=C: the shells cross at t = 0 and the run starts just after the
crossing). For each delta:
  - a reference run to tmax and a run with both initial radii moved by eps; section points are head-on crossings
    with A inside, A moving out (chart (R, M_mid/delta)); the k-th section points of the two runs are compared, and
    the growth of their distance d_k with k separates linear (regular) from exponential (chaotic) divergence;
  - a renormalized two-orbit exponent on the section map (lambda_N at several N; regular motion gives ~ ln N/N);
  - the section points themselves, for a bifurcation-style plot of section values against delta.
Results are written one file per delta (OUT_parts/), so an interrupted scan resumes where it stopped. Each file is
named by delta and a hash of all run parameters, stores those parameters, and is reused only if they match.
usage: python bcr_delta_scan.py OUT.json [--tmax 20000] [--eps 1e-9] [--returns 4000] [--deltas a,b,...]"""

import os, sys, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from common import N

W, RI = 0.2, 1.5
MFAC = 0.9 * 20 ** (-2 * W)
SC_R = 1.0


def shells(delta):
    m = MFAC * delta
    return N.Shell("A", m, W), N.Shell("B", m, W)


def section_points(events):
    pts, ts = [], []
    for e in events:
        if e["loser"] == "A" and e["v_before"][0] > 0 > e["v_before"][1]:
            pts.append([e["r"], e["masses_before"][1]])
            ts.append(e["t"])
    return np.array(pts).reshape(-1, 2), np.array(ts)


START = os.environ.get("BCR_START", "D")


def flow(delta, ri, tmax):
    A, B = shells(delta)
    sa = np.sqrt(-N.potential(ri, 0.0, delta / 2, A))
    sb = np.sqrt(-N.potential(ri, delta / 2, delta, B))
    if START == "D":
        masses, state, order = [0.0, delta / 2, delta], [ri, -sa, ri, sb], [0, 1]
    else:
        mid, va, vb, _, _ = N.crossing(ri, sa, -sb, 0.0, delta / 2, delta, A, B)
        masses, state, order = [0.0, mid, delta], [ri, va, ri, vb], [1, 0]
    r = N.simulate(
        [A, B],
        masses,
        state,
        order=order,
        tmax=tmax,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * delta / 0.03,
        max_step=0.02,
        max_events=2_000_000,
    )
    p, t = section_points(r["events"])
    return r["status"], r["t_end"], p, t


def map_F(delta, x):
    A, B = shells(delta)
    pa, pb = N.potential(x[0], 0.0, x[1], A), N.potential(x[0], x[1], delta, B)
    if pa >= 0 or pb >= 0:
        raise ValueError("forbidden")
    va, vb = np.sqrt(-pa), -np.sqrt(-pb)
    newmid, na, nb, _, _ = N.crossing(x[0], va, vb, 0.0, x[1], delta, A, B)
    r = N.simulate(
        [A, B],
        [0.0, newmid, delta],
        [x[0], na, x[0], nb],
        order=[1, 0],
        tmax=200,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * delta / 0.03,
        max_step=0.02,
        max_events=60,
    )
    p, _ = section_points(r["events"])
    if len(p) == 0:
        raise RuntimeError("no return")
    return p[0]


def benettin(delta, x0, n, d0=1e-9, renorm=10):
    sc = np.array([1.0, 1.0 / delta])
    x = np.array(x0, float)
    dirn = np.array([1.0, 1.0]) / sc
    dirn /= np.linalg.norm(dirn * sc)
    y = x + d0 * dirn
    S = 0.0
    out = []
    for k in range(1, n + 1):
        x = map_F(delta, x)
        y = map_F(delta, y)
        if k % renorm == 0:
            d = np.linalg.norm((y - x) * sc)
            S += np.log(d / d0)
            y = x + (y - x) * (d0 / d)
        if k in (100, 300, 1000, 2000, 4000, 8000) or k == n:
            out.append([k, S / k])
    return out


def one(args):
    delta, tmax, eps, nret = args
    t0 = time.time()
    st, te, p, tt = flow(delta, RI, tmax)
    st2, te2, q, _ = flow(delta, RI + eps, tmax)
    res = dict(delta=delta, status=st, t_end=te, n_section=len(p), status_nudged=st2)
    if len(p) > 50 and len(q) > 50:
        n = min(len(p), len(q))
        sc = np.array([1.0, 1.0 / delta])
        d = np.linalg.norm((p[:n] - q[:n]) * sc, axis=1)
        env = np.maximum.accumulate(d)
        ks = [k for k in (10, 30, 100, 300, 1000, 3000) if k < n] + [n - 1]
        res["env"] = [[k, float(env[k])] for k in ks]
        k = (np.arange(n) >= 30) & (env < 1e-3) & (env > 0)
        if k.sum() > 20:
            kk = np.arange(n)[k]
            res["loglog_slope"] = float(np.polyfit(np.log(kk), np.log(env[k]), 1)[0])
        res["return_time"] = float((tt[-1] - tt[0]) / (len(tt) - 1)) if len(tt) > 1 else None
        try:
            res["benettin"] = benettin(delta, p[0], nret)
        except Exception as ex:
            res["benettin_error"] = str(ex)
        res["points"] = p[:4000].tolist()
    res["seconds"] = time.time() - t0
    print(
        f"delta {delta:.4f}: {st}, {len(p)} section pts, slope {res.get('loglog_slope')}, "
        f"lambda_N {[(k, round(l, 5)) for k, l in res.get('benettin', [])]}, {res['seconds']:.0f}s",
        flush=True,
    )
    return res


def run_params(delta, tmax, eps, nret):
    """Everything that determines a case's result; part of its cache key."""
    return dict(
        delta=delta,
        tmax=tmax,
        eps=eps,
        returns=nret,
        start=START,
        R_i=RI,
        w=W,
        rtol=1e-12,
        max_step=0.02,
        d0=1e-9,
        renorm=10,
    )


def part_path(outdir, params):
    tag = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(outdir, f"delta_{params['delta']:.4f}_{tag}.json")


def one_saved(args):
    delta, tmax, eps, nret, outdir = args
    params = run_params(delta, tmax, eps, nret)
    path = part_path(outdir, params)
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("params") == params:
            return r
    r = one((delta, tmax, eps, nret))
    r["params"] = params
    json.dump(r, open(path, "w"))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=20000.0)
    p.add_argument("--eps", type=float, default=1e-9)
    p.add_argument("--returns", type=int, default=4000)
    p.add_argument("--deltas", default=",".join(f"{x:.4f}" for x in np.arange(0.001, 0.03405, 0.001)) + ",0.0360,0.0370")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)  # one file per delta: restartable
    with Pool(a.workers) as pool:
        res = pool.map(one_saved, [(d, a.tmax, a.eps, a.returns, outdir) for d in deltas], chunksize=1)
    json.dump(
        dict(params=dict(W=W, RI=RI, tmax=a.tmax, eps=a.eps, returns=a.returns, start=START), runs=res),
        open(a.out, "w"),
    )
    print("done")


if __name__ == "__main__":
    main()
