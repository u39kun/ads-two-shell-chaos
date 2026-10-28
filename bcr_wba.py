"""Weighted Birkhoff averages (WBA) on the head-on section and at all crossings with A inside.

The head-on section (bcr_section.py) keeps only crossings with A inside, A moving out, and B moving in. Along an orbit
that also overtakes (both shells moving in, or both out, at a crossing with A inside), the head-on return map changes
branch: a return then takes 8 or 10 crossings instead of 2, and the sampled sequence is the restriction of a smooth
map to a subset, which spoils the super-convergence of WBA on quasi-periodic orbits. The set of all crossings with A
inside is a smooth section of the flow (away from grazing), so WBA of smooth observables super-converge there on a
quasi-periodic orbit. The observables are recorded along trajectories; R and M_mid alone do not identify the state
on this larger section (the velocity signs differ between its branches), and no map is built from them.

Diagnostic: d(n) = -log10 |W_n - W_2n| / |W_n|, with W_n the WBA (weight exp(-1/(t(1-t)))) of an observable over the
first n samples; also the agreement of the WBA over the two halves of the first 2n samples. Observables: R, M_mid,
and the time since the previous sample.

Orbits (each with both samplings):
  example   BCR's Fig. 5 example, m = 0.00815, both velocity assignments (D, C of bcr_section.py);
  tolerance start D again at other integration settings (relative tolerance, step-size limit), to see whether the
            level at which the agreement stops improving moves with integration accuracy;
  control   starts on the head-on section at the example's parameters: 1e-4 from the saddle along R+ (strongly
            chaotic in Sec. 3.2), 1e-2 and 0.07 from it along R+ (regular), and 1e-3 from an elliptic point along R;
  family    the type (ii) data of bcr_delta_scan.py (m = 0.9*20^(-0.4)*delta, R_i = 1.5) on its delta grid, both
            velocity assignments.
usage: python bcr_wba.py SECTION.json OUT.json [--n 8000] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS
import bcr_delta_scan as DS
from common import N

NS = (500, 1000, 2000, 4000, 8000)
# (relative tolerance, step-size limit); the default is (1e-12, 0.02)
TOLERANCES = [(1e-13, 0.02), (1e-11, 0.02), (1e-10, 0.02), (1e-8, 0.02), (1e-12, 0.005), (1e-12, 0.05), (1e-12, 0.2),
              (1e-8, 0.5), (1e-6, 1.0)]


def digits(x, n):
    u, v = BS.wb(x[:n]), BS.wb(x[: 2 * n])
    return float(-np.log10(abs(u - v) / abs(u))) if u != v else 16.0


def halves(x, n):
    u, v = BS.wb(x[:n]), BS.wb(x[n : 2 * n])
    return float(-np.log10(abs(u - v) / abs(u))) if u != v else 16.0


def table(R, M, t, nmax):
    obs = dict(R=R[1:], M_mid=M[1:], dt=np.diff(t))  # drop the first sample so all three have the same length
    out = {}
    for n in NS:
        if n > nmax or len(obs["R"]) < 2 * n:
            break
        out[str(n)] = {k: dict(n_vs_2n=digits(v, n), halves=halves(v, n)) for k, v in obs.items()}
    return out


def analyse(events, nmax):
    ain = [e for e in events if e["loser"] == "A"]
    kind = ["head_on" if e["v_before"][0] > 0 > e["v_before"][1] else
            "both_in" if e["v_before"][0] < 0 and e["v_before"][1] < 0 else
            "both_out" if e["v_before"][0] > 0 and e["v_before"][1] > 0 else "other" for e in ain]
    idx = {id(e): i for i, e in enumerate(events)}
    hi = [idx[id(e)] for e, k in zip(ain, kind) if k == "head_on"]
    per = np.diff(hi)
    head = [e for e, k in zip(ain, kind) if k == "head_on"]
    th = np.array([e["t"] for e in head])
    res = dict(
        crossings=len(events), a_inside=len(ain), counts={k: kind.count(k) for k in sorted(set(kind))},
        crossings_per_return={str(int(k)): int(c) for k, c in zip(*np.unique(per, return_counts=True))},
        return_time=dict(mean=float(np.mean(np.diff(th))), min=float(np.min(np.diff(th))),
                         max=float(np.max(np.diff(th)))) if len(th) > 2 else None,
    )
    for name, es in (("all_a_inside", ain), ("head_on", head)):
        R = np.array([e["r"] for e in es])
        M = np.array([e["masses_before"][1] for e in es])
        t = np.array([e["t"] for e in es])
        res[name] = table(R, M, t, nmax)
    return res


def run(case):
    t0 = time.time()
    nmax, kind = case["n"], case["kind"]
    # 2 nmax + 1 samples with A inside; crossings alternate A inside / B inside
    rtol = case.get("rtol", 1e-12)
    cap = dict(tmax=200.0 * nmax, anchor=1, rtol=rtol, max_step=case.get("max_step", 0.02), max_events=4 * nmax + 8)
    if kind in ("example", "tolerance"):
        masses, y, order = BS.initial(case["start"])
        r = N.simulate([BS.A, BS.B], masses, y, order=order, atol=rtol * 1e-2, **cap)
    elif kind == "control":
        x = case["x0"]
        va, vb = BS.velocities(*x)
        mid, na, nb, _, _ = N.crossing(x[0], va, vb, 0.0, x[1], BS.TOT, BS.A, BS.B)
        r = N.simulate([BS.A, BS.B], [0.0, mid, BS.TOT], [x[0], na, x[0], nb], order=[1, 0], atol=1e-14, **cap)
        # the start is itself a head-on crossing with A inside
        r["events"] = [dict(loser="A", v_before=[va, vb], r=x[0], masses_before=[0.0, x[1], BS.TOT], t=0.0)] + r[
            "events"]
    else:
        d = case["delta"]
        A, B = DS.shells(d)
        sa = np.sqrt(-N.potential(DS.RI, 0.0, d / 2, A))
        sb = np.sqrt(-N.potential(DS.RI, d / 2, d, B))
        if case["start"] == "D":
            masses, state, order = [0.0, d / 2, d], [DS.RI, -sa, DS.RI, sb], [0, 1]
        else:
            mid, va, vb, _, _ = N.crossing(DS.RI, sa, -sb, 0.0, d / 2, d, A, B)
            masses, state, order = [0.0, mid, d], [DS.RI, va, DS.RI, vb], [1, 0]
        r = N.simulate([A, B], masses, state, order=order, atol=1e-14 * d / 0.03, **cap)
    res = dict(case=case, status=r["status"], t_end=r["t_end"], **analyse(r["events"], nmax))
    res["seconds"] = time.time() - t0
    a2 = res["all_a_inside"]
    last = max(a2, key=int) if a2 else None
    print(f"{case['label']}: {r['status']}, {res['a_inside']} A-inside crossings {res['counts']}, "
          f"all-crossings digits at n={last}: "
          f"{ {k: round(v['n_vs_2n'], 1) for k, v in a2[last].items()} if last else None}, "
          f"{res['seconds']:.0f} s", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--deltas", default=",".join(f"{x:.4f}" for x in np.arange(0.001, 0.03405, 0.001)) + ",0.0360,0.0370")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    fp = json.load(open(a.section))["fixed_points"]
    saddle = next(f["x"] for f in fp if f["kind"] == "hyperbolic")
    centre = next(f["x"] for f in fp if f["kind"] == "elliptic")
    cases = [dict(kind="example", start=s, label=f"example {s}") for s in ("D", "C")]
    cases += [dict(kind="tolerance", start="D", rtol=rt, max_step=ms, label=f"tolerance rtol {rt:g} max_step {ms:g}")
              for rt, ms in TOLERANCES]
    cases += [dict(kind="control", x0=[saddle[0] + d, saddle[1]], label=f"control saddle R+ {d:g}")
              for d in (1e-4, 1e-2, 0.07)]
    cases += [dict(kind="control", x0=[centre[0] + 1e-3, centre[1]], label="control centre R+ 0.001")]
    cases += [dict(kind="family", delta=float(d), start=s, label=f"family delta {float(d):.4f} {s}")
              for d in a.deltas.split(",") for s in ("D", "C")]
    for c in cases:
        c["n"] = a.n
    with Pool(a.workers) as pool:
        runs = pool.map(run, cases, chunksize=1)
    json.dump(dict(params=dict(n=a.n, NS=NS, saddle=saddle, centre=centre), runs=runs,
                   seconds=time.time() - t0), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
