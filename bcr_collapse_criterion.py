"""Checks of the operational collapse criterion (Sec. 2 and App. B of the paper).

A run collapses when min f over both sides of every shell falls below horizon_cutoff (1e-6 in every scan); the
stopping time is the collapse time. Because Schwarzschild time approaches a horizon only asymptotically, that time
depends on the cutoff. Two checks:
  cutoff     twelve collapsing starts of the survey at delta = 0.037 (SURVEY, bcr_survey.py section), spread over
             its collapse times, rerun as in bcr_survey.py with cutoffs 1e-4, 1e-6, 1e-8, 1e-10, in the middle-region
             clock (anchor 1, the survey's) and the exterior (boundary) clock (anchor 2): collapse time, crossings,
             the trapping region and its mass, and, within each clock, whether the crossing sequence (kinds, times,
             radii) changes with the cutoff;
  threshold  the no-collapse threshold of bcr_validation.py (runs to 1000 time units, same initial data and
             settings), bisected 30 times from its grid bracket [0.0370, 0.0372] in the boundary clock (anchor 2,
             as bcr_validation.py) and in the middle-region clock (anchor 1).
usage: python bcr_collapse_criterion.py SURVEY.json OUT.json [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_delta_scan as DS
import bcr_validation as V
from common import N

DELTA, NPICK = 0.037, 12
CUTS = (1e-4, 1e-6, 1e-8, 1e-10)
BRACKET, NBISECT = (0.0370, 0.0372), 30


def rerun(args):
    R, M, anchor, cut = args
    A, B = DS.shells(DELTA)
    va, vb = np.sqrt(-N.potential(R, 0.0, M, A)), -np.sqrt(-N.potential(R, M, DELTA, B))
    mid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, M, DELTA, A, B)
    s = N.simulate(
        [A, B],
        [0.0, mid, DELTA],
        [R, na, R, nb],
        order=[1, 0],
        tmax=20000.0,
        anchor=anchor,
        rtol=1e-12,
        atol=1e-14 * DELTA / 0.03,
        max_step=0.02,
        max_events=10**7,
        horizon_cutoff=cut,
    )
    st, ms = s["state_end"], s["masses_end"]
    order = [0 if n == "A" else 1 for n in s["order_end"]]
    fmin, region = min((float(N.f(st[2 * i], ms[k + j])), k + j) for k, i in enumerate(order) for j in (0, 1))
    return dict(
        R=R,
        M=M,
        anchor=anchor,
        cut=cut,
        status=s["status"],
        t_end=float(s["t_end"]),
        crossings=len(s["events"]),
        trapping_region=region,
        mass_inside=float(ms[region]),
        # the crossing sequence: time, radius, and kind (which shell was inside; head-on or overtaking)
        sequence=[(e["t"], e["r"], e["loser"], bool(e["v_before"][0] * e["v_before"][1] < 0)) for e in s["events"]],
    )


def validation_run(args):
    """bcr_validation.run with a choice of clock: collapse (True/False), end time, crossings."""
    delta, anchor = args
    A, B = N.Shell("A", V.MFAC * delta, V.W), N.Shell("B", V.MFAC * delta, V.W)
    vin = -np.sqrt(-N.potential(V.RI, 0.0, delta / 2, A))
    vout = np.sqrt(-N.potential(V.RI, delta / 2, delta, B))
    r = N.simulate(
        [A, B],
        [0.0, delta / 2, delta],
        [V.RI, vin, V.RI, vout],
        order=[0, 1],
        tmax=V.TMAX,
        anchor=anchor,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=100000,
    )
    return r["status"] == "horizon_approach", float(r["t_end"]), len(r["events"])


def bisect(anchor):
    lo, hi = BRACKET
    assert not validation_run((lo, anchor))[0] and validation_run((hi, anchor))[0]
    for _ in range(NBISECT):
        mid = 0.5 * (lo + hi)
        if validation_run((mid, anchor))[0]:
            hi = mid
        else:
            lo = mid
    c, t, n = validation_run((hi, anchor))
    return dict(anchor=anchor, threshold=[lo, hi], collapse_time_at_hi=t, crossings_at_hi=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("survey")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    g = [g for g in json.load(open(a.survey))["grids"] if g["delta"] == DELTA][0]
    col = sorted((r for r in g["runs"] if r.get("event_status") == "horizon_approach"), key=lambda r: r["event_t_end"])
    pick = [col[int(i)] for i in np.linspace(0, len(col) - 1, NPICK)]
    jobs = [(r["x0"][0], r["x0"][1], an, c) for r in pick for an in (1, 2) for c in CUTS]
    with Pool(a.workers) as pool:
        res = pool.map(rerun, jobs, chunksize=1)
        thr = pool.map(bisect, (2, 1))
    starts = []
    for k, r0 in enumerate(pick):
        rr = res[k * 2 * len(CUTS) : (k + 1) * 2 * len(CUTS)]
        # compare the crossing sequences across cutoffs within each clock: same kinds, and the largest differences
        # of crossing time and radius from the run at the survey's cutoff (1e-6)
        same = {}
        for an in (1, 2):
            seqs = [x["sequence"] for x in rr if x["anchor"] == an]
            ref = seqs[CUTS.index(1e-6)]
            same[an] = dict(
                kinds_identical=all([q[2:] for q in sq] == [q[2:] for q in ref] for sq in seqs),
                max_dt=max((abs(a[0] - b[0]) for sq in seqs for a, b in zip(sq, ref)), default=0.0),
                max_dr=max((abs(a[1] - b[1]) for sq in seqs for a, b in zip(sq, ref)), default=0.0),
            )
        for x in rr:
            del x["sequence"]
        t = {an: [x["t_end"] for x in rr if x["anchor"] == an] for an in (1, 2)}
        starts.append(
            dict(
                x0=r0["x0"],
                survey_t_end=r0["event_t_end"],
                runs=rr,
                shift_per_factor_100={an: np.diff(t[an]).tolist() for an in (1, 2)},
                crossings=sorted({x["crossings"] for x in rr}),
                statuses=sorted({x["status"] for x in rr}),
                boundary_over_middle=t[2][1] / t[1][1],
                sequence_vs_cutoff=same,
            )
        )
    sh = [s for st in starts for an in (1, 2) for s in st["shift_per_factor_100"][an]]
    for st in starts:
        print(
            f"survey t {st['survey_t_end']:9.2f}: shifts {[round(x, 3) for x in st['shift_per_factor_100'][1]]}, "
            f"crossings {st['crossings']}, {st['statuses']}, boundary/middle {st['boundary_over_middle']:.4f}; "
            f"sequences: {[(an, v['kinds_identical'], v['max_dt'], v['max_dr']) for an, v in st['sequence_vs_cutoff'].items()]}"
        )
    print(
        f"shift per factor 100 in the cutoff: {min(sh):.3f}-{max(sh):.3f}; "
        f"one crossing count per start: {all(len(st['crossings']) == 1 for st in starts)}"
    )
    for t in thr:
        print(
            f"threshold, anchor {t['anchor']}: {t['threshold'][0]:.12f} .. {t['threshold'][1]:.12f} "
            f"(collapse at t = {t['collapse_time_at_hi']:.1f} after {t['crossings_at_hi']} crossings)"
        )
    json.dump(
        dict(delta=DELTA, cutoffs=CUTS, starts=starts, threshold=thr, seconds=time.time() - t0),
        open(a.out, "w"),
        default=float,
    )
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
