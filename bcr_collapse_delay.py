"""Delayed collapse near the edge of a collapse band at delta = 0.037 (Sec. 4.3 of the paper).

At delta = 0.037 part of the head-on section collapses (bcr_survey.py section). Along a short line across the edge of
the outer band that faces the saddle, at M_mid/delta = 0.5, starts spaced 1.25e-6 apart in R are followed with the
event-located solver (n_shells.simulate) to t = 20000: whether and when each collapses (status horizon_approach).
usage: python bcr_collapse_delay.py OUT.json [--r0 1.0975] [--r1 1.1] [--n 2000] [--tmax 20000] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_delta_scan as DS
from common import N

DELTA, MD = 0.037, 0.5


def run(args):
    R, tmax = args
    M = MD * DELTA
    A, B = DS.shells(DELTA)
    pa, pb = N.potential(R, 0.0, M, A), N.potential(R, M, DELTA, B)
    if pa >= 0 or pb >= 0:
        return dict(R=R, status="forbidden")
    mid, na, nb, _, _ = N.crossing(R, np.sqrt(-pa), -np.sqrt(-pb), 0.0, M, DELTA, A, B)
    s = N.simulate(
        [A, B],
        [0.0, mid, DELTA],
        [R, na, R, nb],
        order=[1, 0],
        tmax=tmax,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * DELTA / 0.03,
        max_step=0.02,
        max_events=10**7,
    )
    return dict(R=R, status=s["status"], t_end=float(s["t_end"]), crossings=len(s["events"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--r0", type=float, default=1.0975)
    p.add_argument("--r1", type=float, default=1.1)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--tmax", type=float, default=20000.0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    Rs = np.linspace(a.r0, a.r1, a.n)
    with Pool(a.workers) as pool:
        res = pool.map(run, [(float(R), a.tmax) for R in Rs], chunksize=4)
    col = np.array([r["status"] == "horizon_approach" for r in res])
    t = np.array([r.get("t_end", np.nan) for r in res])
    alive = {f"{x:g}": int(np.sum(~col | (t > x))) for x in (3000, 5000, 10000, 15000, a.tmax)}
    # interleaving: changes of outcome between neighbours, with collapse counted by t = 3000 and by tmax
    early = col & (t <= 3000)
    summary = dict(
        starts=len(res),
        spacing=float(Rs[1] - Rs[0]),
        collapsed=int(col.sum()),
        alive_at=alive,
        switches_by_3000=int(np.sum(np.diff(early.astype(int)) != 0)),
        switches_by_tmax=int(np.sum(np.diff(col.astype(int)) != 0)),
        collapse_time_quantiles=np.nanquantile(t[col], [0, 0.25, 0.5, 0.75, 1]).tolist() if col.any() else None,
    )
    json.dump(
        dict(delta=DELTA, md=MD, r0=a.r0, r1=a.r1, tmax=a.tmax, summary=summary, runs=res, seconds=time.time() - t0),
        open(a.out, "w"),
    )
    print(summary)
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
