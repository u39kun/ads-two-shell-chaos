"""Addendum: the t = 2e5 runs quoted in Sec. 4.3, App. C, and the Fig. 7 caption, at BCR's Fig. 5
parameters, with the event-located solver and the settings of ../bcr_section_overview.py (anchor 1, rtol 1e-12,
atol 1e-14, steps up to 0.02).

  end orbit    the strongly chaotic orbit of Fig. 7, started at (R, 100 M_mid) = (0.19, 1.575): every crossing,
               classified by radius (R < 0.4, 0.4-2.0, R > 2.0), by which shell is inside, and as head-on (shells
               moving in opposite directions) or overtaking. The paper quotes that its ~40000 crossings at
               0.4 < R < 2.0 are all overtaking, so none is on the head-on section.
  saddle starts  eight starts within 1e-3 of the saddle of ../results/bcr_section/bcr_section.json in the chart
               (R, 100 M_mid): R +- 1e-9, 1e-5, 1e-3 and 100 M_mid +- 1e-3. The paper quotes their head-on section
               crossings (A inside, A moving out, B moving in; about 44000 each) and the R range they span.
For every orbit: the number of crossings, the head-on section crossings by radius zone, and their R range.
usage: python addendum/long_orbits.py results/bcr_section/bcr_section.json OUT.json [--tmax 2e5] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
import bcr_section as BS
from common import N

NEAR_SMALL, NEAR_LARGE = 0.4, 2.0
ZONES = ("R<0.4", "0.4-2.0", "R>2.0")


def zone(r):
    return np.where(r < NEAR_SMALL, ZONES[0], np.where(r > NEAR_LARGE, ZONES[2], ZONES[1]))


def run(args):
    R0, M0, label, tmax = args
    t0 = time.time()
    va0 = np.sqrt(-N.potential(R0, 0.0, M0, BS.A))
    vb0 = -np.sqrt(-N.potential(R0, M0, BS.TOT, BS.B))
    mid, na, nb, _, _ = N.crossing(R0, va0, vb0, 0.0, M0, BS.TOT, BS.A, BS.B)
    s = N.simulate(
        [BS.A, BS.B],
        [0.0, mid, BS.TOT],
        [R0, na, R0, nb],
        order=[1, 0],
        tmax=tmax,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=10**7,
    )
    ev = s["events"]
    r = np.array([e["r"] for e in ev])
    inner = np.array([e["loser"] for e in ev])
    va = np.array([e["v_before"][0] for e in ev])
    vb = np.array([e["v_before"][1] for e in ev])
    z = zone(r)
    headon = va * vb < 0
    table = {
        f"{zz} | {sh} inside | {kind}": int(((z == zz) & (inner == sh) & (headon == ho)).sum())
        for zz in ZONES
        for sh in ("A", "B")
        for ho, kind in ((True, "head-on"), (False, "overtaking"))
    }
    sec = (inner == "A") & (va > 0) & (vb < 0)  # the head-on section of the paper (Sec. 2)
    rs = r[sec]
    return dict(
        label=label,
        start=[R0, M0],
        status=s["status"],
        crossings=int(len(ev)),
        crossing_table=table,
        section_crossings=int(sec.sum()),
        section_by_zone={zz: int((zone(rs) == zz).sum()) for zz in ZONES},
        section_R_range=[float(rs.min()), float(rs.max())] if len(rs) else None,
        seconds=time.time() - t0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=2e5)
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count())))
    a = p.parse_args()
    saddle = [f["x"] for f in json.load(open(a.section))["fixed_points"] if f["kind"] == "hyperbolic"][0]
    starts = [(0.19, 0.01575, "end orbit (Fig. 7)")]
    for d in (1e-9, 1e-5, 1e-3):
        starts += [(saddle[0] + d, saddle[1], f"saddle R+{d:g}"), (saddle[0] - d, saddle[1], f"saddle R-{d:g}")]
    starts += [(saddle[0], saddle[1] + 1e-5, "saddle 100M+1e-3"), (saddle[0], saddle[1] - 1e-5, "saddle 100M-1e-3")]
    t0 = time.time()
    with Pool(min(a.workers, len(starts))) as pool:
        runs = pool.map(run, [(R, M, lab, a.tmax) for R, M, lab in starts])
    for o in runs:
        print(
            f"{o['label']:20s} {o['status']:11s} crossings {o['crossings']:6d}  section {o['section_crossings']:6d} "
            f"{o['section_by_zone']}  R {o['section_R_range'][0]:.4f}-{o['section_R_range'][1]:.4f}  ({o['seconds']:.0f} s)"
        )
    end = runs[0]["crossing_table"]
    mid = {k: v for k, v in end.items() if k.startswith("0.4-2.0")}
    print(f"end orbit at 0.4 < R < 2.0: {mid} (total {sum(mid.values())})")
    sad = runs[1:]
    print(
        f"saddle starts: section crossings {min(o['section_crossings'] for o in sad)}-"
        f"{max(o['section_crossings'] for o in sad)}, R {min(o['section_R_range'][0] for o in sad):.4f}-"
        f"{max(o['section_R_range'][1] for o in sad):.4f}"
    )
    out = dict(params=dict(tmax=a.tmax, saddle=saddle, zones=[NEAR_SMALL, NEAR_LARGE]), runs=runs, seconds=time.time() - t0)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
