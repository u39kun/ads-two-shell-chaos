"""Overtaking crossings within sampled returns of the orbits in Figure 3(a).

Restart from archived head-on points, so the extra crossings follow the plotted
trajectories rather than a separate long integration of a chaotic orbit. Sample
one return out of four, matching the head-on sampling of the layer scan.
Usage: python bcr_overtaking.py OUT.json [--workers 8]
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import bcr_section as BS
import tangent_map as TM
from common import N

RESULTS = Path(__file__).resolve().parent / "results" / "bcr_section"


def one_return(x):
    va, vb = BS.velocities(*x)
    mid, na, nb, _, _ = N.crossing(x[0], va, vb, 0.0, x[1], BS.TOT, BS.A, BS.B)
    for limit in (2, 12, 40):
        run = N.simulate(
            [BS.A, BS.B],
            [0.0, mid, BS.TOT],
            [x[0], na, x[0], nb],
            order=[1, 0],
            tmax=80,
            anchor=1,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.02,
            max_events=limit,
        )
        omitted = []
        for event in run["events"]:
            if event["loser"] != "A":
                continue
            vi, vo = event["v_before"]
            point = [event["r"], event["masses_before"][1]]
            if vi > 0 > vo:
                return point, omitted
            if vi * vo <= 0:
                raise RuntimeError("Unexpected crossing branch")
            omitted.append(("both_in" if vi < 0 else "both_out", point))
    raise RuntimeError("No head-on return within 40 crossings")


def crossings(case):
    label, points = case
    branches = dict(both_in=[], both_out=[])
    worst = 0.0
    checks = {0, len(points) // 2, len(points) - 1}
    solver = TM.TwoShellMap(m=BS.M, w=BS.W, Min=0.0, Mt=BS.TOT, rtol=1e-12, atol=1e-14, max_step=0.02)
    for i, x in enumerate(points):
        returned, omitted = one_return(x)
        for branch, point in omitted:
            branches[branch].append(point)
        if i in checks:
            status, target, _, _, _ = TM._return(
                np.array(x),
                solver.m,
                solver.w,
                solver.Min,
                solver.Mt,
                solver.rtol,
                solver.atol,
                solver.max_step,
                True,
                solver.tmax,
            )
            error = float(np.max(np.abs((np.array(returned) - target) * [1.0, 100.0])))
            if status != 0 or error > 1e-7:
                raise RuntimeError(f"Return mismatch for {label}: {status}, {error}")
            worst = max(worst, error)
    print(
        f"{label}: {len(points)} returns; overtaking {sum(map(len, branches.values()))}; check {worst:.3g}", flush=True
    )
    return dict(label=label, sampled_returns=len(points), return_check_max=worst, **branches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    layer = json.loads((RESULTS / "layer_tangent_example.json").read_text())["runs"]
    cases = []
    for run in layer:
        case = run["case"]
        if case["direction"] == "example":
            continue
        logs = run["cumulative_log"]
        every = run["every"]
        late = (logs[6000 // every - 1] - logs[2000 // every - 1]) / 4000
        if late > 3e-3 or (case["direction"] in ("R+", "R-") and case["dist"] == 0.07):
            cases.append((case["label"], run["points"][:-1]))
    checks = json.loads((RESULTS / "bcr_checks.json").read_text())
    cases.append(("BCR-5", checks["long_points"][::4]))
    with Pool(args.workers) as pool:
        runs = pool.map(crossings, cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                sampling="one return from each archived fourth head-on point; A inside; pre-crossing middle mass",
                solver=dict(rtol=1e-12, atol=1e-14, max_step=0.02, anchor=1),
                runs=runs,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
