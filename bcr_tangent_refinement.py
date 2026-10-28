"""Refinement check of the variational tangent (App. C of the paper).

At BCR's Fig. 5 parameters, three orbits are followed for 6000 returns with the variational tangent
(bcr_layer_tangent.run) at the settings of the scans (rtol = 1e-12, max_step = 0.02) and with a tighter tolerance
and a smaller maximum step: an orbit in the chaotic layer (R+ at 1e-4 from the saddle), a regular orbit near the
saddle (R+ at 1e-2), and BCR's example. The chart is (R, 100 M_mid) and the tangent vector starts along (1, 1), as
in the scans. For the chaotic orbit the computed trajectory changes with the settings, so the check is that its
late-block exponent stays clearly positive and its classification is unchanged, not that the values agree.
usage: python bcr_tangent_refinement.py SECTION.json OUT.json [--returns 6000] [--workers N]"""

import os, sys, json, argparse
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bcr_layer_tangent as LT

SETTINGS = [(1e-12, 0.02), (1e-13, 0.02), (1e-13, 0.01), (1e-11, 0.02)]
BLOCKS = [(2000, 6000), (3000, 6000), (2000, 4000)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=6000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    S = json.load(open(a.section))
    saddle = np.array([fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    sc = np.array([1.0, 100.0])
    starts = [
        ("R+ 1e-4", (saddle + np.array([1e-4, 0.0])).tolist()),
        ("R+ 1e-2", (saddle + np.array([1e-2, 0.0])).tolist()),
        ("BCR's example", S["bcr_orbit"][0]["points"][0]),
    ]
    cases = [
        dict(
            label=f"{lab}, rtol {rt:g}, max step {ms:g}",
            orbit=lab,
            rtol=rt,
            max_step=ms,
            m=0.00815,
            Mt=0.03,
            x0=x0,
            scale=sc.tolist(),
            returns=a.returns,
        )
        for lab, x0 in starts
        for rt, ms in SETTINGS
    ]
    with Pool(a.workers) as pool:
        res = pool.map(LT.run, cases, chunksize=1)
    for r in res:
        r["late"] = {f"{n1}-{n2}": LT.late(r, n1, n2) for n1, n2 in BLOCKS}
        r.pop("points")
    json.dump(dict(settings=SETTINGS, blocks=BLOCKS, runs=res), open(a.out, "w"))
    print("late-block exponents over returns 2000-6000:")
    for r in res:
        print(f"  {r['case']['label']:40s} {r['late']['2000-6000']: .6g}  ({r['status']})")


if __name__ == "__main__":
    main()
