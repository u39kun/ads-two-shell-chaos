"""Why small companion distances inflate two-orbit exponents near the saddle (App. C).

For an orbit started 0.01 from the saddle along R+ at BCR's Fig. 5 parameters, follow blocks of 10 returns. In each
block compare the growth factor of the separation of two orbits (bcr_layer_convergence.advance, companion d0 away,
renormalized at the end of the block) with the growth factor of the variational tangent (tangent_map.py) over the same
block, for d0 = 1e-12, 1e-10, and 1e-8. The tangent's growth factor ranges from strong compression to strong
stretching as the orbit passes the saddle; where the separation has been compressed, rounding-level noise in the two
orbits is no longer small compared with it.
usage: python bcr_noise_trace.py OUT.json [--blocks 200]"""

import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tangent_map as TM
import bcr_layer_convergence as LC

HERE = os.path.dirname(os.path.abspath(__file__))
SC = np.array([1.0, 100.0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--blocks", type=int, default=200)
    p.add_argument("--dist", type=float, default=1e-2)
    a = p.parse_args()
    S = json.load(open(os.path.join(HERE, "results", "bcr_section", "bcr_section.json")))
    saddle = np.array([fp["x"] for fp in S["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    P = TM.TwoShellMap()
    x = saddle + np.array([a.dist, 0.0])
    e = np.array([1.0, 1.0]) / SC
    e /= np.linalg.norm(e * SC)
    d0s = (1e-12, 1e-10, 1e-8)
    ys = {d0: x + d0 * e for d0 in d0s}
    v = e.copy()
    rows = []
    for b in range(a.blocks):
        px, _ = LC.advance(x, 10, 1e-12, 0.02)
        z = x.copy()
        for k in range(10):
            z, D = P(z, tangent=True)
            v = D @ v
        gt = float(np.linalg.norm(v * SC))
        v /= gt
        row = dict(tangent=gt)
        for d0 in d0s:
            py, _ = LC.advance(ys[d0], 10, 1e-12, 0.02)
            d = np.linalg.norm((py[-1] - px[-1]) * SC)
            row[f"{d0:g}"] = float(d / d0)
            ys[d0] = px[-1] + (py[-1] - px[-1]) * (d0 / d)
        rows.append(row)
        x = px[-1]
    gt = np.array([r["tangent"] for r in rows])
    summary = dict(tangent_factor_range=[float(gt.min()), float(gt.max())])
    for d0 in d0s:
        g = np.array([r[f"{d0:g}"] for r in rows])
        dev = np.abs(np.log(g / gt))
        summary[f"{d0:g}"] = dict(
            max_abs_log_ratio=float(dev.max()),
            median_abs_log_ratio=float(np.median(dev)),
            exponent=float(np.log(g).sum() / (10 * len(g))),
        )
    summary["tangent_exponent"] = float(np.log(gt).sum() / (10 * len(gt)))
    print(json.dumps(summary, indent=1))
    json.dump(dict(dist=a.dist, blocks=a.blocks, summary=summary, rows=rows), open(a.out, "w"))


if __name__ == "__main__":
    main()
