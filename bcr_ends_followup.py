"""Follow-up of the finer grids at the section's ends where they detect nothing (Sec. 4.3 and App. C of the paper).

The finer grids (bcr_survey.py ends, T = 2000) count a start as strongly chaotic when its rate over [T/4, T] exceeds
1e-2 per unit time. Where no start does, the NTOP starts with the largest rates are rerun exactly as in
bcr_survey.py but to T = 2e4, giving the rate over [T/4, T] and over four 5000-unit blocks, with the mean number of
crossings per return and the distance from the saddle (LAYER_DELTA, bcr_layer_delta.py) in the chart
(R, 3 M_mid/delta).
usage: python bcr_ends_followup.py SURVEY_ENDS.json LAYER_DELTA.json OUT.json [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import bcr_survey as SV

TIP_THR, NTOP, T = 1e-2, 5, 20000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ends")
    p.add_argument("layer_delta")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    saddle = {s["delta"]: s["saddle"] for s in json.load(open(a.layer_delta))["summary"]}
    cases, meta = [], []
    for g in json.load(open(a.ends))["grids"]:
        rates = [r.get("rate_per_time", -np.inf) for r in g["runs"]]
        if max(rates) > TIP_THR:
            continue
        for i in np.argsort(rates)[::-1][:NTOP]:
            r = g["runs"][i]
            cases.append(dict(delta=g["delta"], R=r["case"]["R"], md=r["case"]["md"], T=T))
            meta.append(dict(region=g["region"], delta=g["delta"], short_rate=rates[i], short_blocks=r["block_rates"]))
    with Pool(a.workers) as pool:
        res = pool.map(SV.run, cases, chunksize=1)
    out = []
    for m, c, r in zip(meta, cases, res):
        s = saddle.get(m["delta"])
        dist = float(np.hypot(c["R"] - s[0], 3 * c["md"] - 3 * s[1] / m["delta"])) if s else None
        m.update(
            R=c["R"],
            md=c["md"],
            status=r["status"],
            rate=r.get("rate_per_time"),
            blocks=r.get("block_rates"),
            crossings_per_return=r["crossings_per_return_mean"],
            distance_from_saddle=dist,
        )
        out.append(m)
        print(
            f"{m['region']:8s} delta {m['delta']:.3f} ({c['R']:.4f}, {3 * c['md']:.3f}): rate {m['short_rate']:.4f} to"
            f" T = 2000, {m['rate']:.5f} to T = 2e4, blocks {np.round(m['blocks'], 5).tolist()}, crossings/return"
            f" {m['crossings_per_return']:.1f}, from saddle {dist if dist is None else round(dist, 3)}"
        )
    json.dump(dict(threshold=TIP_THR, ntop=NTOP, T=T, runs=out, seconds=time.time() - t0), open(a.out, "w"))
    print(f"wrote {a.out}; {len(out)} starts, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
