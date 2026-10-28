"""The type (ii) family followed to 5e4 returns with the variational tangent (Sec. 4.1 of the paper).

Same orbits, chart, and settings as bcr_layer_tangent.py family (first section point of each run of SCAN,
bcr_delta_scan.py start D; chart (R, M_mid/delta); rtol 1e-12, max step 0.02), but 5e4 returns, the length of the
BCR-5 test, and only the accumulated log growth is stored (every 10 returns), not the section points.
Stored per delta: the cumulative finite-time exponent lambda_N at N = 4000, 1e4, 2e4, 5e4 (those not beyond the run)
and at the run's last return; the late-block exponent over returns [N/5, N] of a run of N returns, with its interval
(None for runs shorter than LATE_MIN returns); the mean return time; and ln N/N for comparison. With the default
5e4 returns the late block is returns 1e4-5e4.
usage: python bcr_family_long.py SCAN.json OUT.json [--returns 50000] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
import bcr_delta_scan as DS

CHECK = (4000, 10000, 20000, 50000)
LATE_MIN = 1000  # shortest run for which a late block [N/5, N] is reported


def run(job):
    delta, x0, n = job
    t0 = time.time()
    P = TM.TwoShellMap(
        m=DS.MFAC * delta, w=0.2, Min=0.0, Mt=delta, rtol=1e-12, atol=1e-14 * delta / 0.03, max_step=0.02
    )
    sc = np.array([1.0, 1.0 / delta])
    x = np.array(x0, float)
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S = np.zeros(n + 1)  # accumulated log growth after k returns
    clock = 0.0
    for k in range(1, n + 1):
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            return dict(delta=delta, status=f"tangent status {st}", returns=k - 1)
        v = DF @ v
        g = np.linalg.norm(v * sc)
        S[k] = S[k - 1] + np.log(g)
        v /= g
        clock += tr
        x = x1
    lam = {N: float(S[N] / N) for N in sorted({N for N in CHECK if N <= n} | {n})}
    a = n // 5
    late = dict(returns=[a, n], exponent=float((S[n] - S[a]) / (n - a))) if n >= LATE_MIN else None
    return dict(
        delta=delta,
        status="ok",
        returns=n,
        return_time=clock / n,
        lambda_N=lam,
        late_block=late,
        lnN_over_N=float(np.log(n) / n),
        cumulative_log_every_10=S[10::10].tolist(),
        seconds=time.time() - t0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scan")
    p.add_argument("out")
    p.add_argument("--returns", type=int, default=50000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    runs = sorted(json.load(open(a.scan))["runs"], key=lambda r: r["delta"])
    with Pool(a.workers) as pool:
        res = pool.map(run, [(r["delta"], r["points"][0], a.returns) for r in runs], chunksize=1)
    for r in res:
        if r["status"] != "ok":
            print(f"delta {r['delta']:.3f}: {r['status']}")
            continue
        print(
            f"delta {r['delta']:.3f}: lambda_N "
            + " ".join(f"{N}:{v:.2e}" for N, v in r["lambda_N"].items())
            + (
                f"; late block {r['late_block']['returns']} {r['late_block']['exponent']:.1e}"
                if r["late_block"]
                else "; no late block"
            )
            + f"; return time {r['return_time']:.3f}"
        )
    json.dump(dict(returns=a.returns, runs=res, seconds=time.time() - t0), open(a.out, "w"))
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
