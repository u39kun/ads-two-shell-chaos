"""Survey of the whole head-on section across BCR's type (ii) family (Sec. 4.3 and Fig. 8 of the paper).

Starts are followed with the variational tangent (tangent_map.py) for a fixed physical time T in the clock of the
middle region. Near the section's ends a return can contain from two to about 200 crossings, so exponents per return
would not compare growth rates between orbits; the rate reported is the accumulated log growth of a tangent vector
(started along (1, 1) in the chart (R, 3 M_mid/delta)) per unit time over [T/4, T], with the same rate over four
equal blocks of T. A start the tangent cannot follow to its next return within 5000 time units is classified with the
event-located solver (n_shells.simulate) up to T: the tangent map does not detect horizon formation, and in the
middle-region clock an infalling shell never returns.
  section  a grid of nr x nm cells uniform in (R, M_mid/delta) over the bounding box of the allowed head-on crossings
           (box below), at each delta; the allowed cell centres are followed for T = 20000;
  ends     finer grids at the section's two ends, from the same bounding box: small R [R_lo, R_lo + 0.15] x
           M_mid/delta 0.44-0.62 (24 x 18), large R [R_hi - 0.25, R_hi] x 0.44-0.58 (24 x 14), followed for
           T = 2000.
Stored for each start: its position, status, number of returns, mean crossings per return, rate, block rates, and
for starts classified by the event solver its outcome and end time.
usage: python bcr_survey.py section|ends OUT.json [--deltas d1,d2,...] [--workers N]"""

import os, sys, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
import bcr_delta_scan as DS
from common import N

DELTAS = {
    "section": "0.006,0.012,0.02,0.03,0.037",
    "ends": "0.006,0.01,0.012,0.014,0.017,0.02,0.022,0.024,0.026,0.028,0.03,0.037",
}
T_RUN = {"section": 20000.0, "ends": 2000.0}
STATUS = {1: "forbidden start", 2: "integrator failure", 3: "return-time limit"}


def allowed(delta, R, M):
    """A head-on crossing at radius R with middle mass M exists: both shells are allowed at R, and f(R; M) > 0."""
    A, B = DS.shells(delta)
    return N.potential(R, 0.0, M, A) < 0 and N.potential(R, M, delta, B) < 0 and N.f(R, M) > 0


def box(delta):
    """Bounding box (R_lo, R_hi, m_lo, m_hi) of the allowed head-on crossings, m = M_mid/delta, with a 0.01 margin."""
    Rs, ms = np.linspace(0.1, 3.0, 300), np.linspace(0, 1, 100)[1:-1]
    R, m = np.array([(R, m) for R in Rs for m in ms if allowed(delta, R, m * delta)]).T
    return R.min() - 0.01, R.max() + 0.01, max(m.min() - 0.01, 0.0), min(m.max() + 0.01, 1.0)


def grid(delta, r0, r1, m0, m1, nr, nm):
    """Allowed cell centres of an nr x nm grid over [r0, r1] x [m0, m1] (m = M_mid/delta)."""
    Rc = r0 + (np.arange(nr) + 0.5) * (r1 - r0) / nr
    mc = m0 + (np.arange(nm) + 0.5) * (m1 - m0) / nm
    return [(float(R), float(m)) for R in Rc for m in mc if allowed(delta, R, m * delta)]


def run(case):
    delta, R, md, T = case["delta"], case["R"], case["md"], case["T"]
    t0 = time.time()
    sc = np.array([1.0, 3.0 / delta])
    P = TM.TwoShellMap(
        m=DS.MFAC * delta, w=0.2, Min=0.0, Mt=delta, rtol=1e-12, atol=1e-14 * delta / 0.03, max_step=0.02, tmax=5000.0
    )
    x = np.array([R, md * delta])
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S, clock, k, ncs = 0.0, 0.0, 0, []
    cum = [[0.0, 0.0]]  # (elapsed time, accumulated log growth) every 10 returns
    status = "ok"
    while clock < T:
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            status = STATUS.get(st, str(st))
            break
        v = DF @ v
        g = np.linalg.norm(v * sc)
        S += float(np.log(g))
        v /= g
        clock += float(tr)
        ncs.append(int(nc))
        k += 1
        if k % 10 == 0:
            cum.append([clock, S])
        x = x1
    cum.append([clock, S])
    out = dict(
        case=case,
        x0=[R, md * delta],
        status=status,
        returns=k,
        time=clock,
        crossings_per_return_mean=float(np.mean(ncs)) if ncs else None,
    )
    if status == "ok":
        c = np.array(cum)
        i = int(np.searchsorted(c[:, 0], T / 4))
        out["rate_per_time"] = float((c[-1, 1] - c[i, 1]) / (c[-1, 0] - c[i, 0]))
        edges = np.searchsorted(c[:, 0], np.linspace(0, T, 5))
        edges[-1] = len(c) - 1
        out["block_rates"] = [float((c[j, 1] - c[i, 1]) / (c[j, 0] - c[i, 0])) for i, j in zip(edges[:-1], edges[1:])]
    else:
        # classify with the event-located solver: collapse (horizon_approach) or not, up to T
        A, B = DS.shells(delta)
        M = md * delta
        va, vb = np.sqrt(-N.potential(R, 0.0, M, A)), -np.sqrt(-N.potential(R, M, delta, B))
        mid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, M, delta, A, B)
        s = N.simulate(
            [A, B],
            [0.0, mid, delta],
            [R, na, R, nb],
            order=[1, 0],
            tmax=T,
            anchor=1,
            rtol=1e-12,
            atol=1e-14 * delta / 0.03,
            max_step=0.02,
            max_events=10**7,
        )
        out.update(event_status=s["status"], event_t_end=float(s["t_end"]), event_crossings=len(s["events"]))
    out["seconds"] = time.time() - t0
    return out


def key(case):
    return hashlib.sha1(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]


def cached(case, outdir):
    path = os.path.join(outdir, f"{key(case)}.json")
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get("case") == case:
            return r
    r = run(case)
    json.dump(r, open(path, "w"))
    return r


def summary(delta, runs):
    rate = [r.get("rate_per_time") for r in runs]
    coll = sum(1 for r in runs if r.get("event_status") == "horizon_approach")
    other = sum(1 for r in runs if r["status"] != "ok" and r.get("event_status") != "horizon_approach")
    return (
        f"delta {delta:g}: {len(runs)} starts; rate > 1e-3: {sum(1 for x in rate if x is not None and x > 1e-3)}, "
        f"> 1e-2: {sum(1 for x in rate if x is not None and x > 1e-2)}; collapse: {coll}; other stops: {other}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=("section", "ends"))
    p.add_argument("out")
    p.add_argument("--deltas", default=None)
    p.add_argument("--nr", type=int, default=48, help="section: grid columns in R")
    p.add_argument("--nm", type=int, default=24, help="section: grid rows in M_mid/delta")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    T = T_RUN[a.which]
    deltas = [float(d) for d in (a.deltas or DELTAS[a.which]).split(",")]
    grids = []  # (delta, label, window, nr, nm, starts)
    for dl in deltas:
        r0, r1, m0, m1 = box(dl)
        if a.which == "section":
            wins = [("whole", (r0, r1, m0, m1), a.nr, a.nm)]
        else:
            wins = [("small R", (r0, r0 + 0.15, 0.44, 0.62), 24, 18), ("large R", (r1 - 0.25, r1, 0.44, 0.58), 24, 14)]
        for lab, w, nr, nm in wins:
            grids.append(dict(delta=dl, region=lab, box=list(w), bbox=[r0, r1, m0, m1], nr=nr, nm=nm, T=T))
    cases = []
    for gi, g in enumerate(grids):
        g["starts"] = grid(g["delta"], *g["box"], g["nr"], g["nm"])
        cases += [(gi, dict(delta=g["delta"], R=R, md=m, T=T)) for R, m in g["starts"]]
    outdir = os.path.splitext(a.out)[0] + "_parts"
    os.makedirs(outdir, exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.starmap(cached, [(c, outdir) for _, c in cases], chunksize=1)
    for g in grids:
        g["runs"] = []
        del g["starts"]
    for (gi, _), r in zip(cases, res):
        grids[gi]["runs"].append(r)
    for g in grids:
        print(f"{g['region']}: " + summary(g["delta"], g["runs"]))
    json.dump(dict(which=a.which, T=T, grids=grids, seconds=time.time() - t0), open(a.out, "w"))
    print(f"wrote {a.out}; {len(cases)} starts, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
