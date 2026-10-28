"""Checks of the strongly chaotic starts near the section's ends (Table 1 and Fig. 8f of the paper, App. C).

Rows: the strongest start of the finer grids at the section's ends (bcr_survey.py ends) at each delta where they first
detect strong chaos, and the strongly chaotic survey cells at delta = 0.03, with two controls at delta = 0.03 that stay
below the threshold (one central, one near the small-R end with many overtaking crossings). For each row:
  tangent       rates per unit time from the variational tangent over four successive 5000-unit blocks of T = 20000,
                at the default setting (rtol 1e-12, max step 0.02) and a tighter one (rtol 1e-13, max step 0.005);
  crossings     event-located run to t = 20000 (n_shells.simulate): the share of the crossings with A inside that are
                overtaking, and where the head-on crossings fall relative to the section's R extent at that delta
                (lowest fifth, middle, highest fifth), with the number of switches between the two ends.
usage: python bcr_survey_checks.py OUT.json [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
import bcr_delta_scan as DS
from common import N

# (delta, R, M_mid/delta, label); the low-delta rows are the strongest starts of the finer grids at the ends
ROWS = [
    (0.010, 0.219212, 0.505, "small R"),
    (0.014, 0.215763, 0.505, "small R"),
    (0.017, 0.215763, 0.505, "small R"),
    (0.020, 0.206064, 0.515, "small R"),
    (0.020, 2.363940, 0.505, "large R"),
    (0.03, 0.182, 0.525, "small R"),
    (0.03, 2.308, 0.508, "large R"),
    (0.03, 0.944, 0.492, "control (central)"),
    (0.03, 0.246, 0.543, "control (near small-R end)"),
]
T, NB = 20000.0, 4
SETTINGS = {"default": (1e-12, 0.02), "tight": (1e-13, 0.005)}


def start(delta, R, md):
    A, B = DS.shells(delta)
    M = md * delta
    va, vb = np.sqrt(-N.potential(R, 0.0, M, A)), -np.sqrt(-N.potential(R, M, delta, B))
    mid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, M, delta, A, B)
    return A, B, mid, na, nb


def tangent(args):
    delta, R, md, setting = args
    rtol, ms = SETTINGS[setting]
    sc = np.array([1.0, 3.0 / delta])
    P = TM.TwoShellMap(
        m=DS.MFAC * delta,
        w=0.2,
        Min=0.0,
        Mt=delta,
        rtol=rtol,
        atol=rtol * 1e-2 * delta / 0.03,
        max_step=ms,
        tmax=5000.0,
    )
    x = np.array([R, md * delta])
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S, clock, k, status, cum, ncs = 0.0, 0.0, 0, "ok", [(0.0, 0.0)], []
    while clock < T:
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            status = {1: "forbidden start", 2: "integrator failure", 3: "return-time limit"}[st]
            break
        v = DF @ v
        g = np.linalg.norm(v * sc)
        S += np.log(g)
        v /= g
        clock += tr
        k += 1
        ncs.append(nc)
        cum.append((clock, S))
        x = x1
    c = np.array(cum)
    blocks = []
    for a, b in zip(np.linspace(0, T, NB + 1)[:-1], np.linspace(0, T, NB + 1)[1:]):
        i, j = np.searchsorted(c[:, 0], a), min(np.searchsorted(c[:, 0], b), len(c) - 1)
        blocks.append(float((c[j, 1] - c[i, 1]) / (c[j, 0] - c[i, 0])) if c[j, 0] > c[i, 0] + 100 else None)
    return dict(
        setting=setting,
        rtol=rtol,
        max_step=ms,
        status=status,
        time=float(clock),
        returns=k,
        crossings_per_return=float(np.mean(ncs)) if ncs else None,
        blocks=blocks,
    )


def extent(delta):
    """R range of the allowed head-on section (R > 0.1) at this delta."""
    A, B = DS.shells(delta)
    Rs = np.linspace(0.1, 3.0, 2901)
    Ms = np.linspace(0.3 * delta, 0.7 * delta, 801)
    ok = np.array([[N.potential(R, 0.0, M, A) < 0 and N.potential(R, M, delta, B) < 0 for R in Rs] for M in Ms])
    cols = ok.any(axis=0)
    return float(Rs[cols].min()), float(Rs[cols].max())


def crossings(args):
    delta, R, md = args
    A, B, mid, na, nb = start(delta, R, md)
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
    ain = [e for e in s["events"] if e["loser"] == "A"]
    head = [e for e in ain if e["v_before"][0] > 0 > e["v_before"][1]]
    lo, hi = extent(delta)
    r = np.array([e["r"] for e in head])
    zone = np.where(r < lo + 0.2 * (hi - lo), 0, np.where(r > hi - 0.2 * (hi - lo), 2, 1))
    ends = zone[zone != 1]
    return dict(
        status=s["status"],
        a_inside=len(ain),
        overtaking_share=(len(ain) - len(head)) / max(len(ain), 1),
        section_R=[lo, hi],
        headon=len(head),
        headon_R=[float(r.min()), float(r.max())] if len(r) else None,
        near_small_end=int((zone == 0).sum()),
        middle=int((zone == 1).sum()),
        near_large_end=int((zone == 2).sum()),
        end_switches=int((np.diff(ends) != 0).sum()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    tj = [(d, R, md, s) for d, R, md, _ in ROWS for s in SETTINGS]
    cj = [(d, R, md) for d, R, md, _ in ROWS]
    with Pool(a.workers) as pool:
        cr = pool.map_async(crossings, cj, chunksize=1)
        tg = pool.map(tangent, tj, chunksize=1)
        cr = cr.get()
    rows = []
    for i, (d, R, md, lab) in enumerate(ROWS):
        row = dict(
            delta=d,
            R=R,
            md=md,
            label=lab,
            control=lab.startswith("control"),
            tangent={r["setting"]: r for r in tg[2 * i : 2 * i + 2]},
            crossings=cr[i],
        )
        rows.append(row)
        f = lambda bl: [None if x is None else round(x, 4) for x in bl]
        print(
            f"delta {d:g} {lab:27s} ({R}, {md}): overtaking {100 * cr[i]['overtaking_share']:.1f}%, "
            f"head-on near ends {cr[i]['near_small_end']}/{cr[i]['middle']}/{cr[i]['near_large_end']}; "
            f"default {f(row['tangent']['default']['blocks'])}; tight {f(row['tangent']['tight']['blocks'])}",
            flush=True,
        )
    json.dump(dict(rows=rows, T=T, seconds=time.time() - t0), open(a.out, "w"), indent=1)
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
