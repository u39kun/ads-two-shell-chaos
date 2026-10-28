"""Connected components of the admissible head-on section (Sec. 4.3 of the paper).

A head-on crossing with A inside at radius R and middle mass M is admissible when both shells are allowed there
(V_A(R; 0, M) < 0, V_B(R; M, delta) < 0), every region is untrapped at R (f(R; M) > 0, f(R; delta) > 0), and both
junctions are on the ordinary branch (beta_out = (hi - lo)/mu - mu/(2R) > 0 for each shell).
  components  on a grid geometric in R from delta to 3 (3000 points) by 1500 in M_mid, at BCR's Fig. 5 parameters
              and at the survey's five values of delta of the type (ii) family: the bounding box of each component;
  onset       values of delta at which the near-horizon component first appears, on a finer grid next to the
              exterior horizon (R from R_h to 1.6 R_h);
  survey      on the survey's grid (bcr_survey.py section), the starts allowed by bcr_survey.allowed() (which omits
              the f(R; delta) and branch tests) and whether any of them fails the full test;
  fate        a grid over the near-horizon component at delta = 0.02, 0.03 (Fig. 5 parameters), 0.037, followed with
              the event-located solver at the survey's settings: outcome, time, crossings after the start.
usage: python bcr_section_domain.py OUT.json [--workers N]"""

import os, sys, json, time, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from scipy import ndimage
from scipy.optimize import brentq
import bcr_delta_scan as DS
import bcr_survey as SV
from common import N

W = 0.2
M5 = 0.00815  # BCR's Fig. 5 shells (as printed), as in bcr_section.py
CASES = [("BCR-5", 0.03, M5)] + [("type (ii)", d, DS.MFAC * d) for d in (0.006, 0.012, 0.02, 0.03, 0.037)]
ONSET = np.round(np.arange(0.0170, 0.0206, 0.0005), 4)
FATE = [(0.02, DS.MFAC * 0.02), (0.03, M5), (0.037, DS.MFAC * 0.037)]


def admissible(delta, m, R, M):
    """Masks: allowed by bcr_survey.allowed(), and by the full test; and the squared velocities."""
    S = N.Shell("A", m, W)
    mu = S.mu(R)
    va2, vb2 = -N.potential(R, 0.0, M, S), -N.potential(R, M, delta, S)
    survey = (va2 > 0) & (vb2 > 0) & (N.f(R, M) > 0)
    full = survey & (N.f(R, delta) > 0) & (M / mu - mu / (2 * R) > 0) & ((delta - M) / mu - mu / (2 * R) > 0)
    return survey, full, va2, vb2


def horizon(delta):
    return brentq(lambda r: N.f(r, delta), 0.5 * delta, 3 * delta)


def components(delta, m, Rs, nm):
    Ms = np.linspace(0, delta, nm + 2)[1:-1]
    R, M = np.meshgrid(Rs, Ms, indexing="ij")
    survey, full, _, _ = admissible(delta, m, R, M)
    lab, n = ndimage.label(full)
    out = []
    for k in range(1, n + 1):
        sel = lab == k
        out.append(dict(cells=int(sel.sum()), R=[R[sel].min(), R[sel].max()], M=[M[sel].min(), M[sel].max()]))
    out.sort(key=lambda c: c["R"][0])
    extra = survey & ~full
    return out, dict(cells=int(extra.sum()), R_max=float(R[extra].max()) if extra.any() else None)


def case_components(case):
    label, delta, m = case
    Rs = np.geomspace(delta, 3.0, 3000)
    comps, extra = components(delta, m, Rs, 1500)
    rh = horizon(delta)
    # the near-horizon component starts at the first grid radius above the exterior horizon
    at_horizon = bool(comps and comps[0]["R"][0] == Rs[np.argmax(Rs > rh)])
    return dict(
        label=label,
        delta=delta,
        m=m,
        R_horizon=rh,
        components=comps,
        lowest_component_at_horizon=at_horizon,
        survey_only=extra,
    )


def onset(delta):
    rh = horizon(delta)
    comps, _ = components(delta, DS.MFAC * delta, rh * (1 + np.geomspace(1e-9, 0.6, 3000)), 4000)
    near = [c for c in comps if c["R"][1] < 0.1]
    return dict(delta=float(delta), R_horizon=rh, near_horizon=near[0] if near else None)


def survey_grid(delta):
    r0, r1, m0, m1 = SV.box(delta)
    Rc = r0 + (np.arange(48) + 0.5) * (r1 - r0) / 48
    mc = m0 + (np.arange(24) + 0.5) * (m1 - m0) / 24
    R, md = np.meshgrid(Rc, mc, indexing="ij")
    survey, full, _, _ = admissible(delta, DS.MFAC * delta, R, md * delta)
    starts = SV.grid(delta, r0, r1, m0, m1, 48, 24)
    return dict(
        delta=delta,
        box=[r0, r1, m0, m1],
        starts=len(starts),
        allowed=int(survey.sum()),
        allowed_not_full=int((survey & ~full).sum()),
    )


def follow(args):
    delta, m, R, M = args
    S = N.Shell("A", m, W), N.Shell("B", m, W)
    va, vb = np.sqrt(-N.potential(R, 0.0, M, S[0])), -np.sqrt(-N.potential(R, M, delta, S[1]))
    mid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, M, delta, *S)
    s = N.simulate(
        list(S),
        [0.0, mid, delta],
        [R, na, R, nb],
        order=[1, 0],
        tmax=500.0,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * delta / 0.03,
        max_step=0.02,
        max_events=10**6,
    )
    return dict(R=R, M=M, status=s["status"], t_end=float(s["t_end"]), crossings=len(s["events"]))


def fate(delta, m, workers):
    comps, _ = components(delta, m, np.geomspace(delta, 3.0, 3000), 1500)
    c = comps[0]
    if c["R"][1] > 0.1:
        return dict(delta=delta, m=m, starts=0)
    pts = [
        (delta, m, R, M)
        for R in np.linspace(*c["R"], 14)[1:-1]
        for M in np.linspace(*c["M"], 14)[1:-1]
        if admissible(delta, m, np.array(R), np.array(M))[1]
    ]
    with Pool(workers) as p:
        runs = p.map(follow, pts, chunksize=4)
    return dict(
        delta=delta,
        m=m,
        starts=len(runs),
        outcomes=dict(Counter(r["status"] for r in runs)),
        t_end_max=max(r["t_end"] for r in runs),
        crossings_max=max(r["crossings"] for r in runs),
        runs=runs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    with Pool(a.workers) as pool:
        comps = pool.map(case_components, CASES)
        ons = pool.map(onset, ONSET)
        grids = pool.map(survey_grid, [c[1] for c in CASES[1:]])
    fates = [fate(d, m, a.workers) for d, m in FATE]
    for c in comps:
        print(
            f"{c['label']} delta {c['delta']:g}: R_h {c['R_horizon']:.5f} (lowest component starts there: "
            f"{c['lowest_component_at_horizon']}); "
            + "; ".join(
                f"R [{x['R'][0]:.5f}, {x['R'][1]:.5f}] M_mid [{x['M'][0]:.6f}, {x['M'][1]:.6f}]"
                for x in c["components"]
            )
            + f"; survey-test-only cells {c['survey_only']['cells']} (R <= {c['survey_only']['R_max']})"
        )
    first = next(o["delta"] for o in ons if o["near_horizon"])
    last = max(o["delta"] for o in ons if not o["near_horizon"] and o["delta"] < first)
    print(f"near-horizon component appears between delta = {last:g} and {first:g}")
    for g in grids:
        print(f"survey grid delta {g['delta']:g}: {g['starts']} starts, {g['allowed_not_full']} fail the full test")
    for f in fates:
        print(
            f"fate delta {f['delta']:g}: {f['starts']} starts, {f.get('outcomes')}, t <= {f.get('t_end_max')}, "
            f"crossings after the start <= {f.get('crossings_max')}"
        )
    out = dict(
        components=comps,
        onset=list(ons),
        onset_bracket=[last, first],
        survey_grids=grids,
        fate=fates,
        seconds=time.time() - t0,
    )
    json.dump(out, open(a.out, "w"), default=float)
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
