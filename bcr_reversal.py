"""The two velocity assignments of BCR's examples are one orbit run in opposite directions of time.

BCR start both shells at the same radius R_i, one contracting and one expanding, without saying which. The speeds are
fixed by the energy equations, |dR/dtau| = sqrt(-V(R_i)), so only the signs are open:
  D: inner shell contracting, outer expanding (the shells separate at t = 0);
  C: inner shell expanding, outer contracting (the shells cross at t = 0).
C's initial state is D's with both velocities reversed. If the evolution, including the crossing rule, is
time-reversible, C's run is therefore D's run continued into the past.

Test of reversibility: run D forward to T, reverse both velocities, run forward again for T - t0, and compare with D's
state at t0 with its velocities reversed (t0 = 10 avoids stopping at the crossing at t = 0). Clock: static time of the
region between the shells (anchor 1), as in bcr_lyapunov.py and bcr_delta_scan.py.
Cases: BCR's Fig. 5 example (m = 0.00815, R_i = 1) and type (ii) data (m = 0.9*20^(-0.4)*delta, R_i = 1.5).
usage: python bcr_reversal.py OUT.json"""

import os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import N

W, T0 = 0.2, 10.0
KW = dict(anchor=1, rtol=1e-12, max_step=0.02, max_events=200000)


def case(m, total, ri):
    A, B = N.Shell("A", m, W), N.Shell("B", m, W)
    masses = [0.0, total / 2, total]
    sa = np.sqrt(-N.potential(ri, masses[0], masses[1], A))
    sb = np.sqrt(-N.potential(ri, masses[1], masses[2], B))
    return [A, B], dict(masses=masses, state=[ri, -sa, ri, sb], order=[0, 1]), 1e-14 * total / 0.03


def run(shells, atol, masses, state, order, T):
    return N.simulate(shells, masses, state, order=order, tmax=T, atol=atol, **KW)


def check(name, m, total, ri, Ts):
    shells, init, atol = case(m, total, ri)
    names = {s.name: i for i, s in enumerate(shells)}
    ref = run(shells, atol, **init, T=T0)
    target = np.array(ref["state_end"])
    target[1::2] *= -1
    out = []
    for T in Ts:
        t1 = time.time()
        fwd = run(shells, atol, **init, T=T)
        st = np.array(fwd["state_end"])
        st[1::2] *= -1
        back = run(shells, atol, fwd["masses_end"], st.tolist(), [names[x] for x in fwd["order_end"]], T - T0)
        same_order = back["order_end"] == ref["order_end"]
        d_state = float(np.abs(np.array(back["state_end"]) - target).max()) if same_order else None
        d_mass = float(np.abs(np.array(back["masses_end"]) - np.array(ref["masses_end"])).max())
        r = dict(
            T=T,
            crossings_forward=len(fwd["events"]),
            crossings_back=len(back["events"]),
            same_order=same_order,
            max_state_difference=d_state,
            max_mass_difference=d_mass,
            seconds=time.time() - t1,
        )
        print(
            f"{name}  T = {T:g}: crossings {r['crossings_forward']} / {r['crossings_back']}, order kept {same_order}, "
            f"max |state difference| {d_state:.1e}, max |mass difference| {d_mass:.1e}",
            flush=True,
        )
        out.append(r)
    return dict(case=name, m=m, total_mass=total, R_i=ri, t0=T0, runs=out)


def main():
    res = [check("Fig. 5 example", 0.00815, 0.03, 1.0, (300.0, 3000.0, 20000.0))]
    for d in (0.006, 0.02, 0.037):
        res.append(check(f"type (ii), delta = {d}", 0.9 * 20 ** (-2 * W) * d, d, 1.5, (3000.0, 20000.0)))
    json.dump(
        dict(clock="static time between the shells (anchor 1)", rtol=KW["rtol"], cases=res),
        open(sys.argv[1], "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
