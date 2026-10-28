"""A second implementation of the two-shell model, written separately from n_shells.py, and the comparison.

What is shared and what is not:
  shared     the physical model: thin shells with mu(R) = m R^(-2w), the Israel junction condition (Eq. (2) of the
             paper), transparent crossings, and the clock (static time t of the region between the shells, BCR's
             evolution variable); SciPy's DOP853 as the integrator.
  different  variables: coordinates R and X = dR/dt (BCR's Eq. (14)) instead of R and dR/dtau with lapse matching;
             equations of motion: d^2R/dt^2 = -Vhat'(R)/2 with Vhat = f^2 V/(f - V), where V = -(dR/dtau)^2 is taken
             from the junction condition solved for dR/dtau and Vhat' is differentiated symbolically (SymPy), not
             copied from the hand-written potential derivative;
             the crossing: built geometrically, by composing the jumps of grad R across each shell in a light-cone
             frame (the model of sympy_crossing_proof.py) and reading off the new middle mass and velocities, instead
             of the closed-form update of Eq. (3);
             event location: SciPy's root finding on the dense output, as in n_shells.py, not the regula falsi on
             re-integrated steps of the compiled solver fastshells.py.
The comparison runs both codes from the same initial data and compares the crossing sequences (time, radius, and
middle mass after each crossing).
usage: python independent_check.py OUT.json [--tmax 300]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["SOLVER"] = "reference"  # compare with the reference solver n_shells.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore", message="At least one element of `rtol` is too small")

# ------------------------------------------------------------------ the model, from the junction condition
_R, _lo, _hi, _Mm, _m, _w = sp.symbols("R lo hi Mm m w", positive=True)
_f = lambda M: 1 - 2 * M / _R + _R**2
_mu = _m * _R ** (-2 * _w)
_Rdot2 = ((_hi - _lo + _mu**2 / (2 * _R)) / _mu) ** 2 - _f(_lo)  # Eq. (2) solved for (dR/dtau)^2
_V = -_Rdot2
_Vhat = _f(_Mm) ** 2 * _V / (_f(_Mm) - _V)  # (dR/dt)^2 = -Vhat, t = middle-region time
Rdot2 = sp.lambdify((_R, _lo, _hi, _m, _w), _Rdot2, "math")
dVhat = sp.lambdify((_R, _lo, _hi, _Mm, _m, _w), sp.diff(_Vhat, _R), "math")


def fz(R, M):
    return 1.0 - 2.0 * M / R + R * R


def X_from_Rdot(Rdot, R, Mm):
    f = fz(R, Mm)
    return Rdot * f / np.sqrt(Rdot * Rdot + f)


def Rdot_from_X(X, R, Mm):
    f = fz(R, Mm)
    return X * np.sqrt(f) / np.sqrt(f * f - X * X)


def dot(a, b):  # light-cone components, ds^2 = -dx+ dx-
    return -(a[0] * b[1] + a[1] * b[0]) / 2


def crossing(R, XA, XB, Min, Mm, Mout, mA, mB, w):
    """Geometric crossing of inner shell A and outer shell B at radius R: returns (M', XA', XB') with X in the new
    middle region's time. Frame: A at rest, uA = (1, 1), nA = (1, -1); grad R in the middle region g = (p, q)."""
    RdA, RdB = Rdot_from_X(XA, R, Mm), Rdot_from_X(XB, R, Mm)
    f = fz(R, Mm)
    bA = np.sqrt(RdA * RdA + f)
    g = (-RdA + bA, -RdA - bA)  # u_A.g = RdA, n_A.g = bA > 0, g.g = f
    cand = []
    for sgn in (1, -1):  # u_B = (s, 1/s): u_B.g = RdB, n_B.g = +beta_B
        s = (-RdB + sgn * np.sqrt(RdB * RdB + f)) / g[1]
        if s > 0:
            uB, nB = (s, 1 / s), (s, -1 / s)
            if dot(nB, g) > 0:
                cand.append((uB, nB))
    uB, nB = cand[0]
    uA, nA = (1.0, 1.0), (1.0, -1.0)
    aA, aB = mA * R ** (-2 * w) / R, mB * R ** (-2 * w) / R
    g_in = (g[0] + aA * nA[0], g[1] + aA * nA[1])  # across A inward
    g_out = (g[0] - aB * nB[0], g[1] - aB * nB[1])  # across B outward
    g_new = (g_in[0] - aB * nB[0], g_in[1] - aB * nB[1])
    checks = (dot(g_in, g_in) - fz(R, Min), dot(g_out, g_out) - fz(R, Mout))
    fn = dot(g_new, g_new)
    Mnew = R * (1 + R * R - fn) / 2
    RdA2, RdB2 = dot(uA, g_out), dot(uB, g_in)
    return Mnew, X_from_Rdot(RdA2, R, Mnew), X_from_Rdot(RdB2, R, Mnew), max(abs(c) for c in checks)


def evolve(m, w, masses, state, order, tmax, rtol=1e-13, atol=1e-15, max_events=100000):
    """state = [R_A, X_A, R_B, X_B] (X = dR/dt, middle clock); order = [inner index, outer index]."""
    Min, Mm, Mout = masses
    y = np.array(state, float)
    order = list(order)
    t = 0.0
    events = []
    worst = 0.0
    while t < tmax and len(events) < max_events:
        i, o = order
        lo = (Min, Mm)
        hi = (Mm, Mout)

        def rhs(_, z):
            out = np.empty(4)
            for rank, k in enumerate(order):
                out[2 * k] = z[2 * k + 1]
                out[2 * k + 1] = -0.5 * dVhat(z[2 * k], lo[rank], hi[rank], Mm, m[k], w)
            return out

        def meet(_, z):
            return z[2 * i] - z[2 * o]

        meet.terminal, meet.direction = True, 1
        sol = solve_ivp(rhs, (t, tmax), y, method="DOP853", rtol=rtol, atol=atol, events=meet)
        t = float(sol.t[-1])
        y = sol.y[:, -1].copy()
        if not len(sol.t_events[0]):
            break
        R = 0.5 * (y[2 * i] + y[2 * o])
        Mnew, Xi, Xo, chk = crossing(R, y[2 * i + 1], y[2 * o + 1], Min, Mm, Mout, m[i], m[o], w)
        worst = max(worst, chk)
        events.append(dict(t=t, r=R, M_before=Mm, M_after=Mnew, inner=i))
        y[2 * i] = y[2 * o] = R
        y[2 * i + 1] = Xi
        y[2 * o + 1] = Xo
        Mm = Mnew
        order = [o, i]
    return events, worst


def initial_independent(m, w, total, ri):
    """BCR start D: both shells at R_i, inner (A) contracting, outer (B) expanding; region masses 0, total/2, total."""
    Mm = total / 2
    RdA = -np.sqrt(Rdot2(ri, 0.0, Mm, m, w))
    RdB = np.sqrt(Rdot2(ri, Mm, total, m, w))
    return [0.0, Mm, total], [ri, X_from_Rdot(RdA, ri, Mm), ri, X_from_Rdot(RdB, ri, Mm)], [0, 1]


def compare(name, m, total, ri, tmax, rtols):
    from common import N
    import fastshells

    A, B = N.Shell("A", m), N.Shell("B", m)
    sa = np.sqrt(-N.potential(ri, 0.0, total / 2, A))
    sb = np.sqrt(-N.potential(ri, total / 2, total, B))
    out = dict(case=name, m=m, total=total, R_i=ri, tmax=tmax, rows=[])
    t0 = time.time()
    ind, worst = evolve([m, m], 0.2, *initial_independent(m, 0.2, total, ri), tmax, rtol=1e-13, atol=1e-15)
    ind2, _ = evolve([m, m], 0.2, *initial_independent(m, 0.2, total, ri), tmax, rtol=3e-14, atol=3e-16)
    n = min(len(ind), len(ind2))
    out["independent_self"] = dict(
        rtol=[1e-13, 3e-14],
        crossings=[len(ind), len(ind2)],
        max_dt=max(abs(x["t"] - y["t"]) for x, y in zip(ind[:n], ind2[:n])),
        max_dR=max(abs(x["r"] - y["r"]) for x, y in zip(ind[:n], ind2[:n])),
        max_dM=max(abs(x["M_after"] - y["M_after"]) for x, y in zip(ind[:n], ind2[:n])),
    )
    print(f"{name}: second implementation, rtol 1e-13 vs 3e-14: {out['independent_self']}", flush=True)
    out["independent_seconds"] = time.time() - t0
    out["junction_check"] = worst
    for solver, rt in [("n_shells", r) for r in rtols] + [("fastshells", r) for r in rtols]:
        sim = N.simulate if solver == "n_shells" else fastshells.simulate
        ref = sim(
            [A, B],
            [0.0, total / 2, total],
            [ri, -sa, ri, sb],
            order=[0, 1],
            tmax=tmax,
            anchor=1,
            rtol=rt,
            atol=rt * 1e-2 * total / 0.03,
            max_step=0.02,
            max_events=100000,
        )
        ev = ref["events"]
        n = min(len(ev), len(ind))
        same = all(e["loser"] == ["A", "B"][q["inner"]] for e, q in zip(ev[:n], ind[:n]))
        dt = max(abs(e["t"] - q["t"]) for e, q in zip(ev[:n], ind[:n]))
        dr = max(abs(e["r"] - q["r"]) for e, q in zip(ev[:n], ind[:n]))
        dm = max(abs(e["masses_after"][1] - q["M_after"]) for e, q in zip(ev[:n], ind[:n]))
        row = dict(
            solver=solver,
            rtol=rt,
            crossings=[len(ev), len(ind)],
            same_sequence=bool(same and len(ev) == len(ind)),
            max_dt=dt,
            max_dR=dr,
            max_dM=dm,
        )
        out["rows"].append(row)
        print(
            f"{name}: {solver} rtol {rt:g}: crossings {len(ev)} / {len(ind)}, same sequence {row['same_sequence']}, "
            f"max |dt| {dt:.1e}, |dR| {dr:.1e}, |dM_mid| {dm:.1e}",
            flush=True,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=300.0)
    a = p.parse_args()
    cases = [("Fig. 5 example", 0.00815, 0.03, 1.0)] + [
        (f"type (ii), delta = {d}", 0.9 * 20 ** (-0.4) * d, d, 1.5) for d in (0.006, 0.03, 0.037)
    ]
    res = [compare(*c, a.tmax, (1e-12, 1e-13)) for c in cases]
    json.dump(dict(tmax=a.tmax, independent_rtol=1e-13, cases=res), open(a.out, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
