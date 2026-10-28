"""Transparent spherical pressure shells in AdS, G=c=ell=1.

The time coordinate is synchronized by induced-metric matching, anchored to
one vacuum region (the exterior by default). Only the ordinary untrapped
junction branch is supported. Material identities persist through crossings.
"""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Shell:
    name: str
    m: float
    w: float = 0.2

    def mu(self, radius):
        return self.m * np.asarray(radius) ** (-2 * self.w)


def f(radius, mass):
    return 1 - 2 * mass / np.asarray(radius) + np.asarray(radius) ** 2


def potential(radius, lo, hi, shell):
    mu = shell.mu(radius)
    return (
        1
        + np.asarray(radius) ** 2
        - (hi + lo) / radius
        - (hi - lo) ** 2 / mu**2
        - mu**2 / (4 * np.asarray(radius) ** 2)
    )


def derivative(radius, lo, hi, shell):
    mu = shell.mu(radius)
    return (
        2 * radius
        + (hi + lo) / radius**2
        - 4 * shell.w * (hi - lo) ** 2 / (radius * mu**2)
        + (2 + 4 * shell.w) * mu**2 / (4 * radius**3)
    )


def crossing(radius, va, vb, lo, mid, hi, a, b):
    """A is inner before crossing and B outer; return their material velocities."""
    ma, mb = a.mu(radius), b.mu(radius)
    fm = f(radius, mid)
    if fm <= 0:
        raise ValueError("Crossing outside the supported untrapped region")
    ba, bb = np.sqrt(va * va + fm), np.sqrt(vb * vb + fm)
    gamma = (ba * bb - va * vb) / fm
    rapidity_sinh = (bb * va - ba * vb) / fm
    transfer = ma * mb * gamma / radius
    return lo + hi - mid + transfer, va - mb * rapidity_sinh / radius, vb - ma * rapidity_sinh / radius, gamma, transfer


def clock_factors(state, masses, order, shells, anchor=None):
    """Return d tau_i / dT and dt_region / dT, indexed by material/radial rank.

    Across a shell: dt_inside/dt_outside = beta_inside*f_outside /
    (f_inside*beta_outside). Normalizing the exterior lapse gives any desired
    anchor region. This construction is valid while every f is positive.
    """
    n = len(shells)
    anchor = n if anchor is None else anchor
    if not 0 <= anchor <= n:
        raise ValueError("Bad anchor region")
    lapses = np.ones(n + 1)
    raw = np.zeros(n)
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        radius, velocity = state[2 * i : 2 * i + 2]
        fi, fo = f(radius, masses[rank]), f(radius, masses[rank + 1])
        if min(fi, fo) <= 0:
            raise ValueError("A lapse evaluation reached f<=0")
        bi, bo = np.sqrt(velocity * velocity + fi), np.sqrt(velocity * velocity + fo)
        raw[i] = lapses[rank + 1] * fo / bo
        lapses[rank] = lapses[rank + 1] * bi * fo / (fi * bo)
    normalization = lapses[anchor]
    return raw / normalization, lapses / normalization


def constraint_error(state, masses, order, shells):
    worst = 0.0
    for rank, i in enumerate(order):
        radius, velocity = state[2 * i : 2 * i + 2]
        lo, hi = masses[rank : rank + 2]
        p = potential(radius, lo, hi, shells[i])
        err = abs(velocity * velocity + p) / (1 + velocity * velocity + abs(p))
        bi = np.sqrt(velocity * velocity + f(radius, lo))
        bo = np.sqrt(velocity * velocity + f(radius, hi))
        junction = abs((bi - bo) - shells[i].mu(radius) / radius) / (1 + bi + bo)
        worst = max(worst, float(err), float(junction))
    return worst


def simulate(
    shells,
    masses,
    state,
    *,
    order=None,
    tmax=200.0,
    anchor=None,
    rtol=1e-10,
    atol=None,
    max_step=0.05,
    max_events=10000,
    horizon_cutoff=1e-6,
    collect=False,
):
    """Integrate all shells; solve_ivp selects the next adjacent-pair event.

    No distributed pending-event clocks are needed with this common slicing.
    Generic binary crossings are supported; simultaneous triple collisions
    are rejected rather than resolved by an arbitrary order.
    """
    n = len(shells)
    masses = np.asarray(masses, dtype=float).copy()
    state = np.asarray(state, dtype=float).copy()
    order = list(range(n)) if order is None else list(order)
    if len(masses) != n + 1 or len(state) != 2 * n or sorted(order) != list(range(n)):
        raise ValueError("Inconsistent state dimensions")
    if any(s.m <= 0 for s in shells) or np.any(np.diff(masses) <= 0):
        raise ValueError("Require positive shell parameters and region mass jumps")
    if any(state[2 * order[k]] > state[2 * order[k + 1]] + 1e-12 for k in range(n - 1)):
        raise ValueError("Shell order disagrees with radius")
    if constraint_error(state, masses, order, shells) > 1e-7:
        raise ValueError("Initial data violates a junction constraint")
    atol = rtol * 0.01 if atol is None else atol
    anchor = n if anchor is None else anchor
    initial = {
        "shells": [asdict(s) for s in shells],
        "masses": masses.tolist(),
        "state": state.tolist(),
        "order": order.copy(),
        "anchor": anchor,
        "tmax": tmax,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "horizon_cutoff": horizon_cutoff,
    }
    events, segments = [], []
    time = 0.0
    worst = constraint_error(state, masses, order, shells)
    status = "time_limit"
    while time < tmax and len(events) < max_events:

        def rhs(t, y):
            factors, _ = clock_factors(y, masses, order, shells, anchor)
            out = np.zeros(2 * n)
            for rank, i in enumerate(order):
                radius, velocity = y[2 * i : 2 * i + 2]
                out[2 * i] = velocity * factors[i]
                out[2 * i + 1] = -0.5 * derivative(radius, masses[rank], masses[rank + 1], shells[i]) * factors[i]
            return out

        meets = []
        for rank in range(n - 1):
            i, j = order[rank : rank + 2]

            def meet(t, y, i=i, j=j):
                return y[2 * i] - y[2 * j]

            meet.terminal = True
            meet.direction = 1
            meets.append(meet)

        def horizon(t, y):
            return (
                min(min(f(y[2 * i], masses[k]), f(y[2 * i], masses[k + 1])) for k, i in enumerate(order))
                - horizon_cutoff
            )

        horizon.terminal = True
        horizon.direction = -1
        # Starting on the stopping surface would otherwise request another step.
        if horizon(time, state) <= 0:
            status = "horizon_approach"
            break
        sol = solve_ivp(
            rhs,
            (time, tmax),
            state,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=meets + [horizon],
            dense_output=collect,
        )
        for column in range(sol.y.shape[1]):
            worst = max(worst, constraint_error(sol.y[:, column], masses, order, shells))
        if collect:
            segments.append(
                {"t": sol.t.copy(), "y": sol.y.copy(), "masses": masses.copy(), "order": order.copy(), "dense": sol.sol}
            )
        time = float(sol.t[-1])
        state = sol.y[:, -1].copy()
        if not sol.success:
            status = "integrator_failed"
            break
        if len(sol.t_events[-1]):
            status = "horizon_approach"
            break
        hit = [k for k, e in enumerate(sol.t_events[:-1]) if len(e)]
        if not hit:
            break
        rank = hit[0]
        a, b = order[rank : rank + 2]
        radius = float((state[2 * a] + state[2 * b]) / 2)
        neighbors = [j for j in order if j not in (a, b) and abs(state[2 * j] - radius) < 1e-9]
        if neighbors:
            status = "triple_collision"
            break
        va, vb = state[2 * a + 1], state[2 * b + 1]
        before_state = state.copy()
        before_masses = masses.copy()
        mp, na, nb, gamma, transfer = crossing(radius, va, vb, *masses[rank : rank + 3], shells[a], shells[b])
        masses[rank + 1] = mp
        state[2 * a] = radius
        state[2 * b] = radius
        state[2 * a + 1] = na
        state[2 * b + 1] = nb
        order[rank : rank + 2] = [b, a]
        event = {
            "number": len(events) + 1,
            "t": time,
            "r": radius,
            "pair": [shells[a].name, shells[b].name],
            "loser": shells[a].name,
            "gainer": shells[b].name,
            "rank": rank,
            "masses_before": before_masses.tolist(),
            "masses_after": masses.tolist(),
            "v_before": [float(va), float(vb)],
            "v_after": [float(na), float(nb)],
            "gamma": float(gamma),
            "delta_e": float(transfer),
            "state_before": before_state.tolist(),
            "state_after": state.tolist(),
            "order_after": [shells[i].name for i in order],
        }
        if np.any(np.diff(masses) <= 0):
            event["constraint_residual"] = None
            events.append(event)
            status = "nonpositive_mass_jump"
            break
        if horizon(time, state) <= 0:
            event["constraint_residual"] = None
            events.append(event)
            status = "horizon_approach"
            break
        event["constraint_residual"] = constraint_error(state, masses, order, shells)
        worst = max(worst, event["constraint_residual"])
        events.append(event)
    if len(events) >= max_events:
        status = "event_limit"
    return {
        "parameters": initial,
        "status": status,
        "t_end": time,
        "state_end": state.tolist(),
        "masses_end": masses.tolist(),
        "order_end": [shells[i].name for i in order],
        "events": events,
        "max_constraint_residual": worst,
        "segments": segments,
    }


def serializable(run):
    return {k: v for k, v in run.items() if k != "segments"}


def state_on_orbit(radius, lo, hi, shell, sign):
    p = potential(radius, lo, hi, shell)
    if p >= 0:
        raise ValueError(f"Forbidden radius {radius} for shell {shell.name}")
    return float(sign * np.sqrt(-p))
