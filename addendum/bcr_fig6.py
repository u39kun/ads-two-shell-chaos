"""Reconstruct Figure 4a's separation data with two outgoing-velocity sign rules.

copied: reuse incoming signs; corrected: use transformed proper-velocity signs.
Reads bcr_fig6_separation.npy (t, |Delta R_1|) and bcr_fig6_initial.npy
(dR/dt, R; outer then inner shell) from addendum/data/, in BCR's L = 20 units.
Writes OUT.npz with t, archived, copied, and corrected separations in the paper's
L = 1 units, and OUT.json with reconstruction diagnostics.

usage (from code/): python addendum/bcr_fig6.py addendum/results/bcr_fig6"""

import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
L, W = 20.0, 0.2
TOTAL, MASS = 0.6, 0.54
R_START, R_NEARBY = 20.0, 20.001
RTOL, RTOL_REFINED = 2e-11, 2e-13
CUTOFFS = (50, 100, 200, 400, 640)  # paper's time units


class ShellModel:
    def __init__(self, total=TOTAL, mass=MASS):
        self.total, self.mass = total, mass

    def quantities(self, r, mid, outer):
        """Middle-region f, beta, (dR/dt)^2 and coordinate acceleration of the outer or inner shell."""
        mu = self.mass * r ** (-2 * W)
        f = 1 - 2 * mid / r + (r / L) ** 2
        fp = 2 * mid / r**2 + 2 * r / L**2
        energy = self.total - mid if outer else mid
        sign = 1 if outer else -1
        beta = energy / mu + sign * mu / (2 * r)
        beta_p = 2 * W * energy / (r * mu) - sign * (2 * W + 1) * mu / (2 * r * r)
        speed2 = f * f * (1 - f / beta**2)
        # one half the radial derivative of speed2, with the region masses fixed
        acceleration = f * fp - 1.5 * f * f * fp / beta**2 + f**3 * beta_p / beta**3
        return f, beta, speed2, acceleration

    def crossing(self, r, mid, v_outer, v_inner):
        """Transparent crossing: new middle mass, incoming and outgoing proper radial velocities (outer, inner).

        The shells pass through each other, so the outgoing outer shell is the incoming inner shell."""
        f, bo, _, _ = self.quantities(r, mid, True)
        _, bi, _, _ = self.quantities(r, mid, False)
        uo, ui = v_outer * bo / f, v_inner * bi / f
        mu = self.mass * r ** (-2 * W)
        gamma = (bo * bi - uo * ui) / f
        S = (bo * ui - bi * uo) / f  # S > 0 when the inner shell meets or overtakes the outer shell
        new_mid = self.total - mid + mu * mu * gamma / r  # from the incoming velocities, for both rules
        new_u = np.array([ui - mu * S / r, uo - mu * S / r])
        return new_mid, np.array([uo, ui]), new_u

    def gamma(self, r, mid, u):
        """Relative Lorentz factor of the outer and inner shells with proper radial velocities u."""
        f, bo, _, _ = self.quantities(r, mid, True)
        _, bi, _, _ = self.quantities(r, mid, False)
        return (bo * bi - u[0] * u[1]) / f

    def integrate(self, radius, end_time, rule, rtol=RTOL):
        mid, t = self.total / 2, 0.0
        signs = np.array([1.0, -1.0])  # outer expanding, inner contracting
        segments, events = [], []
        while t < end_time:
            speed2 = np.array([self.quantities(radius, mid, q)[2] for q in (True, False)])
            if np.min(speed2) < -1e-7:
                raise RuntimeError("forbidden restart velocity: " + str(speed2))
            y0 = np.r_[radius, radius, signs * np.sqrt(np.maximum(speed2, 0))]
            if y0[2] <= y0[3]:
                raise RuntimeError("restart must separate the shells")

            def rhs(tt, y):
                return [y[2], y[3], self.quantities(y[0], mid, True)[3], self.quantities(y[1], mid, False)[3]]

            def crossing_event(tt, y):
                # exclude the coincident initial state, not a subsequent encounter
                return y[0] - y[1] if tt > t + 1e-8 else 1e-9

            crossing_event.terminal = True
            crossing_event.direction = -1
            sol = solve_ivp(rhs, (t, end_time), y0, method="DOP853", events=crossing_event, rtol=rtol,
                            atol=rtol * 0.01, max_step=1.0, dense_output=True)
            if not sol.success:
                raise RuntimeError(sol.message)
            end = sol.t[-1]
            segments.append((end, sol.sol))
            if end >= end_time:
                break
            radius = float((sol.y[0, -1] + sol.y[1, -1]) / 2)
            vo, vi = sol.y[2:, -1]
            new_mid, old_u, new_u = self.crossing(radius, mid, vo, vi)
            copied, corrected = np.sign([vi, vo]), np.sign(new_u)
            events.append(dict(crossing=len(events) + 1, t=float(end), r=radius, mid_before=float(mid),
                               mid_after=float(new_mid), incoming_u=old_u.tolist(), outgoing_u=new_u.tolist(),
                               copied_signs=copied.tolist(), corrected_signs=corrected.tolist()))
            signs = copied if rule == "copied" else corrected
            mid, t = new_mid, end
            if len(events) > 3000:
                raise RuntimeError("unexpectedly many crossings")
        return segments, events


def radii(segments, times):
    """Outermost radius at the given times from the dense output of each segment."""
    indices = np.searchsorted([s[0] for s in segments], times)
    values = np.empty(len(times))
    for i in np.unique(indices):
        mask = indices == i
        values[mask] = segments[i][1](times[mask])[0]
    return values


def first_difference(events):
    return next((e for e in events if e["copied_signs"] != e["corrected_signs"]), None)


def crossing_check(model):
    """gamma_AB across the first crossing at which the two rules differ, at the refined tolerance."""
    _, events = model.integrate(R_START, 1000.0, "corrected", rtol=RTOL_REFINED)
    e = first_difference(events)
    r, mid, new_mid = e["r"], e["mid_before"], e["mid_after"]
    u_in, u_out = np.array(e["incoming_u"]), np.array(e["outgoing_u"])
    u_copied = np.abs(u_out) * np.array(e["copied_signs"])
    return dict(
        crossing=e["crossing"], t=e["t"] / L, R=r / L,
        incoming_proper_velocities=u_in.tolist(),
        outgoing_proper_velocities_corrected=u_out.tolist(),
        outgoing_proper_velocities_copied=u_copied.tolist(),
        gamma_before=float(model.gamma(r, mid, u_in)),
        gamma_after_corrected=float(model.gamma(r, new_mid, u_out)),
        gamma_after_copied=float(model.gamma(r, new_mid, u_copied)),
    )


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    archive = np.load(os.path.join(HERE, "data", "bcr_fig6_separation.npy"))
    initial = np.load(os.path.join(HERE, "data", "bcr_fig6_initial.npy"))
    times, archived = archive[:, 0], archive[:, 1]
    vo, vi = initial[0, 0], initial[1, 0]

    def residual(p):  # initial outer and inner velocities at R = 20 with M_mid = M/2
        model = ShellModel(*p)
        return [np.sqrt(model.quantities(R_START, p[0] / 2, True)[2]) - vo,
                np.sqrt(model.quantities(R_START, p[0] / 2, False)[2]) + vi]

    fit = root(residual, [0.5, 0.45])
    if not fit.success or np.max(np.abs(residual(fit.x))) > 1e-10:
        raise RuntimeError("could not infer the parameters from the initial velocities")
    print(f"inferred from the archived initial velocities: M = {fit.x[0]:.12f}, m = {fit.x[1]:.12f}; "
          f"used: M = {TOTAL}, m = {MASS}, M_mid = {TOTAL / 2}, w = {W}, L = {L:g}")
    print(f"paper's units: m = {MASS / L ** (1 + 2 * W):.10f}; R_i = {R_START / L:g}, {R_NEARBY / L:g}")
    model = ShellModel()
    t0 = time.time()
    separation, crossings = {}, {}
    for rule in ("copied", "corrected"):
        start, start_events = model.integrate(R_START, times[-1], rule)
        nearby, nearby_events = model.integrate(R_NEARBY, times[-1], rule)
        separation[rule] = np.abs(radii(start, times) - radii(nearby, times)) / L
        crossings[rule] = (start_events, nearby_events)
        print(f"{rule}: {len(start_events)} and {len(nearby_events)} crossings to t = {times[-1] / L:g}")
    t_run = time.time() - t0
    t, archived = times / L, archived / L
    differ = [dict(crossing=e["crossing"], t=e["t"] / L, R=e["r"] / L)
              for e in crossings["copied"][0] if e["copied_signs"] != e["corrected_signs"]]
    result = dict(
        inferred_parameters=dict(M=float(fit.x[0]), m=float(fit.x[1])),
        parameters=dict(M=TOTAL, m=MASS, M_mid=TOTAL / 2, w=W, L=L, R_i=[R_START, R_NEARBY], rtol=RTOL,
                        atol=RTOL * 0.01, max_step=1.0, rtol_refined=RTOL_REFINED),
        m_paper_units=MASS / L ** (1 + 2 * W),
        samples=len(t), t_end=float(t[-1]),
        max_separation=dict(archived=float(archived.max()), copied=float(separation["copied"].max()),
                            corrected=float(separation["corrected"].max())),
        rms_copied_vs_archived={str(c): float(np.sqrt(np.mean((separation["copied"] - archived)[t <= c] ** 2)))
                                for c in CUTOFFS},
        copied_run_sign_differences=differ,
        first_sign_difference=crossing_check(model),
    )
    np.savez_compressed(out + ".npz", t=t, archived=archived, copied=separation["copied"],
                        corrected=separation["corrected"])
    with open(out + ".json", "w") as f:
        json.dump(result, f, indent=1)
    for key in ("max_separation", "rms_copied_vs_archived", "first_sign_difference"):
        print(key, json.dumps(result[key], indent=1))
    print(f"crossings in the copied-sign run where the rules differ: {[d['crossing'] for d in differ]}")
    print(f"wrote {out}.npz and {out}.json; integration {t_run:.0f} s, total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
