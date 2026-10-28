"""Numerical check of the invariant measure of the two-shell map (App. A of the paper).

Two identical shells A and B (m = 0.9*20^(-0.4)*0.03, w = 0.2, region masses 0, M_mid, and 0.03).
Section: states just before a head-on crossing with A inside (A moving out, B moving in), in the chart
x = (R, M_mid), R the crossing radius. On this branch the junction conditions fix both velocities, so x fixes the
state. F: the return map (flights and crossings up to the next head-on crossing with A inside).
Claim: F preserves g dR dM_mid with g = |1/X_B - 1/X_A|, X = dR/dt = f v/beta in the middle region's static time,
so det DF(x) = g(x)/g(F(x)) and the cumulative sum of log det DF along an orbit telescopes to log g(x_0) - log g(x_N).
The algebraic identities behind the claim are checked symbolically by sympy_crossing_proof.py.
usage: python two_shell_measure.py OUT.json"""

import os, sys, json

os.environ["SOLVER"] = "reference"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import N, A, B, TOTAL

SCALE = np.array([1.0, 0.01])  # M_mid is ~100x smaller than R
CENTER = np.array([0.7894196446, 0.0115548500])  # lower center of the two-crossing cycle (labels swapped: A light)


def velocities(R, mid):
    pa, pb = N.potential(R, 0.0, mid, A), N.potential(R, mid, TOTAL, B)
    if pa >= 0 or pb >= 0:
        raise ValueError(f"forbidden section point R={R}, M_mid={mid}")
    return np.sqrt(-pa), -np.sqrt(-pb)


def g(x):
    R, mid = x
    va, vb = velocities(R, mid)
    fm = float(N.f(R, mid))
    XA, XB = va * fm / np.sqrt(va * va + fm), vb * fm / np.sqrt(vb * vb + fm)
    return abs(1 / XB - 1 / XA)


def F(x, rtol=1e-12):
    R, mid = x
    va, vb = velocities(R, mid)
    newmid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, mid, TOTAL, A, B)
    r = N.simulate(
        [A, B],
        [0.0, newmid, TOTAL],
        [R, na, R, nb],
        order=[1, 0],
        tmax=60,
        anchor=1,
        rtol=rtol,
        atol=rtol * 1e-2,
        max_step=0.02,
        max_events=2,
    )
    if len(r["events"]) < 2:
        raise RuntimeError(f'map undefined: {r["status"]}')
    e = r["events"][1]
    if e["loser"] != "A" or not (e["v_before"][0] > 0 > e["v_before"][1]):
        raise RuntimeError("next A-inside crossing is not head-on")
    return np.array([e["r"], e["masses_before"][1]])


def jacobian(x, h=1e-6, **kw):
    J = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = h * SCALE[j]
        J[:, j] = (F(x + d, **kw) - F(x - d, **kw)) / (2 * d[j])
    return J


def main():
    out = dict(points=[], orbit={})
    J = jacobian(CENTER)
    ev = np.linalg.eigvals(J)
    rho = float(np.abs(np.angle(ev[0])) / (2 * np.pi))
    out["center"] = dict(
        x=CENTER.tolist(),
        residual=(F(CENTER) - CENTER).tolist(),
        eigenvalues_abs=np.abs(ev).tolist(),
        rotation_number=rho,
    )
    print(f"elliptic fixed point: |eigenvalues| = {np.abs(ev)}, rotation number {rho:.5f} per return")
    # energy split at the elliptic fixed point (A-heavy labeling): energies just before the head-on crossing and
    # the transfer
    xh = np.array([0.7894196446, 0.0186390612])
    va, vb = velocities(*xh)
    newmid, _, _, _, dE = N.crossing(xh[0], va, vb, 0.0, xh[1], TOTAL, A, B)
    out["energy_split"] = dict(
        E_A=float(xh[1]), E_B=float(TOTAL - xh[1]), share_A=float(xh[1] / TOTAL), transfer=float(dE)
    )
    print(
        f"energy split at the elliptic fixed point: E_A = {xh[1]:.6f}, E_B = {TOTAL - xh[1]:.6f} (A carries {xh[1]/TOTAL:.1%}); "
        f"transfer per crossing {dE:.3e}"
    )
    for x in ((0.800, 0.018), (0.850, 0.014)):
        x = np.array(x)
        J = jacobian(x)
        Fx = F(x)
        row = dict(
            x=x.tolist(),
            det=float(np.linalg.det(J)),
            g_ratio=float(g(x) / g(Fx)),
            residual=float(np.log(np.linalg.det(J)) + np.log(g(Fx)) - np.log(g(x))),
        )
        print(
            f'x = ({x[0]:.3f}, {x[1]:.3f}): det DF = {row["det"]:.8f}, g(x)/g(F(x)) = {row["g_ratio"]:.8f}, '
            f'log residual {row["residual"]:+.1e}'
        )
        out["points"].append(row)
    x = CENTER + np.array([0.06, 0.0])
    x0 = x.copy()
    tot, Sn, res = 0.0, [], []
    for k in range(60):
        J = jacobian(x)
        Fx = F(x)
        ld = float(np.log(np.linalg.det(J)))
        tot += ld
        Sn.append(tot)
        res.append(ld + np.log(g(Fx)) - np.log(g(x)))
        x = Fx
    res = np.abs(np.array(res))
    out["orbit"] = dict(
        x0=x0.tolist(),
        returns=60,
        cumulative_logdet_range=[min(Sn), max(Sn)],
        telescoped=float(np.log(g(x0)) - np.log(g(x))),
        cumulative_last=Sn[-1],
        max_residual=float(res.max()),
        median_residual=float(np.median(res)),
    )
    print(
        f"island orbit, 60 returns: cumulative log det in [{min(Sn):+.4f}, {max(Sn):+.4f}], last {Sn[-1]:+.6f} "
        f'vs log g(x0) - log g(x60) = {out["orbit"]["telescoped"]:+.6f}; per-return residual max {res.max():.1e}'
    )
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
