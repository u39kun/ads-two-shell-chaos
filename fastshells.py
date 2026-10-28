"""Compiled drop-in for n_shells.simulate (transparent pressure shells in AdS, G = c = ell = 1).

Same physics, slicing, and event rules as the reference solver n_shells.py:
  * common time T by induced-metric lapse matching, anchored to one vacuum region (exterior by default);
  * each shell: dR/dT = v * dtau/dT,  dv/dT = -V'(R)/2 * dtau/dT  (v = dR/dtau);
  * terminal events: adjacent shells meet (inner minus outer radius crosses 0 upward), or
    min f over all shell sides falls through horizon_cutoff;
  * crossings use the same transparent-crossing update; triple collisions, non-positive mass jumps, and
    horizon approach are reported the same way;
  * returns the same dictionary (parameters, status, t_end, state_end, masses_end, order_end, events, ...).
Differences: the integrator is a numba DOP853 (same coefficients and step control as scipy's DOP853),
events are located by re-integrating single steps from the start of the step (no dense interpolant),
and a trial stage that would evaluate a lapse at f <= 0 shrinks the step instead of raising.
If the step size collapses below 1e-14, the call raises the ValueError that n_shells raises when a lapse
evaluation reaches f <= 0.
"""

from dataclasses import asdict, is_dataclass
import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients as _co

_A = np.ascontiguousarray(_co.A[:12, :12])
_B = np.ascontiguousarray(_co.B)
_E3 = np.ascontiguousarray(_co.E3)
_E5 = np.ascontiguousarray(_co.E5)

ST_TIME, ST_HORIZON, ST_EVENTS, ST_MASS, ST_TRIPLE, ST_FAIL = 0, 1, 2, 3, 4, 5
STATUS = {
    ST_TIME: "time_limit",
    ST_HORIZON: "horizon_approach",
    ST_EVENTS: "event_limit",
    ST_MASS: "nonpositive_mass_jump",
    ST_TRIPLE: "triple_collision",
    ST_FAIL: "integrator_failed",
}


@njit(cache=True)
def _f(R, M):
    return 1.0 - 2.0 * M / R + R * R


@njit(cache=True)
def _mu(m, w, R):
    return m * R ** (-2.0 * w)


@njit(cache=True)
def _pot(R, lo, hi, m, w):
    u = _mu(m, w, R)
    return 1.0 + R * R - (hi + lo) / R - (hi - lo) ** 2 / u**2 - u**2 / (4.0 * R * R)


@njit(cache=True)
def _dpot(R, lo, hi, m, w):
    u = _mu(m, w, R)
    return 2.0 * R + (hi + lo) / R**2 - 4.0 * w * (hi - lo) ** 2 / (R * u**2) + (2.0 + 4.0 * w) * u**2 / (4.0 * R**3)


@njit(cache=True)
def _rhs(y, masses, order, m, w, anchor, out):
    """Returns False if a lapse would be evaluated at f <= 0."""
    n = order.shape[0]
    lapses = np.ones(n + 1)
    raw = np.zeros(n)
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        R = y[2 * i]
        v = y[2 * i + 1]
        fi = _f(R, masses[rank])
        fo = _f(R, masses[rank + 1])
        if fi <= 0.0 or fo <= 0.0 or not (R > 0.0):
            return False
        bi = np.sqrt(v * v + fi)
        bo = np.sqrt(v * v + fo)
        raw[i] = lapses[rank + 1] * fo / bo
        lapses[rank] = lapses[rank + 1] * bi * fo / (fi * bo)
    nrm = lapses[anchor]
    for rank in range(n):
        i = order[rank]
        fac = raw[i] / nrm
        R = y[2 * i]
        out[2 * i] = y[2 * i + 1] * fac
        out[2 * i + 1] = -0.5 * _dpot(R, masses[rank], masses[rank + 1], m[i], w[i]) * fac
    return True


@njit(cache=True)
def _step(y, f0, h, masses, order, m, w, anchor, rtol, atol, K, ynew, fnew, tmp):
    """One DOP853 step of size h from (y, f0). Returns (ok, error_norm)."""
    nn = y.shape[0]
    for k in range(nn):
        K[0, k] = f0[k]
    for s in range(1, 12):
        for k in range(nn):
            acc = 0.0
            for j in range(s):
                acc += _A[s, j] * K[j, k]
            tmp[k] = y[k] + h * acc
        if not _rhs(tmp, masses, order, m, w, anchor, K[s]):
            return False, 0.0
    for k in range(nn):
        acc = 0.0
        for j in range(12):
            acc += _B[j] * K[j, k]
        ynew[k] = y[k] + h * acc
    if not _rhs(ynew, masses, order, m, w, anchor, fnew):
        return False, 0.0
    for k in range(nn):
        K[12, k] = fnew[k]
    e5 = 0.0
    e3 = 0.0
    for k in range(nn):
        sc = atol + max(abs(y[k]), abs(ynew[k])) * rtol
        a5 = 0.0
        a3 = 0.0
        for j in range(13):
            a5 += K[j, k] * _E5[j]
            a3 += K[j, k] * _E3[j]
        e5 += (a5 / sc) ** 2
        e3 += (a3 / sc) ** 2
    if e5 == 0.0 and e3 == 0.0:
        return True, 0.0
    return True, abs(h) * e5 / np.sqrt((e5 + 0.01 * e3) * nn)


@njit(cache=True)
def _events(y, masses, order, cutoff, g):
    """g[k] = R_inner - R_outer for adjacent ranks k = 0..n-2; g[n-1] = min f - cutoff."""
    n = order.shape[0]
    for k in range(n - 1):
        g[k] = y[2 * order[k]] - y[2 * order[k + 1]]
    hm = 1e300
    for rank in range(n):
        R = y[2 * order[rank]]
        a = _f(R, masses[rank])
        b = _f(R, masses[rank + 1])
        hm = min(hm, a, b)
    g[n - 1] = hm - cutoff


@njit(cache=True)
def _constraint(y, masses, order, m, w):
    worst = 0.0
    for rank in range(order.shape[0]):
        i = order[rank]
        R = y[2 * i]
        v = y[2 * i + 1]
        lo = masses[rank]
        hi = masses[rank + 1]
        p = _pot(R, lo, hi, m[i], w[i])
        err = abs(v * v + p) / (1.0 + v * v + abs(p))
        bi = np.sqrt(max(v * v + _f(R, lo), 0.0))
        bo = np.sqrt(max(v * v + _f(R, hi), 0.0))
        junc = abs((bi - bo) - _mu(m[i], w[i], R) / R) / (1.0 + bi + bo)
        worst = max(worst, err, junc)
    return worst


@njit(cache=True)
def _crossing(R, va, vb, lo, mid, hi, ma_, wa, mb_, wb):
    ma = _mu(ma_, wa, R)
    mb = _mu(mb_, wb, R)
    fm = _f(R, mid)
    ba = np.sqrt(va * va + fm)
    bb = np.sqrt(vb * vb + fm)
    gamma = (ba * bb - va * vb) / fm
    rs = (bb * va - ba * vb) / fm
    transfer = ma * mb * gamma / R
    return lo + hi - mid + transfer, va - mb * rs / R, vb - ma * rs / R, gamma, transfer, fm


@njit(cache=True)
def _core(
    y,
    masses,
    order,
    m,
    w,
    anchor,
    tmax,
    rtol,
    atol,
    max_step,
    max_events,
    cutoff,
    ev_t,
    ev_rank,
    ev_r,
    ev_mb,
    ev_ma,
    ev_vb,
    ev_va,
    ev_gamma,
    ev_tr,
    ev_sb,
    ev_sa,
    ev_oa,
    ev_res,
):
    n = order.shape[0]
    nn = 2 * n
    K = np.empty((13, nn))
    ynew = np.empty(nn)
    fnew = np.empty(nn)
    f0 = np.empty(nn)
    yr = np.empty(nn)
    fr = np.empty(nn)
    Kr = np.empty((13, nn))
    tmp = np.empty(nn)
    g0 = np.empty(n)
    g1 = np.empty(n)
    gr = np.empty(n)
    t = 0.0
    nev = 0
    status = ST_TIME
    worst = _constraint(y, masses, order, m, w)
    h = -1.0
    while t < tmax and nev < max_events:
        _events(y, masses, order, cutoff, g0)
        if g0[n - 1] <= 0.0:
            status = ST_HORIZON
            break
        if not _rhs(y, masses, order, m, w, anchor, f0):
            status = ST_FAIL
            break
        if h <= 0.0:  # initial step as in scipy's select_initial_step (order 7 error estimator)
            d0 = 0.0
            d1 = 0.0
            for k in range(nn):
                sc = atol + abs(y[k]) * rtol
                d0 += (y[k] / sc) ** 2
                d1 += (f0[k] / sc) ** 2
            d0 = np.sqrt(d0 / nn)
            d1 = np.sqrt(d1 / nn)
            h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
            h0 = min(h0, tmax - t)
            for k in range(nn):
                yr[k] = y[k] + h0 * f0[k]
            if _rhs(yr, masses, order, m, w, anchor, fr):
                d2 = 0.0
                for k in range(nn):
                    sc = atol + abs(y[k]) * rtol
                    d2 += ((fr[k] - f0[k]) / sc) ** 2
                d2 = np.sqrt(d2 / nn) / h0
                if d1 <= 1e-15 and d2 <= 1e-15:
                    h1 = max(1e-6, h0 * 1e-3)
                else:
                    h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
                h = min(100.0 * h0, h1, max_step)
            else:
                h = min(h0, max_step)
        # adaptive step
        rejected = False
        if tmax - t <= 1e-13 * max(1.0, tmax):
            t = tmax
            break
        while True:
            h = min(h, max_step, tmax - t)
            if h < 1e-14:
                return t, nev, ST_FAIL, worst
            ok, err = _step(y, f0, h, masses, order, m, w, anchor, rtol, atol, K, ynew, fnew, tmp)
            if not ok:
                h *= 0.5
                rejected = True
                continue
            if err < 1.0:
                if err == 0.0:
                    fac = 10.0
                else:
                    fac = min(10.0, 0.9 * err ** (-1.0 / 8.0))
                if rejected:
                    fac = min(1.0, fac)
                hstep = h
                h = h * fac
                break
            h *= max(0.2, 0.9 * err ** (-1.0 / 8.0))
            rejected = True
        # events in (t, t + hstep]
        _events(ynew, masses, order, cutoff, g1)
        first = -1
        tfirst = 1e300
        for k in range(n):
            up = g0[k] <= 0.0 and g1[k] >= 0.0
            down = g0[k] >= 0.0 and g1[k] <= 0.0
            active = up if k < n - 1 else down
            if not active:
                continue
            a = 0.0
            b = hstep
            ga = g0[k]
            gb = g1[k]
            side = 0
            for it in range(200):  # Illinois regula falsi on the single-step map
                if gb == ga:
                    c = 0.5 * (a + b)
                else:
                    c = b - gb * (b - a) / (gb - ga)
                if not (c > a and c < b):
                    c = 0.5 * (a + b)
                okr, _ = _step(y, f0, c, masses, order, m, w, anchor, rtol, atol, Kr, yr, fr, tmp)
                if not okr:
                    b = c
                    continue
                _events(yr, masses, order, cutoff, gr)
                gc = gr[k]
                if (gc >= 0.0 and k < n - 1) or (gc <= 0.0 and k == n - 1):
                    b = c
                    gb = gc
                    if side == -1:
                        ga *= 0.5
                    side = -1
                else:
                    a = c
                    ga = gc
                    if side == 1:
                        gb *= 0.5
                    side = 1
                if b - a <= 4e-16 * max(1.0, abs(t + b)) or gc == 0.0:
                    break
            if b < tfirst:
                tfirst = b
                first = k
        if first < 0:
            for k in range(nn):
                y[k] = ynew[k]
            t += hstep
            worst = max(worst, _constraint(y, masses, order, m, w))
            continue
        okr, _ = _step(y, f0, tfirst, masses, order, m, w, anchor, rtol, atol, Kr, yr, fr, tmp)
        for k in range(nn):
            y[k] = yr[k]
        t += tfirst
        worst = max(worst, _constraint(y, masses, order, m, w))
        if first == n - 1:
            status = ST_HORIZON
            break
        rank = first
        ia = order[rank]
        ib = order[rank + 1]
        radius = 0.5 * (y[2 * ia] + y[2 * ib])
        triple = False
        for k in range(n):
            j = order[k]
            if j != ia and j != ib and abs(y[2 * j] - radius) < 1e-9:
                triple = True
        if triple:
            status = ST_TRIPLE
            break
        va = y[2 * ia + 1]
        vb = y[2 * ib + 1]
        for k in range(nn):
            ev_sb[nev, k] = y[k]
        for k in range(n + 1):
            ev_mb[nev, k] = masses[k]
        mp, na, nb, gam, tr, fm = _crossing(
            radius, va, vb, masses[rank], masses[rank + 1], masses[rank + 2], m[ia], w[ia], m[ib], w[ib]
        )
        if fm <= 0.0:
            return t, nev, ST_FAIL, worst
        masses[rank + 1] = mp
        y[2 * ia] = radius
        y[2 * ib] = radius
        y[2 * ia + 1] = na
        y[2 * ib + 1] = nb
        order[rank] = ib
        order[rank + 1] = ia
        ev_t[nev] = t
        ev_rank[nev] = rank
        ev_r[nev] = radius
        ev_vb[nev, 0] = va
        ev_vb[nev, 1] = vb
        ev_va[nev, 0] = na
        ev_va[nev, 1] = nb
        ev_gamma[nev] = gam
        ev_tr[nev] = tr
        for k in range(n + 1):
            ev_ma[nev, k] = masses[k]
        for k in range(nn):
            ev_sa[nev, k] = y[k]
        for k in range(n):
            ev_oa[nev, k] = order[k]
        badmass = False
        for k in range(n):
            if masses[k + 1] - masses[k] <= 0.0:
                badmass = True
        _events(y, masses, order, cutoff, g1)
        if badmass:
            ev_res[nev] = np.nan
            nev += 1
            status = ST_MASS
            break
        if g1[n - 1] <= 0.0:
            ev_res[nev] = np.nan
            nev += 1
            status = ST_HORIZON
            break
        ev_res[nev] = _constraint(y, masses, order, m, w)
        worst = max(worst, ev_res[nev])
        nev += 1
    if nev >= max_events and status == ST_TIME:
        status = ST_EVENTS
    return t, nev, status, worst


def _constraint_py(state, masses, order, shells):
    m = np.array([s.m for s in shells], float)
    w = np.array([s.w for s in shells], float)
    return float(_constraint(np.asarray(state, float), np.asarray(masses, float), np.asarray(order, np.int64), m, w))


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
    if _constraint_py(state, masses, order, shells) > 1e-7:
        raise ValueError("Initial data violates a junction constraint")
    atol = rtol * 0.01 if atol is None else atol
    anchor = n if anchor is None else anchor
    initial = {
        "shells": [asdict(s) if is_dataclass(s) else dict(name=s.name, m=s.m, w=s.w) for s in shells],
        "masses": masses.tolist(),
        "state": state.tolist(),
        "order": order.copy(),
        "anchor": anchor,
        "tmax": tmax,
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "horizon_cutoff": horizon_cutoff,
        "solver": "fastshells",
    }
    cap = int(min(max_events, 1_000_000)) + 1
    ev_t = np.zeros(cap)
    ev_rank = np.zeros(cap, np.int64)
    ev_r = np.zeros(cap)
    ev_mb = np.zeros((cap, n + 1))
    ev_ma = np.zeros((cap, n + 1))
    ev_vb = np.zeros((cap, 2))
    ev_va = np.zeros((cap, 2))
    ev_g = np.zeros(cap)
    ev_tr = np.zeros(cap)
    ev_sb = np.zeros((cap, 2 * n))
    ev_sa = np.zeros((cap, 2 * n))
    ev_oa = np.zeros((cap, n), np.int64)
    ev_res = np.zeros(cap)
    m = np.array([s.m for s in shells], float)
    w = np.array([s.w for s in shells], float)
    o = np.array(order, np.int64)
    t, nev, st, worst = _core(
        state,
        masses,
        o,
        m,
        w,
        int(anchor),
        float(tmax),
        float(rtol),
        float(atol),
        float(max_step),
        int(max_events),
        float(horizon_cutoff),
        ev_t,
        ev_rank,
        ev_r,
        ev_mb,
        ev_ma,
        ev_vb,
        ev_va,
        ev_g,
        ev_tr,
        ev_sb,
        ev_sa,
        ev_oa,
        ev_res,
    )
    if st == ST_FAIL:
        raise ValueError("A lapse evaluation reached f<=0 (step size collapsed)")
    names = [s.name for s in shells]
    events = []
    prev_order = list(order)
    for k in range(nev):
        rank = int(ev_rank[k])
        a, b = prev_order[rank], prev_order[rank + 1]
        oa = [int(x) for x in ev_oa[k]]
        res = None if np.isnan(ev_res[k]) else float(ev_res[k])
        events.append(
            {
                "number": k + 1,
                "t": float(ev_t[k]),
                "r": float(ev_r[k]),
                "pair": [names[a], names[b]],
                "loser": names[a],
                "gainer": names[b],
                "rank": rank,
                "masses_before": ev_mb[k].tolist(),
                "masses_after": ev_ma[k].tolist(),
                "v_before": ev_vb[k].tolist(),
                "v_after": ev_va[k].tolist(),
                "gamma": float(ev_g[k]),
                "delta_e": float(ev_tr[k]),
                "state_before": ev_sb[k].tolist(),
                "state_after": ev_sa[k].tolist(),
                "order_after": [names[i] for i in oa],
                "constraint_residual": res,
            }
        )
        prev_order = oa
    return {
        "parameters": initial,
        "status": STATUS[st],
        "t_end": float(t),
        "state_end": state.tolist(),
        "masses_end": masses.tolist(),
        "order_end": [names[i] for i in o],
        "events": events,
        "max_constraint_residual": float(worst),
        "segments": [],
    }


def warmup():
    """Compile (or load from cache) all kernels once, e.g. before starting a process pool."""
    from collections import namedtuple

    S = namedtuple("S", "name m w")
    m = 0.9 * 20 ** (-0.4) * 0.03
    sh = [S("A", m, 0.2), S("B", m, 0.2)]
    R = 0.7894196446
    mid = 0.0186390612
    tot = 0.03
    va = float(np.sqrt(-_pot(R, 0.0, mid, m, 0.2)))
    vb = -float(np.sqrt(-_pot(R, mid, tot, m, 0.2)))
    simulate(sh, [0.0, mid, tot], [R, va, R, vb], tmax=3.0)
