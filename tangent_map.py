"""Return map of the two-shell section together with its derivative from the variational equations
(the 'variational tangent': variational equations + saltation at crossings).

Section and chart are those of bcr_section.py and two_shell_measure.py: states just before a head-on crossing with
shell A inside (A moving out, B moving in), coordinates x = (R, M_mid). The map F follows the pair through that
crossing, the flights, and any further crossings, to the next such state. Here DF(x) is computed along with F(x):

  * flights: DOP853 in the static time of the middle region (anchor 1), with the step control of fastshells.py.
    The variational equations are integrated with the same Runge-Kutta stages (internal differentiation), so the
    tangent is the derivative of the computed flow for the step sequence taken. The Jacobian of the right-hand side
    is evaluated by complex-step differentiation, which is accurate to rounding level (no subtraction, no step-size
    choice);
  * crossings: the crossing update of Eq. (3) is differentiated by complex step, and the dependence of the crossing
    time on the data enters through the saltation matrix
        T+ = DG (T- + f- dt) - f+ dt,    dt = -(grad h . T-)/(grad h . f-),    h = R_inner - R_outer;
  * the section: the same time correction projects the tangent onto the crossing surface.

Nothing here uses finite differences of F. Checks (bcr_jacobian_check.py): agreement with central differences of F
away from chart folds, and det DF(x) = g(x)/g(F(x)), the invariant-measure identity derived in App. A.
"""

import cmath
import numpy as np
from numba import njit
from scipy.integrate._ivp import dop853_coefficients as _co

_A = np.ascontiguousarray(_co.A[:12, :12])
_B = np.ascontiguousarray(_co.B)
_E3 = np.ascontiguousarray(_co.E3)
_E5 = np.ascontiguousarray(_co.E5)
_HCS = 1e-30  # complex-step size


# ----------------------------------------------------------------------------------------------- shell physics
@njit(cache=True)
def _dpot(R, lo, hi, m, w):
    u = m * R ** (-2.0 * w)
    return 2.0 * R + (hi + lo) / R**2 - 4.0 * w * (hi - lo) ** 2 / (R * u**2) + (2.0 + 4.0 * w) * u**2 / (4.0 * R**3)


@njit(cache=True)
def _mu_c(m, w, R):
    return m * cmath.exp(-2.0 * w * cmath.log(R))


@njit(cache=True)
def _pot_c(R, lo, hi, m, w):
    u = _mu_c(m, w, R)
    return 1.0 + R * R - (hi + lo) / R - (hi - lo) ** 2 / u**2 - u**2 / (4.0 * R * R)


@njit(cache=True)
def _dpot_c(R, lo, hi, m, w):
    u = _mu_c(m, w, R)
    return 2.0 * R + (hi + lo) / R**2 - 4.0 * w * (hi - lo) ** 2 / (R * u**2) + (2.0 + 4.0 * w) * u**2 / (4.0 * R**3)


@njit(cache=True)
def _rhs(y, M, order, m, w, Min, Mt, out):
    """Two shells, clock = static time of the middle region: dR/dt = v phi, dv/dt = -V'(R)/2 phi,
    phi = dtau/dt = f(R; M)/sqrt(v^2 + f(R; M)) for both shells (both border the middle region)."""
    for rank in range(2):
        i = order[rank]
        R = y[2 * i]
        v = y[2 * i + 1]
        if not (R > 0.0):
            return False
        fm = 1.0 - 2.0 * M / R + R * R
        if fm <= 0.0:
            return False
        lo = Min if rank == 0 else M
        hi = M if rank == 0 else Mt
        phi = fm / np.sqrt(v * v + fm)
        out[2 * i] = v * phi
        out[2 * i + 1] = -0.5 * _dpot(R, lo, hi, m[i], w[i]) * phi
    return True


@njit(cache=True)
def _rhs_c(y, M, order, m, w, Min, Mt, out):
    for rank in range(2):
        i = order[rank]
        R = y[2 * i]
        v = y[2 * i + 1]
        fm = 1.0 - 2.0 * M / R + R * R
        if rank == 0:
            lo = complex(Min, 0.0)
            hi = M
        else:
            lo = M
            hi = complex(Mt, 0.0)
        phi = fm / cmath.sqrt(v * v + fm)
        out[2 * i] = v * phi
        out[2 * i + 1] = -0.5 * _dpot_c(R, lo, hi, m[i], w[i]) * phi


@njit(cache=True)
def _jac_rhs(y, M, order, m, w, Min, Mt, J, yc, oc):
    """J (4 x 5) = d rhs / d(y, M) by complex step."""
    for j in range(5):
        for k in range(4):
            yc[k] = complex(y[k], 0.0)
        Mc = complex(M, 0.0)
        if j < 4:
            yc[j] = complex(y[j], _HCS)
        else:
            Mc = complex(M, _HCS)
        _rhs_c(yc, Mc, order, m, w, Min, Mt, oc)
        for k in range(4):
            J[k, j] = oc[k].imag / _HCS


@njit(cache=True)
def _cross_c(y, M, ia, ib, m, w, Min, Mt, yo):
    """Transparent crossing of the inner shell ia and the outer shell ib (Eq. (3)); complex-safe. Returns M'."""
    R = 0.5 * (y[2 * ia] + y[2 * ib])
    va = y[2 * ia + 1]
    vb = y[2 * ib + 1]
    ma = _mu_c(m[ia], w[ia], R)
    mb = _mu_c(m[ib], w[ib], R)
    fm = 1.0 - 2.0 * M / R + R * R
    ba = cmath.sqrt(va * va + fm)
    bb = cmath.sqrt(vb * vb + fm)
    gam = (ba * bb - va * vb) / fm
    rs = (bb * va - ba * vb) / fm
    yo[2 * ia] = R
    yo[2 * ib] = R
    yo[2 * ia + 1] = va - mb * rs / R
    yo[2 * ib + 1] = vb - ma * rs / R
    return Min + Mt - M + ma * mb * gam / R


@njit(cache=True)
def _embed_cross_c(R, M, m, w, Min, Mt, yo):
    """Section point x = (R, M) -> state just after its crossing (A = shell 0 inner, moving out; B = 1 outer, in)."""
    y = np.empty(4, np.complex128)
    y[0] = R
    y[2] = R
    y[1] = cmath.sqrt(-_pot_c(R, complex(Min, 0.0), M, m[0], w[0]))
    y[3] = -cmath.sqrt(-_pot_c(R, M, complex(Mt, 0.0), m[1], w[1]))
    return _cross_c(y, M, 0, 1, m, w, Min, Mt, yo)


# ----------------------------------------------------------------------------------------------- integrator
@njit(cache=True)
def _step(y, M, f0, h, order, m, w, Min, Mt, rtol, atol, K, Ys, ynew, fnew, tmp):
    """One DOP853 step (as fastshells._step), keeping the stage states Ys. Returns (ok, error_norm)."""
    for k in range(4):
        K[0, k] = f0[k]
        Ys[0, k] = y[k]
    for s in range(1, 12):
        for k in range(4):
            acc = 0.0
            for j in range(s):
                acc += _A[s, j] * K[j, k]
            tmp[k] = y[k] + h * acc
            Ys[s, k] = tmp[k]
        if not _rhs(tmp, M, order, m, w, Min, Mt, K[s]):
            return False, 0.0
    for k in range(4):
        acc = 0.0
        for j in range(12):
            acc += _B[j] * K[j, k]
        ynew[k] = y[k] + h * acc
    if not _rhs(ynew, M, order, m, w, Min, Mt, fnew):
        return False, 0.0
    for k in range(4):
        K[12, k] = fnew[k]
    e5 = 0.0
    e3 = 0.0
    for k in range(4):
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
    return True, abs(h) * e5 / np.sqrt((e5 + 0.01 * e3) * 4)


@njit(cache=True)
def _tangent_step(Ys, M, T, h, order, m, w, Min, Mt, KT, TY, J, yc, oc):
    """Variational equations over the step whose stage states are Ys (same RK coefficients). T is 5 x nc."""
    nc = T.shape[1]
    for s in range(12):
        for k in range(4):
            for c in range(nc):
                acc = 0.0
                for j in range(s):
                    acc += _A[s, j] * KT[j, k, c]
                TY[k, c] = T[k, c] + h * acc
        for c in range(nc):
            TY[4, c] = T[4, c]
        _jac_rhs(Ys[s], M, order, m, w, Min, Mt, J, yc, oc)
        for k in range(4):
            for c in range(nc):
                acc = 0.0
                for j in range(5):
                    acc += J[k, j] * TY[j, c]
                KT[s, k, c] = acc
    for k in range(4):
        for c in range(nc):
            acc = 0.0
            for j in range(12):
                acc += _B[j] * KT[j, k, c]
            T[k, c] += h * acc


@njit(cache=True)
def _return(x0, m, w, Min, Mt, rtol, atol, max_step, want_tangent, tmax):
    """One application of the return map. Returns (status, x1, DF, flight_time, crossings).
    status: 0 ok, 1 forbidden start, 2 integrator failure, 3 no return before tmax."""
    x1 = np.zeros(2)
    DF = np.zeros((2, 2))
    # start: section point -> state just after the crossing, with its derivative (complex step)
    yo = np.empty(4, np.complex128)
    pa = _pot_c(complex(x0[0], 0.0), complex(Min, 0.0), complex(x0[1], 0.0), m[0], w[0]).real
    pb = _pot_c(complex(x0[0], 0.0), complex(x0[1], 0.0), complex(Mt, 0.0), m[1], w[1]).real
    if pa >= 0.0 or pb >= 0.0:
        return 1, x1, DF, 0.0, 0
    y = np.empty(4)
    T = np.zeros((5, 2))
    Mc = _embed_cross_c(complex(x0[0], 0.0), complex(x0[1], 0.0), m, w, Min, Mt, yo)
    M = Mc.real
    for k in range(4):
        y[k] = yo[k].real
    if want_tangent:
        for c in range(2):
            Rc = complex(x0[0], _HCS if c == 0 else 0.0)
            Mi = complex(x0[1], _HCS if c == 1 else 0.0)
            Mo = _embed_cross_c(Rc, Mi, m, w, Min, Mt, yo)
            for k in range(4):
                T[k, c] = yo[k].imag / _HCS
            T[4, c] = Mo.imag / _HCS
    order = np.array([1, 0], np.int64)  # B inside after the first crossing
    K = np.empty((13, 4))
    Ys = np.empty((12, 4))
    Kr = np.empty((13, 4))
    Ysr = np.empty((12, 4))
    ynew = np.empty(4)
    fnew = np.empty(4)
    tmp = np.empty(4)
    f0 = np.empty(4)
    yr = np.empty(4)
    fr = np.empty(4)
    KT = np.empty((12, 4, 2))
    TY = np.empty((5, 2))
    J = np.empty((4, 5))
    yc = np.empty(4, np.complex128)
    oc = np.empty(4, np.complex128)
    Fm = np.zeros(5)
    Fp = np.zeros(5)
    DG = np.zeros((5, 5))
    Tt = np.zeros((5, 2))
    dt = np.zeros(2)
    t = 0.0
    h = -1.0
    ncross = 1
    while t < tmax:
        if not _rhs(y, M, order, m, w, Min, Mt, f0):
            return 2, x1, DF, t, ncross
        if h <= 0.0:  # initial step as in fastshells / scipy
            d0 = 0.0
            d1 = 0.0
            for k in range(4):
                sc = atol + abs(y[k]) * rtol
                d0 += (y[k] / sc) ** 2
                d1 += (f0[k] / sc) ** 2
            d0 = np.sqrt(d0 / 4)
            d1 = np.sqrt(d1 / 4)
            h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
            h0 = min(h0, tmax - t)
            for k in range(4):
                yr[k] = y[k] + h0 * f0[k]
            if _rhs(yr, M, order, m, w, Min, Mt, fr):
                d2 = 0.0
                for k in range(4):
                    sc = atol + abs(y[k]) * rtol
                    d2 += ((fr[k] - f0[k]) / sc) ** 2
                d2 = np.sqrt(d2 / 4) / h0
                if d1 <= 1e-15 and d2 <= 1e-15:
                    h1 = max(1e-6, h0 * 1e-3)
                else:
                    h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
                h = min(100.0 * h0, h1, max_step)
            else:
                h = min(h0, max_step)
        rejected = False
        while True:
            h = min(h, max_step, tmax - t)
            if h < 1e-14:
                return 2, x1, DF, t, ncross
            ok, err = _step(y, M, f0, h, order, m, w, Min, Mt, rtol, atol, K, Ys, ynew, fnew, tmp)
            if not ok:
                h *= 0.5
                rejected = True
                continue
            if err < 1.0:
                fac = 10.0 if err == 0.0 else min(10.0, 0.9 * err ** (-1.0 / 8.0))
                if rejected:
                    fac = min(1.0, fac)
                hstep = h
                h = h * fac
                break
            h *= max(0.2, 0.9 * err ** (-1.0 / 8.0))
            rejected = True
        ia = order[0]
        ib = order[1]
        g0 = y[2 * ia] - y[2 * ib]
        g1 = ynew[2 * ia] - ynew[2 * ib]
        if not (g0 <= 0.0 and g1 >= 0.0):
            if want_tangent:
                _tangent_step(Ys, M, T, hstep, order, m, w, Min, Mt, KT, TY, J, yc, oc)
            for k in range(4):
                y[k] = ynew[k]
            t += hstep
            continue
        # locate the crossing on the single-step map (Illinois regula falsi, as fastshells)
        a = 0.0
        b = hstep
        ga = g0
        gb = g1
        side = 0
        for it in range(200):
            c = 0.5 * (a + b) if gb == ga else b - gb * (b - a) / (gb - ga)
            if not (c > a and c < b):
                c = 0.5 * (a + b)
            okr, _ = _step(y, M, f0, c, order, m, w, Min, Mt, rtol, atol, Kr, Ysr, yr, fr, tmp)
            if not okr:
                b = c
                continue
            gc = yr[2 * ia] - yr[2 * ib]
            if gc >= 0.0:
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
        ok, _ = _step(y, M, f0, b, order, m, w, Min, Mt, rtol, atol, K, Ys, ynew, fnew, tmp)
        if want_tangent:
            _tangent_step(Ys, M, T, b, order, m, w, Min, Mt, KT, TY, J, yc, oc)
        for k in range(4):
            y[k] = ynew[k]
        t += b
        # time correction: dt = -(grad h . T)/(grad h . f-), T~ = T + f- dt
        _rhs(y, M, order, m, w, Min, Mt, fnew)
        for k in range(4):
            Fm[k] = fnew[k]
        Fm[4] = 0.0
        if want_tangent:
            den = Fm[2 * ia] - Fm[2 * ib]
            for c in range(2):
                dt[c] = -(T[2 * ia, c] - T[2 * ib, c]) / den
                for k in range(5):
                    Tt[k, c] = T[k, c] + Fm[k] * dt[c]
        if ia == 0 and y[1] > 0.0 > y[3]:  # the section: A inside, head-on
            x1[0] = 0.5 * (y[0] + y[2])
            x1[1] = M
            if want_tangent:
                for c in range(2):
                    DF[0, c] = 0.5 * (Tt[0, c] + Tt[2, c])
                    DF[1, c] = Tt[4, c]
            return 0, x1, DF, t, ncross
        # an off-section crossing: update the state and the tangent (saltation)
        if want_tangent:
            for j in range(5):
                for k in range(4):
                    yc[k] = complex(y[k], 0.0)
                Mc = complex(M, 0.0)
                if j < 4:
                    yc[j] = complex(y[j], _HCS)
                else:
                    Mc = complex(M, _HCS)
                Mo = _cross_c(yc, Mc, ia, ib, m, w, Min, Mt, yo)
                for k in range(4):
                    DG[k, j] = yo[k].imag / _HCS
                DG[4, j] = Mo.imag / _HCS
        for k in range(4):
            yc[k] = complex(y[k], 0.0)
        Mo = _cross_c(yc, complex(M, 0.0), ia, ib, m, w, Min, Mt, yo)
        for k in range(4):
            y[k] = yo[k].real
        M = Mo.real
        order[0] = ib
        order[1] = ia
        ncross += 1
        if want_tangent:
            _rhs(y, M, order, m, w, Min, Mt, fnew)
            for k in range(4):
                Fp[k] = fnew[k]
            Fp[4] = 0.0
            for c in range(2):
                for k in range(5):
                    acc = 0.0
                    for j in range(5):
                        acc += DG[k, j] * Tt[j, c]
                    T[k, c] = acc - Fp[k] * dt[c]
    return 3, x1, DF, t, ncross


# ----------------------------------------------------------------------------------------------- Python interface
class TwoShellMap:
    """Section map of two shells (A, B) with rest-mass parameters m, pressure w, inner mass Min, and total mass Mt."""

    def __init__(self, m=0.00815, w=0.2, Min=0.0, Mt=0.03, rtol=1e-12, atol=None, max_step=0.02, tmax=200.0):
        self.m = np.array([m, m], float) if np.ndim(m) == 0 else np.asarray(m, float)
        self.w = np.array([w, w], float) if np.ndim(w) == 0 else np.asarray(w, float)
        self.Min, self.Mt = float(Min), float(Mt)
        self.rtol = float(rtol)
        self.atol = float(rtol * 1e-2 if atol is None else atol)
        self.max_step, self.tmax = float(max_step), float(tmax)

    def __call__(self, x, tangent=False):
        st, x1, DF, t, nc = _return(
            np.asarray(x, float),
            self.m,
            self.w,
            self.Min,
            self.Mt,
            self.rtol,
            self.atol,
            self.max_step,
            bool(tangent),
            self.tmax,
        )
        if st != 0:
            raise RuntimeError({1: "forbidden start", 2: "integrator failure", 3: "no return"}[st])
        return (x1, DF) if tangent else x1

    def info(self, x):
        st, x1, DF, t, nc = _return(
            np.asarray(x, float),
            self.m,
            self.w,
            self.Min,
            self.Mt,
            self.rtol,
            self.atol,
            self.max_step,
            False,
            self.tmax,
        )
        return dict(status=int(st), x=x1, time=float(t), crossings=int(nc))

    def g(self, x):
        """Density of the invariant measure, g = |1/X_B - 1/X_A| with X = dR/dt in the middle region."""
        R, M = x
        va = np.sqrt(-_pot_c(complex(R), complex(self.Min), complex(M), self.m[0], self.w[0]).real)
        vb = -np.sqrt(-_pot_c(complex(R), complex(M), complex(self.Mt), self.m[1], self.w[1]).real)
        fm = 1 - 2 * M / R + R * R
        XA, XB = va * fm / np.sqrt(va * va + fm), vb * fm / np.sqrt(vb * vb + fm)
        return abs(1 / XB - 1 / XA)

    def speeds(self, x):
        """|dR/dtau| of A and B at the section point (small values mean a crossing near a turning point)."""
        R, M = x
        va = np.sqrt(-_pot_c(complex(R), complex(self.Min), complex(M), self.m[0], self.w[0]).real)
        vb = np.sqrt(-_pot_c(complex(R), complex(M), complex(self.Mt), self.m[1], self.w[1]).real)
        return va, vb
