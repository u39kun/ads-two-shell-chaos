"""Reconstruction of BCR's bifurcation diagram (their Fig. 7) with their own sampling rule.

BCR (PRD 94, 024003, Sec. IV) plot, for type (ii) data at each delta, "the points where the orbit of a solution ...
intersects the line R_1/l = 2 in the (R_1, dR_1/dt)-plane", R_1 being the outermost radius and t the static time of
the region between the shells. The direction of crossing is not stated; we keep both (dR_1/dt > 0 and < 0).

Type (ii) data as in bcr_delta_scan.py (start D: inner shell contracting). Each run is integrated with the compiled
solver to tmax; between consecutive crossings the outer shell moves in its own potential with fixed region masses, so
its passages through R = 2 are counted by integrating that one-shell motion over each flight (RK4, step <= 0.005, in
the same clock). At a passage the energy equation fixes the speed:
    dR_1/dt = +- f(2; M_mid) sqrt(-V(2)) / sqrt(f(2; M_mid) - V(2)),   V = V(2; M_mid, delta),
so each sampled value is a function of the middle mass, i.e. of the outer shell's energy, during that flight.
BCR's grid of delta (0.001 to 0.034) is the default. Runs to t = 1000 (--tmax 1000) give the sparser columns of BCR's
figure; the longer default runs fill them in.
usage: python bcr_fig7.py OUT.json [--tmax 20000] [--deltas a,b,...] [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit
from multiprocessing import Pool
import bcr_delta_scan as DS
from common import N

RPROBE = 2.0


@njit(cache=True)
def _dpot(R, lo, hi, m, w):
    u = m * R ** (-2.0 * w)
    return 2.0 * R + (hi + lo) / R**2 - 4.0 * w * (hi - lo) ** 2 / (R * u**2) + (2.0 + 4.0 * w) * u**2 / (4.0 * R**3)


@njit(cache=True)
def _deriv(R, v, M, Mt, m, w):
    fm = 1.0 - 2.0 * M / R + R * R
    phi = fm / np.sqrt(v * v + fm)
    return v * phi, -0.5 * _dpot(R, M, Mt, m, w) * phi


@njit(cache=True)
def _passages(R0, v0, M, Mt, m, w, T, Rs, dtmax):
    """Directions (+1 out, -1 in) of the outer shell's passages through Rs during a flight of duration T."""
    n = max(1, int(np.ceil(T / dtmax)))
    h = T / n
    R = R0
    v = v0
    out = []
    for i in range(n):
        k1R, k1v = _deriv(R, v, M, Mt, m, w)
        k2R, k2v = _deriv(R + 0.5 * h * k1R, v + 0.5 * h * k1v, M, Mt, m, w)
        k3R, k3v = _deriv(R + 0.5 * h * k2R, v + 0.5 * h * k2v, M, Mt, m, w)
        k4R, k4v = _deriv(R + h * k3R, v + h * k3v, M, Mt, m, w)
        Rn = R + h * (k1R + 2 * k2R + 2 * k3R + k4R) / 6.0
        vn = v + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        if (R - Rs) * (Rn - Rs) < 0.0:
            out.append(1 if Rn > R else -1)
        R = Rn
        v = vn
    return out, R


def speed_at(M, delta, shell, R=RPROBE):
    V = float(N.potential(R, M, delta, shell))
    fm = float(N.f(R, M))
    return fm * np.sqrt(-V) / np.sqrt(fm - V) if V < 0 else np.nan


def one(args):
    delta, tmax = args
    t0 = time.time()
    A, B = DS.shells(delta)
    ri = DS.RI
    sa = np.sqrt(-N.potential(ri, 0.0, delta / 2, A))
    sb = np.sqrt(-N.potential(ri, delta / 2, delta, B))
    masses, state, order = [0.0, delta / 2, delta], [ri, -sa, ri, sb], [0, 1]  # start D
    r = N.simulate(
        [A, B],
        masses,
        state,
        order=order,
        tmax=tmax,
        anchor=1,
        rtol=1e-12,
        atol=1e-14 * delta / 0.03,
        max_step=0.02,
        max_events=2_000_000,
    )
    names = ["A", "B"]
    # flights: (t_start, state, masses, outer shell index); the last one runs to t_end
    flights = [(0.0, np.array(state), list(masses), order[1])]
    for e in r["events"]:
        flights.append((e["t"], np.array(e["state_after"]), e["masses_after"], names.index(e["order_after"][1])))
    ends = [f[0] for f in flights[1:]] + [r["t_end"]]
    nxt = [e["r"] for e in r["events"]] + [None]  # radius at the end of each flight (the next crossing)
    vals, mids, times, maxdrift = [], [], [], 0.0
    for (ts, st, ms, io), te, rn in zip(flights, ends, nxt):
        R0, v0 = st[2 * io], st[2 * io + 1]
        dirs, Rend = _passages(R0, v0, ms[1], ms[2], A.m, A.w, te - ts, RPROBE, 0.005)
        if rn is not None:  # check: the one-shell flight ends at the next crossing
            maxdrift = max(maxdrift, abs(Rend - rn))
        if len(dirs):
            sp = speed_at(ms[1], delta, A)
            for d in dirs:
                vals.append(d * sp)
                mids.append(ms[1])
                times.append(ts)
    res = dict(
        delta=delta,
        status=r["status"],
        t_end=r["t_end"],
        crossings=len(r["events"]),
        values=vals,
        M_mid=mids,
        t_flight=times,
        flight_end_mismatch=maxdrift,
        seconds=time.time() - t0,
    )
    print(
        f"delta {delta:.4f}: {r['status']}, {len(r['events'])} crossings, {len(vals)} passages of R1 = 2, "
        f"dR1/dt in [{min(vals) if vals else None}, {max(vals) if vals else None}], flight check {maxdrift:.1e}, {res['seconds']:.0f}s",
        flush=True,
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=20000.0)
    # BCR's grid: delta = 0.001, 0.002, ..., 0.034
    p.add_argument("--deltas", default=",".join(f"{0.001 * i:.3f}" for i in range(1, 35)))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    deltas = [float(x) for x in a.deltas.split(",")]
    with Pool(a.workers) as pool:
        res = pool.map(one, [(d, a.tmax) for d in deltas], chunksize=1)
    json.dump(
        dict(params=dict(R_probe=RPROBE, tmax=a.tmax, start="D", clock="static time between the shells"), runs=res),
        open(a.out, "w"),
    )
    print("done")


if __name__ == "__main__":
    main()
