"""Nudged-initial-condition test of BCR's Fig. 6 (Brito, Cardoso & Rocha, PRD 94, 024003; arXiv:1602.03535).

BCR Fig. 5/6 parameters: two shells with m = 0.00815 (as printed; the type (ii) formula 0.9*20^(-0.4)*0.03 gives
8.146e-3), w = 0.2, region masses 0 (inside), 0.015 (between), 0.03 (outside), both starting at R_i = 1, one
contracting and one expanding. BCR do not say which shell does which, so the assignment is a reconstruction:
  --start D (default, departing): inner shell A contracting, dR/dtau = -sqrt(-V); outer shell B expanding,
            dR/dtau = +sqrt(-V); the shells separate at t = 0.
  --start C (converging): A expanding, B contracting; they cross at t = 0, and the run starts just after that
            crossing (B inside, A outside, middle mass from Eq. (3.18) of Ida & Nakao).
V is each shell's effective potential with its own region masses. Fig. 6 compares two runs whose R_i differ by 5e-5
through the radius of the outer shell, and BCR quote a Lyapunov exponent of about 0.04.

Here: a reference run and nudged runs with R_i + eps (eps in --eps), in two clocks: anchor 1 = static time of the
region between the shells (BCR's evolution variable) and anchor 2 = boundary time (ours). Both shells' radii are
sampled on a uniform grid; we record |dR| for the shell that starts outside (B) and for the outermost radius, which is
the observable of BCR's Fig. 6 (it changes material shell at each crossing). Both get the same diagnostics.
Diagnostics: the running maximum (envelope) of |dR|, its log-log slope (1 for the linear phase drift of
quasi-periodic motion) and the time at which it first reaches given thresholds. For exponential growth at rate
lambda these threshold times shift by ln(10)/lambda per decade of eps; for linear growth they scale like 1/eps.
usage: python bcr_lyapunov.py OUT.json [--tmax 1000] [--eps 5e-5,5e-6,5e-7] [--dt 0.05] [--start D|C]"""

import sys, os, json, time, argparse

os.environ["SOLVER"] = "reference"  # the reference solver keeps dense output (collect=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from common import N

M, W, MASSES = 0.00815, 0.2, [0.0, 0.015, 0.03]
A, B = N.Shell("A", M, W), N.Shell("B", M, W)


def initial(ri, start="D"):
    """Masses, state [R_A, v_A, R_B, v_B] (v = dR/dtau), and shell order (inside out) at t = 0."""
    sa = np.sqrt(-N.potential(ri, MASSES[0], MASSES[1], A))
    sb = np.sqrt(-N.potential(ri, MASSES[1], MASSES[2], B))
    if start == "D":
        return list(MASSES), [ri, -sa, ri, sb], [0, 1]
    mid, va, vb, _, _ = N.crossing(ri, sa, -sb, MASSES[0], MASSES[1], MASSES[2], A, B)
    return [MASSES[0], mid, MASSES[2]], [ri, va, ri, vb], [1, 0]


def trajectory(args):
    ri, anchor, tmax, dt, start = args
    masses, y0, order = initial(ri, start)
    # The saved results use the reference tolerances below. A separate rerun
    # of --start D --tmax 2000 with rtol=1e-8 and atol=1e-10 gave the same
    # reported diagnostics to the precision quoted in the paper.
    r = N.simulate(
        [A, B],
        masses,
        y0,
        order=order,
        tmax=tmax,
        anchor=anchor,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=100000,
        collect=True,
    )
    t = np.arange(0.0, min(tmax, r["t_end"]) + 1e-12, dt)
    R = np.full((len(t), 2), np.nan)
    for seg in r["segments"]:
        lo, hi = seg["t"][0], seg["t"][-1]
        k = (t >= lo) & (t <= hi)
        if k.any():
            y = seg["dense"](t[k])
            R[k, 0], R[k, 1] = y[0], y[2]
    return dict(ri=ri, anchor=anchor, status=r["status"], t_end=r["t_end"], crossings=len(r["events"]), t=t, R=R)


def envelope(x):
    return np.maximum.accumulate(np.nan_to_num(x, nan=0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--tmax", type=float, default=1000.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--eps", default="5e-5,5e-6,5e-7")
    p.add_argument("--ri", type=float, default=1.0)
    p.add_argument("--start", choices=("D", "C"), default="D")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count())))
    a = p.parse_args()
    eps = [float(e) for e in a.eps.split(",")]
    tasks = [(a.ri + e, anc, a.tmax, a.dt, a.start) for anc in (1, 2) for e in [0.0] + eps]
    t0 = time.time()
    with Pool(min(a.workers, len(tasks))) as pool:
        runs = pool.map(trajectory, tasks)
    out = dict(params=dict(m=M, w=W, masses=MASSES, ri=a.ri, eps=eps, tmax=a.tmax, dt=a.dt, start=a.start), clocks={})
    for anc, name in ((1, "between_shells"), (2, "boundary")):
        ref = [r for r in runs if r["anchor"] == anc and r["ri"] == a.ri][0]
        print(f'clock {name}: reference run {ref["status"]}, t_end {ref["t_end"]:.1f}, {ref["crossings"]} crossings')
        rows = []
        for e in eps:
            run = [r for r in runs if r["anchor"] == anc and abs(r["ri"] - (a.ri + e)) < 1e-15][0]
            n = min(len(run["t"]), len(ref["t"]))
            t = ref["t"][:n]
            dB = np.abs(run["R"][:n, 1] - ref["R"][:n, 1])  # shell that starts outside
            dO = np.abs(run["R"][:n].max(1) - ref["R"][:n].max(1))  # outermost radius (BCR's Fig. 6)
            stats = {}
            for obs, dd in (("outer_shell", dB), ("outermost", dO)):
                env = envelope(dd)
                thr = {
                    f"{x:g}": (float(t[np.argmax(env >= x)]) if (env >= x).any() else None) for x in (1e-3, 1e-2, 1e-1)
                }
                # log-log slope of the envelope over the window where it is between 10 eps and 0.1 (pre-saturation)
                k = (t > 10) & (env > 10 * e) & (env < 0.1)
                slope = float(np.polyfit(np.log(t[k]), np.log(env[k]), 1)[0]) if k.sum() > 20 else None
                # BCR-style log-linear fit of the envelope over the same window
                lam = float(np.polyfit(t[k], np.log(env[k]), 1)[0]) if k.sum() > 20 else None
                # log-linear fits of the envelope over growing windows 0 < t <= T (full sampling)
                rates = {}
                for T in (50, 100, 200, 500, 2000):
                    kw = (t > 0) & (t <= T) & (env > 0)
                    rates[str(T)] = float(np.polyfit(t[kw], np.log(env[kw]), 1)[0]) if kw.sum() > 20 else None
                stats[obs] = dict(
                    env=env,
                    thresholds=thr,
                    loglog_slope=slope,
                    loglinear_rate=lam,
                    window_rates=rates,
                    fit_start=float(t[k][0]) if k.any() else None,
                )
                print(
                    f'  eps {e:.0e} [{obs}]: status {run["status"]}; envelope reaches 1e-3/1e-2/1e-1 at {thr}; '
                    f'log-log slope {slope} from t = {stats[obs]["fit_start"]}; window rates {rates}'
                )
            step = max(1, int(round(0.5 / a.dt)))
            so, sm = stats["outer_shell"], stats["outermost"]
            rows.append(
                dict(
                    eps=e,
                    status=run["status"],
                    thresholds=so["thresholds"],
                    loglog_slope=so["loglog_slope"],
                    loglinear_rate=so["loglinear_rate"],
                    window_rates=so["window_rates"],
                    fit_start=so["fit_start"],
                    thresholds_outermost=sm["thresholds"],
                    loglog_slope_outermost=sm["loglog_slope"],
                    loglinear_rate_outermost=sm["loglinear_rate"],
                    window_rates_outermost=sm["window_rates"],
                    fit_start_outermost=sm["fit_start"],
                    t=t[::step].tolist(),
                    dR_outer_shell=dB[::step].tolist(),
                    dR_outermost=dO[::step].tolist(),
                    envelope=so["env"][::step].tolist(),
                    envelope_outermost=sm["env"][::step].tolist(),
                )
            )
        out["clocks"][name] = dict(reference_status=ref["status"], reference_crossings=ref["crossings"], runs=rows)
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"))
    print(f'done in {out["seconds"]:.0f}s')


if __name__ == "__main__":
    main()
