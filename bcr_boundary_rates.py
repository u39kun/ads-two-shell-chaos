"""Rates per unit boundary time of the saddle-layer and section-end orbits (Secs. 2 and 4.3 of the paper).

The rates of Secs. 4.2-4.3 are per unit middle-region time; boundary time is the static time of the exterior, whose
mass never changes. Each orbit is rerun with exactly the calls of its production script, so its tangent growth S is
the stored one:
  cuts   the strongly chaotic starts along the cuts (TANGENT_DELTA, bcr_layer_tangent.py delta: late-block exponent
         >= 1e-2 per return), 6000 returns, late block = returns 2000-6000;
  table  the rows of Table 1 (bcr_survey_checks.ROWS), default setting, 2e4 middle-region time units, four
         5000-unit blocks and [T/4, T].
Each return k is also integrated from its section point x_k with the event-located solver in both clocks (anchor 2
and anchor 1), each stopped after the return's crossings; the ratio of the two times scales the tangent map's return
time to boundary time. Restarting at every x_k avoids following a second, diverging trajectory. On long chaotic
returns the event solver's last crossing can differ in type from the tangent map's (the integrations separate within
the return); such returns are counted, and the ratio of clocks is insensitive to them.
Stored per orbit and block: growth of S, durations in both clocks, rates in both clocks, and their ratio.
usage: python bcr_boundary_rates.py TANGENT_DELTA.json OUT.json [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
import bcr_delta_scan as DS
import bcr_survey_checks as SCK
from common import N


def boundary_return(delta, m, x, nc, tr):
    """Boundary-clock duration of the return that starts at section point x and lasts tr in the middle clock;
    also the event solver's middle-clock duration and whether its last crossing is head-on with A inside."""
    A, B = N.Shell("A", m, 0.2), N.Shell("B", m, 0.2)
    R, M = x
    va, vb = np.sqrt(-N.potential(R, 0.0, M, A)), -np.sqrt(-N.potential(R, M, delta, B))
    mid, na, nb, _, _ = N.crossing(R, va, vb, 0.0, M, delta, A, B)
    t, mismatch = [], False
    for anchor in (2, 1):
        s = N.simulate(
            [A, B],
            [0.0, mid, delta],
            [R, na, R, nb],
            order=[1, 0],
            tmax=3.0 * tr + 50.0,
            anchor=anchor,
            rtol=1e-12,
            atol=1e-14 * delta / 0.03,
            max_step=0.02,
            max_events=nc,
        )
        e = s["events"]
        if len(e) != nc:
            raise RuntimeError(f"return not reproduced: {len(e)} events of {nc}, status {s['status']}")
        mismatch |= not (e[-1]["loser"] == "A" and e[-1]["v_before"][0] > 0 > e[-1]["v_before"][1])
        t.append(e[-1]["t"])
    return tr * t[0] / t[1], t[1], mismatch


def orbit(job):
    t0 = time.time()
    P = TM.TwoShellMap(
        m=job["m"],
        w=0.2,
        Min=0.0,
        Mt=job["delta"],
        rtol=job["rtol"],
        atol=1e-2 * job["rtol"] * job["delta"] / 0.03,
        max_step=job["max_step"],
        tmax=5000.0,
    )
    sc = np.array(job["scale"])
    x = np.array(job["x0"], float)
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    S, tm, tb, k, worst, nmis = 0.0, 0.0, 0.0, 0, 0.0, 0
    rows = [(0.0, 0.0, 0.0)]  # (S, middle time, boundary time) after each return
    S10 = []
    while k < job["returns"] and tm < job["T"]:
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            return dict(job=job, status=f"tangent status {st}", returns=k)
        dtb, tm1, mis = boundary_return(job["delta"], job["m"], x, int(nc), float(tr))
        nmis += mis
        if not mis:
            worst = max(worst, abs(tm1 - tr) / tr)
        v = DF @ v
        g = np.linalg.norm(v * sc)
        S += np.log(g)
        v /= g
        tm += tr
        tb += dtb
        k += 1
        rows.append((S, tm, tb))
        if k % 10 == 0:
            S10.append(S)
        x = x1
    c = np.array(rows)

    def block(i, j):
        dS, dm, db = c[j] - c[i]
        return dict(dS=dS, dt_mid=dm, dt_bdry=db, rate_mid=dS / dm, rate_bdry=dS / db, bdry_per_mid=db / dm)

    out = dict(job=job, status="ok", returns=k, middle_clock_check=worst, mismatched_returns=nmis)
    if job["kind"] == "cuts":
        out["late"] = block(2000, 6000)
        out["cumulative_log"] = S10
    else:
        idx = [int(np.searchsorted(c[:, 1], b)) for b in np.linspace(0, job["T"], 5)]
        idx[-1] = len(c) - 1
        out["blocks"] = [block(i, j) for i, j in zip(idx[:-1], idx[1:])]
        out["T4_T"] = block(idx[1], idx[-1])
    out["seconds"] = time.time() - t0
    return out


def jobs(tangent_delta):
    J = []
    for r in json.load(open(tangent_delta))["runs"]:
        c = r["cumulative_log"]
        if r["returns_done"] >= 6000 and (c[599] - c[199]) / 4000 >= 1e-2:
            cs = r["case"]
            J.append(
                dict(
                    kind="cuts",
                    label=cs["label"],
                    delta=cs["Mt"],
                    m=cs["m"],
                    x0=cs["x0"],
                    scale=cs["scale"],
                    rtol=cs.get("rtol", 1e-12),
                    max_step=cs.get("max_step", 0.02),
                    returns=cs["returns"],
                    T=float("inf"),
                )
            )
    rtol, ms = SCK.SETTINGS["default"]
    for d, R, md, lab in SCK.ROWS:
        J.append(
            dict(
                kind="table",
                label=f"delta {d:g} {lab}",
                delta=d,
                m=DS.MFAC * d,
                x0=[R, md * d],
                scale=[1.0, 3.0 / d],
                rtol=rtol,
                max_step=ms,
                returns=10**9,
                T=SCK.T,
            )
        )
    return J


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tangent_delta")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    J = jobs(a.tangent_delta)
    with Pool(a.workers) as pool:
        res = pool.map(orbit, J, chunksize=1)
    stored = {r["case"]["label"]: r["cumulative_log"] for r in json.load(open(a.tangent_delta))["runs"]}
    for r in res:
        if r["job"]["kind"] == "cuts" and r["status"] == "ok":
            s = np.array(stored[r["job"]["label"]][: len(r["cumulative_log"])])
            r["max_deviation_from_stored"] = float(np.max(np.abs(s - np.array(r.pop("cumulative_log")))))
    for r in res:
        if r["status"] != "ok":
            print(r["job"]["label"], r["status"])
            continue
        b = r["late"] if r["job"]["kind"] == "cuts" else r["T4_T"]
        dev = (
            f", deviation from stored S {r['max_deviation_from_stored']:.1e}"
            if "max_deviation_from_stored" in r
            else ""
        )
        print(
            f"{r['job']['label']:34s} rate {b['rate_mid']:.4f} (middle) {b['rate_bdry']:.4f} (boundary), "
            f"dt_bdry/dt_mid {b['bdry_per_mid']:.4f}; mismatched returns {r['mismatched_returns']}/{r['returns']}"
            f"{dev}"
        )
    json.dump(dict(runs=res, seconds=time.time() - t0), open(a.out, "w"), default=float)
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
