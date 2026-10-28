"""Chart conditioning of the tangent rates near the section's ends, and the periods at the fixed points.

The rates of Table 1 measure the tangent in the chart (R, 3 M_mid/delta), which degenerates at the folds of the
section (turning-point crossings), where the section's ends lie. With renormalization every return, the accumulated
log growth over a block equals the log growth of the block's product of DF, so a change of norm alters it only through
the norm at the block's two ends; it cannot accumulate along the block. To show this, each row of Table 1 is rerun
(bcr_survey_checks.ROWS, default setting, T = 2e4) with a second tangent vector normalized in the embedding of the
section in (R, Rdot_A, Rdot_B, 3 M_mid/delta), where Rdot_i = dR_i/dtau_i are fixed by (R, M_mid) through the junction
conditions. The section is a smooth surface in these coordinates up to its folds, so this norm stays regular there.
Stored per row: block rates in both norms, the largest ratio of the two norms along the orbit, and its values at the
block ends. Also stored: the return time, multipliers, and linearized rotation number at the fixed points of BCR's
Fig. 5 parameters (SECTION, bcr_section.py).
usage: python bcr_chart_check.py SECTION.json OUT.json [--workers N]"""

import os, sys, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import tangent_map as TM
import bcr_delta_scan as DS
import bcr_survey_checks as SCK
from common import N

H = 1e-30  # complex step


def embedding(delta, m, x):
    """J = d(R, Rdot_A, Rdot_B, 3 M/delta)/d(R, M) at section point x, by complex step."""
    S = N.Shell("A", m, 0.2)
    rdot = lambda R, M: (np.sqrt(-N.potential(R, 0.0, M, S)), -np.sqrt(-N.potential(R, M, delta, S)))
    J = np.zeros((4, 2))
    J[0, 0], J[3, 1] = 1.0, 3.0 / delta
    for c in range(2):
        R = complex(x[0], H if c == 0 else 0.0)
        M = complex(x[1], H if c == 1 else 0.0)
        a, b = rdot(R, M)
        J[1, c], J[2, c] = a.imag / H, b.imag / H
    return J


def row(job):
    delta, R, md, label = job
    t0 = time.time()
    rtol, ms = SCK.SETTINGS["default"]
    m = DS.MFAC * delta
    P = TM.TwoShellMap(
        m=m, w=0.2, Min=0.0, Mt=delta, rtol=rtol, atol=rtol * 1e-2 * delta / 0.03, max_step=ms, tmax=5000.0
    )
    sc = np.array([1.0, 3.0 / delta])
    x = np.array([R, md * delta])
    v = np.array([1.0, 1.0]) / sc
    v /= np.linalg.norm(v * sc)
    u = v / np.linalg.norm(embedding(delta, m, x) @ v)
    Sc, Se, clock = 0.0, 0.0, 0.0
    rows = [(0.0, 0.0, 0.0, float(np.log(np.linalg.norm(embedding(delta, m, x) @ v) / np.linalg.norm(v * sc))))]
    while clock < SCK.T:
        st, x1, DF, tr, nc = TM._return(x, P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            return dict(label=label, delta=delta, status=f"tangent status {st}")
        J = embedding(delta, m, x1)
        v = DF @ v
        g = np.linalg.norm(v * sc)
        Sc += np.log(g)
        v /= g
        u = DF @ u
        ge = np.linalg.norm(J @ u)
        Se += np.log(ge)
        u /= ge
        clock += tr
        # log of (embedded norm / chart norm) for the chart-normalized vector: the chart's distortion here
        rows.append((clock, Sc, Se, float(np.log(np.linalg.norm(J @ v)))))
        x = x1
    c = np.array(rows)
    edges = [int(np.searchsorted(c[:, 0], b)) for b in np.linspace(0, SCK.T, SCK.NB + 1)]
    edges[-1] = len(c) - 1
    blocks = []
    for i, j in zip(edges[:-1], edges[1:]):
        dt = c[j, 0] - c[i, 0]
        blocks.append(
            dict(
                rate_chart=(c[j, 1] - c[i, 1]) / dt,
                rate_embedded=(c[j, 2] - c[i, 2]) / dt,
                end_distortion=[c[i, 3], c[j, 3]],
            )
        )
    return dict(
        label=label,
        delta=delta,
        x0=[R, md * delta],
        status="ok",
        returns=len(c) - 1,
        blocks=blocks,
        max_abs_distortion=float(np.max(np.abs(c[:, 3]))),
        max_rate_difference=float(max(abs(b["rate_chart"] - b["rate_embedded"]) for b in blocks)),
        seconds=time.time() - t0,
    )


def fixed_points(section):
    S = json.load(open(section))
    p = S["params"]
    P = TM.TwoShellMap(m=p["m"], w=p["w"], Min=p["masses"][0], Mt=p["masses"][2])
    out = []
    for f in S["fixed_points"]:
        st, x1, DF, tr, nc = TM._return(
            np.array(f["x"]), P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax
        )
        ev = np.linalg.eigvals(DF)
        d = dict(kind=f["kind"], x=f["x"], return_time=float(tr), crossings=int(nc))
        if f["kind"] == "elliptic":
            rho = float(abs(np.angle(ev[0])) / (2 * np.pi))
            d.update(rotation_number=rho, period_returns=1 / rho, period_time=float(tr) / rho)
        else:
            d.update(multipliers=sorted(float(abs(e)) for e in ev.real))
        out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("section")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(row, SCK.ROWS, chunksize=1)
    fps = fixed_points(a.section)
    for r in res:
        if r["status"] != "ok":
            print(r["label"], r["delta"], r["status"])
            continue
        print(
            f"delta {r['delta']:g} {r['label']:28s} chart "
            + " ".join(f"{b['rate_chart']:.5f}" for b in r["blocks"])
            + " | embedded "
            + " ".join(f"{b['rate_embedded']:.5f}" for b in r["blocks"])
            + f" | max rate difference {r['max_rate_difference']:.1e}, max |log distortion| {r['max_abs_distortion']:.1f}"
        )
    for f in fps:
        print(f)
    json.dump(dict(rows=res, fixed_points=fps, seconds=time.time() - t0), open(a.out, "w"), default=float)
    print(f"wrote {a.out}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
