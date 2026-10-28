"""Data for the trajectory explorer page: crossings of seven orbits of the two-shell map, each with a nudged twin,
computed with the paper's solver (fastshells via common.N, rtol 1e-12, max step 0.02, middle-region clock).
The two orbits of the paper's Fig. 3 (10^-4 and 0.07 from the saddle) and their twins take their section points from
the variational-tangent integrator exactly as bcr_layer_tangent.py does, so they follow the paper's trajectories; the
crossings within each return are filled in by a one-return run of the solver from that section point. The orbit near
the section's ends starts where the paper's Fig. 7 orbit does (bcr_section_overview.py), with the same solver settings;
the page also gets the edges of the section from that script's results.

For every run the page gets one record per flight (the stretch between two crossings):
    t_start, R_A, R_B, dR_A/dtau, dR_B/dtau, M_mid, A inside (1/0), starts with a section crossing (1/0), kind
where kind is the crossing that starts the flight, from the shells' velocities just before it: 1 head on (inner shell
moving out, outer moving in), 2 overtaking with both moving in, 3 overtaking with both moving out, 0 none (the start of
initial data). A section crossing is a head-on crossing with A inside. The page integrates each flight itself from that state, so it stores no dense trajectory. Separations between an orbit and
its twin are computed here at full precision, in the chart (R, 100 M_mid), section point by section point.
usage (from the repository root): python demo/gen_data.py . demo/orbits.json"""

import sys, os, json
import numpy as np

CODE = os.path.abspath(sys.argv[1])
sys.path.insert(0, CODE)
os.chdir(CODE)
from common import N  # noqa: E402
import tangent_map as TM  # noqa: E402

W = 0.2
EPS = 5e-5  # twin offset: in R at the section start, or in both initial radii for BCR's initial data
TMAX = 7000.0


def kind(e):
    vi, vo = e["v_before"]  # inner shell, outer shell
    if vi > 0 > vo:
        return 1
    if vi * vo <= 0:
        raise RuntimeError("Unexpected crossing branch")
    return 2 if vi < 0 else 3


def shells(m):
    return N.Shell("A", m, W), N.Shell("B", m, W)


def from_section(x0, m, Mout):
    A, B = shells(m)
    pa, pb = N.potential(x0[0], 0.0, x0[1], A), N.potential(x0[0], x0[1], Mout, B)
    nm, na, nb, _, _ = N.crossing(x0[0], np.sqrt(-pa), -np.sqrt(-pb), 0.0, x0[1], Mout, A, B)
    return dict(masses=[0.0, nm, Mout], state=[x0[0], na, x0[0], nb], order=[1, 0], first=x0)


def from_initial(R, m, Mmid, Mout):
    """Both shells at R, inner shell (A) contracting, outer (B) expanding (BCR's data, our first assignment)."""
    A, B = shells(m)
    sa = np.sqrt(-N.potential(R, 0.0, Mmid, A))
    sb = np.sqrt(-N.potential(R, Mmid, Mout, B))
    return dict(masses=[0.0, Mmid, Mout], state=[R, -sa, R, sb], order=[0, 1], first=None)


def run(start, m):
    A, B = shells(m)
    r = N.simulate(
        [A, B],
        start["masses"],
        start["state"],
        order=start["order"],
        tmax=TMAX,
        anchor=1,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.02,
        max_events=1_000_000,
    )
    s, ms = start["state"], start["masses"]
    flights = [[0.0, s[0], s[2], s[1], s[3], ms[1], 1 if start["order"][0] == 0 else 0, 1 if start["first"] else 0, 1 if start["first"] else 0]]
    pts = [list(start["first"])] if start["first"] else []
    for e in r["events"]:
        sa = e["state_after"]
        a_inside = 1 if e["order_after"][0] == "A" else 0
        sec = 1 if (e["loser"] == "A" and e["v_before"][0] > 0 > e["v_before"][1]) else 0
        if sec:
            pts.append([e["r"], e["masses_before"][1]])
        flights.append([e["t"], sa[0], sa[2], sa[1], sa[3], e["masses_after"][1], a_inside, sec, kind(e)])
    return dict(flights=flights, t_end=r["t_end"], status=r["status"], points=np.array(pts))


def run_tangent(x0, m, Mout, tmax=TMAX):
    """Section points from the variational-tangent integrator, iterated as in bcr_layer_tangent.py (same settings and
    calls), with the crossings inside each return filled in by a one-return run of the solver from the section point."""
    P = TM.TwoShellMap(m=m, w=W, Min=0.0, Mt=Mout, rtol=1e-12, atol=1e-2 * 1e-12 * Mout / 0.03, max_step=0.02)
    xs, ts = [np.array(x0, float)], [0.0]
    while ts[-1] < tmax:
        st, x1, DF, tr, nc = TM._return(xs[-1], P.m, P.w, P.Min, P.Mt, P.rtol, P.atol, P.max_step, True, P.tmax)
        if st != 0:
            raise RuntimeError(f"return {len(xs)}: status {st}")
        xs.append(x1)
        ts.append(ts[-1] + tr)
    A, B = shells(m)
    flights, worst = [], 0.0
    for k in range(len(xs) - 1):
        s0 = from_section(xs[k], m, Mout)
        r = N.simulate(
            [A, B],
            s0["masses"],
            s0["state"],
            order=s0["order"],
            tmax=ts[k + 1] - ts[k] + 1.0,
            anchor=1,
            rtol=1e-12,
            atol=1e-14,
            max_step=0.02,
            max_events=40,
        )
        s = s0["state"]
        flights.append([ts[k], s[0], s[2], s[1], s[3], s0["masses"][1], 0, 1, 1])
        for e in r["events"]:
            if e["loser"] == "A" and e["v_before"][0] > 0 > e["v_before"][1]:  # the next section crossing: x_{k+1}
                worst = max(
                    worst,
                    abs(e["t"] - (ts[k + 1] - ts[k])),
                    abs(e["r"] - xs[k + 1][0]),
                    100 * abs(e["masses_before"][1] - xs[k + 1][1]),
                )
                break
            sa = e["state_after"]
            flights.append(
                [
                    ts[k] + e["t"],
                    sa[0],
                    sa[2],
                    sa[1],
                    sa[3],
                    e["masses_after"][1],
                    1 if e["order_after"][0] == "A" else 0,
                    0,
                    kind(e),
                ]
            )
        else:
            raise RuntimeError(f"return {k}: the one-return run found no section crossing")
    s0 = from_section(xs[-1], m, Mout)
    s = s0["state"]
    flights.append([ts[-1], s[0], s[2], s[1], s[3], s0["masses"][1], 0, 1, 1])
    return dict(flights=flights, t_end=ts[-1] + 1e-6, status="ok", points=np.array(xs), fill_mismatch=worst)


def rnd(fl):
    return [
        [round(f[0], 7), round(f[1], 8), round(f[2], 8), round(f[3], 7), round(f[4], 7), round(f[5], 9), f[6], f[7], f[8]]
        for f in fl
    ]


def main():
    out = sys.argv[2]
    S = json.load(open("results/bcr_section/bcr_section.json"))
    E1, E2, SAD = [fp["x"] for fp in S["fixed_points"]]
    m0, M0 = 0.00815, 0.03
    d = 0.03
    m2 = 0.9 * 20 ** (-0.4) * d
    orbits = [
        dict(
            key="example",
            name="BCR-5",
            m=m0,
            Mout=M0,
            start=from_initial(1.0, m0, 0.015, M0),
            twin=from_initial(1.0 + EPS, m0, 0.015, M0),
        ),
        dict(
            key="layer",
            name="Chaotic layer",
            m=m0,
            Mout=M0,
            start=from_section([SAD[0] + 1e-4, SAD[1]], m0, M0),
            twin=from_section([SAD[0] + 1e-4 + EPS, SAD[1]], m0, M0),
        ),
        dict(
            key="outside",
            name="Just outside the layer",
            m=m0,
            Mout=M0,
            start=from_section([SAD[0] + 0.07, SAD[1]], m0, M0),
            twin=from_section([SAD[0] + 0.07 + EPS, SAD[1]], m0, M0),
        ),
        dict(
            key="island",
            name="Around an island",
            m=m0,
            Mout=M0,
            start=from_section([E1[0] + 0.12, E1[1]], m0, M0),
            twin=from_section([E1[0] + 0.12 + EPS, E1[1]], m0, M0),
        ),
        dict(
            key="fixed",
            name="Fixed point",
            m=m0,
            Mout=M0,
            start=from_section(E1, m0, M0),
            twin=from_section([E1[0] + EPS, E1[1]], m0, M0),
        ),
        dict(
            key="ends",
            name="Chaos near the ends",
            m=m0,
            Mout=M0,
            start=from_section([0.19, 0.01575], m0, M0),
            twin=from_section([0.19 + EPS, 0.01575], m0, M0),
        ),
        dict(
            key="typeii",
            name="BCR-T2, δ = 0.03",
            m=m2,
            Mout=d,
            start=from_initial(1.5, m2, d / 2, d),
            twin=from_initial(1.5 + EPS, m2, d / 2, d),
        ),
    ]
    V = json.load(open("results/bcr_section/section_overview.json"))
    T = json.load(open("results/bcr_section/layer_tangent_example.json"))
    fig2 = {r["case"]["label"]: r for r in T["runs"]}
    for o in orbits:
        lab = {"layer": "R+ 0.0001", "outside": "R+ 0.07"}.get(o["key"])
        if lab:
            o["fig2"] = fig2[lab]
    data = dict(
        eps=EPS,
        w=W,
        fixed_points=[dict(x=fp["x"], kind=fp["kind"]) for fp in S["fixed_points"]],
        background=[[round(p[0], 5), round(p[1] * 1e3, 5)] for b in S["background"] for p in b["points"][::2]],
        # the section's allowed region: at each R, the interval of 1000 M_mid between the two turning-point edges,
        # closed by the two pointed ends
        edges=dict(
            rows=[[round(r[0], 5), round(r[1] * 1e3, 5), round(r[2] * 1e3, 5)] for r in V["edges"]],
            ends=[[round(V["ends"][k][0], 5), round(V["ends"][k][1] * 1e3, 5)] for k in ("small_R", "large_R")],
        ),
        orbits=[],
    )
    for o in orbits:
        if "fig2" in o:
            x0 = np.array(o["fig2"]["case"]["x0"])
            a, b = run_tangent(x0, o["m"], o["Mout"]), run_tangent(x0 + [EPS, 0.0], o["m"], o["Mout"])
            o["start"]["first"], o["twin"]["first"] = x0.tolist(), (x0 + [EPS, 0.0]).tolist()
            q = np.array(o["fig2"]["points"])  # the paper's points, every 4th return
            n4 = min(len(q), (len(a["points"]) + 3) // 4)
            dev = np.max(np.abs(q[:n4] - a["points"][::4][:n4]))
            print(
                f'{o["name"]}: largest difference from the paper\'s Fig. 3 points over {4 * (n4 - 1)} returns {dev:.1e};'
                f' one-return fill vs next section point {max(a["fill_mismatch"], b["fill_mismatch"]):.1e}'
            )
        else:
            a, b = run(o["start"], o["m"]), run(o["twin"], o["m"])
        # pair each of the orbit's section crossings with the twin's crossing nearest in time; where the twin has no
        # crossing within half a return (one of them skipped a section crossing near a fold of the chart), leave a gap
        ta = np.array([f[0] for f in a["flights"] if f[7] == 1])
        tb = np.array([f[0] for f in b["flights"] if f[7] == 1])
        assert len(ta) == len(a["points"]) and len(tb) == len(b["points"])
        half = 0.5 * np.median(np.diff(ta))
        j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
        j = np.where(np.abs(tb[j - 1] - ta) <= np.abs(tb[j] - ta), j - 1, j)
        ok = np.abs(tb[j] - ta) <= half
        dd = a["points"] - b["points"][j]
        sep = np.where(ok, np.hypot(dd[:, 0], 100 * dd[:, 1]), np.nan)
        pair = np.where(ok, j, -1)
        n = len(sep)
        print(
            f'{o["name"]}: {len(a["flights"])} flights, {len(a["points"])} section points, status {a["status"]};'
            f" {int((~ok).sum())} returns without a twin crossing within half a return;"
            f" separation {sep[1]:.2e} -> {np.nanmax(sep[: min(200, n)]):.2e} (max to 200) -> {np.nanmax(sep):.2e} (max)"
        )
        data["orbits"].append(
            dict(
                key=o["key"],
                name=o["name"],
                m=o["m"],
                Mout=o["Mout"],
                t_end=a["t_end"],
                first=o["start"]["first"],
                twin_first=o["twin"]["first"],
                flights=rnd(a["flights"]),
                twin=rnd(b["flights"]),
                sep=[None if np.isnan(x) else float(f"{x:.3g}") for x in sep],
                pair=[int(x) for x in pair],
            )
        )
    json.dump(data, open(out, "w"), separators=(",", ":"))
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
