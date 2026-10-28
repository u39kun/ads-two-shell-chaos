"""Figures of "Interacting shells in anti-de Sitter space revisited" from the saved results in results/.
Writes vector PDFs fig1.pdf through fig9.pdf, Tables 1-3 (table_edge.tex, table_refinement.tex,
table_convergence.tex),
and the numbers quoted with them (figure_numbers.json) to OUTDIR (default: figs). Set RESULTS_DIR to read another
results folder (compare_results.py).
usage: python make_figures_bcr.py [OUTDIR]"""

import sys, os, json, re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)

C1, C2, C3 = "#2a78d6", "#eb6834", "#199e70"  # categorical colors
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
SEQ = ["#a9c9ef", "#6fa3e3", "#2a78d6", "#174a8c"]  # one hue, light -> dark
DW = 7.0  # figure width (inches)
# point clouds: one small dot per crossing, no outline, so a curve's drawn width is not mistaken for spread in the data
DOT = dict(marker="o", ms=0.6, mew=0, ls="none", rasterized=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "savefig.dpi": 600,  # resolution of the rasterized point clouds
    }
)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), metadata={"CreationDate": None})  # no timestamp: reruns give identical files
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 1: crossing types and two head-on returns
def fig_crossings():
    """An ordinary return and a longer return containing overtaking encounters."""
    data = json.load(open(os.path.join(RES, "paper", "bcr_crossings.json")))
    fig, axes = plt.subplots(1, 2, figsize=(DW, 2.6), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.7], wspace=0.18))
    for index, (ax, example) in enumerate(zip(axes, data["examples"])):
        _crossing_panel(ax, example, index)
    axes[0].set_ylabel(r"shell radius $R$")
    save(fig, "fig1.pdf")
    return {
        example["name"]: dict(
            return_times=example["return_times"], crossings_per_return=example["crossings_per_return"]
        )
        for example in data["examples"]
    }


def _crossing_panel(ax, data, index):
    t0, t1 = data["return_times"]
    duration = t1 - t0
    for flight in data["segments"]:
        times = np.array(flight["t"]) - t0
        ax.plot(times, flight["R_A"], color=C1, lw=1.15)
        ax.plot(times, flight["R_B"], color=C2, lw=1.05)
    for event in data["events"]:
        if event["pair"][0] != "A":
            ax.plot(event["t"] - t0, event["R"], "D", ms=3.3, mfc=MUTED, mec=INK2, mew=0.5, zorder=5)
            continue
        selected = event["kind"] == "selected"
        ax.plot(
            event["t"] - t0,
            event["R"],
            "o",
            ms=4.3 if selected else 3.7,
            mfc=INK if selected else "white",
            mec=INK if selected else INK2,
            mew=0.75,
            zorder=5,
        )

    def label(event, text, position, connectionstyle="arc3"):
        ax.annotate(
            text,
            xy=(event["t"] - t0, event["R"]),
            xytext=position,
            ha="center",
            va="center",
            fontsize=7,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, ls="--", connectionstyle=connectionstyle),
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9),
        )

    selected = [event for event in data["events"] if event["kind"] == "selected"]
    opposite = next(event for event in data["events"] if event["kind"] == "opposite_head_on")
    if index == 0:
        label(
            selected[-1],
            "head-on, recorded\nA inside",
            (0.66 * duration, 3.5),
            connectionstyle="angle,angleA=0,angleB=90,rad=3",
        )
        label(opposite, "B inside", (0.15 * duration, 2.9), connectionstyle="angle,angleA=0,angleB=90,rad=3")
        arrow_event = selected[-1]
    else:
        overtaking = [event for event in data["events"] if event["kind"] == "overtaking" and event["pair"][0] == "A"]
        position = (overtaking[1]["t"] - t0, 4.5)
        # Route the shared leader above every shell arc, then descend directly to each crossing.
        branch_y = 4.1
        for event in overtaking:
            x = event["t"] - t0
            if event is overtaking[1]:
                ax.plot([x, x], [event["R"], position[1]], color=INK2, lw=0.6, ls="--", zorder=2)
            else:
                ax.annotate(
                    "",
                    xy=(x, event["R"]),
                    xytext=(position[0], branch_y),
                    zorder=2,
                    arrowprops=dict(
                        arrowstyle="-",
                        color=INK2,
                        lw=0.6,
                        ls="--",
                        shrinkA=0,
                        shrinkB=0,
                        connectionstyle="angle,angleA=0,angleB=90,rad=3",
                    ),
                )
        ax.text(
            *position,
            "overtaking, omitted\nA inside but both inward",
            ha="center",
            va="center",
            fontsize=7,
            color=INK,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9),
        )
        arrow_event = overtaking[0]
        ax.annotate(
            r"$\ast$",
            xy=(selected[0]["t"] - t0, selected[0]["R"]),
            xytext=(2, 1),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color=INK,
            zorder=6,
        )
    for name, color in (("A", C1), ("B", C2)):
        points = [
            (t - t0, r)
            for flight in data["segments"]
            for t, r in zip(flight["t"], flight["R_" + name])
            if 0.2 * duration < t - t0 < 0.8 * duration
        ]
        x, y = max(points, key=lambda p: p[1])
        if index == 1 and name == "A":
            x, y = 7.3, 1.0
        if index == 1 and name == "B":
            x, y = 6.5, 3.45
        ax.text(x, y + 0.16, name, color=color, fontsize=9, ha="center", va="bottom")
    for phase in (0, duration):
        ax.plot(
            [phase, phase], [0, 1.02], transform=ax.get_xaxis_transform(), color=GRID, lw=0.7, zorder=0, clip_on=False
        )
    # Fixed-size end markers show (initial, final] without shifting either endpoint off its guide.
    ax.plot(
        [0, duration],
        [1.02, 1.02],
        transform=ax.get_xaxis_transform(),
        color=INK2,
        lw=0.7,
        clip_on=False,
    )
    left = Path([(0.35, 0.5), (-0.35, 0), (0.35, -0.5)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    right = Path(
        [(-0.35, 0.5), (0, 0.5), (0, -0.5), (-0.35, -0.5)], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
    )
    for phase, marker in ((0, left), (duration, right)):
        ax.plot(
            phase,
            1.02,
            marker=marker,
            ms=8,
            mfc="none",
            mec=INK2,
            mew=0.7,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    ax.text(
        duration / 2,
        1.045,
        "one return",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    ax.set_xlim(data["window"][0] - t0, data["window"][1] - t0)
    ax.set_ylim(0, 4.9)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticks([0, 2, 4] if index == 0 else [0, 4, 8, 12, 16])
    ax.set_xlabel(r"middle-region time $t-t_0$")
    # One pair of arrows follows the curves on a monotonic incoming flight, clear of the crossing marker.
    a, b = arrow_event["t"] - 0.65, arrow_event["t"] - 0.25
    flight = next(s for s in data["segments"] if s["t"][0] <= a and s["t"][-1] >= b)
    times = np.linspace(a, b, 40)
    for name, color, velocity in zip(("R_A", "R_B"), (C1, C2), arrow_event["incoming_dR_dt"]):
        radii = np.interp(times, flight["t"], flight[name])
        if not np.all(np.diff(radii) * velocity > 0):
            raise ValueError("The illustrative arrow must not straddle a turning point")
        path = Path(np.column_stack((times - t0, radii)))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8.5, color=color, lw=0.8, zorder=4))
    title = "(a) Two crossings" if index == 0 else "(b) Eight crossings, including overtaking"
    ax.text(0.5, 1.22, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)


# ------------------------------------------------------------------ Fig. 4: nudged runs, as in BCR Fig. 6
def fig_bcr_lyapunov():
    D = json.load(open(os.path.join(RES, "paper", "bcr_lyapunov.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(DW, 2.4), gridspec_kw=dict(wspace=0.32))
    shades = {5e-5: SEQ[3], 5e-6: SEQ[2], 5e-7: SEQ[1]}
    # BCR's Fig. 6 observable: the outermost radius R_1 = max(R_A, R_B), which changes shell at each crossing
    bs = D["clocks"]["between_shells"]["runs"]
    bd = D["clocks"]["boundary"]["runs"]
    r0 = [r for r in bs if abs(r["eps"] - 5e-5) < 1e-12][0]
    t = np.array(r0["t"])
    d = np.array(r0["dR_outermost"])
    a.semilogy(t, np.maximum(d, 1e-9), color=SEQ[3], lw=0.5, label=r"$|\Delta R_1|$, clock between shells")
    rb = [r for r in bd if abs(r["eps"] - 5e-5) < 1e-12][0]
    a.semilogy(
        rb["t"],
        np.maximum(np.array(rb["envelope_outermost"]), 1e-9),
        color=C2,
        lw=0.9,
        ls=(0, (3, 2)),
        label="envelope, boundary clock",
    )
    tt = np.linspace(0, 250, 50)
    a.semilogy(tt, 5e-5 * np.exp(0.04 * tt), color=INK, lw=0.9, label=r"$5\times10^{-5}e^{0.04t}$ (rate 0.04)")
    # approximate readings of BCR's Fig. 6 (by eye, not digitized): first times their separation reaches 1e-3, 1e-2,
    # 1e-1, and order unity, and two intermediate peaks
    a.semilogy([80, 125, 450, 500], [1e-3, 1e-2, 1e-1, 1], "D", color=C3, ms=3.5, mew=0, label="BCR Fig. 6: first reaches")
    a.semilogy([170, 280], [0.03, 0.08], "^", mfc=C3, mec=C3, ms=4.5, mew=0.9, label="BCR Fig. 6: peaks")
    a.set_ylim(1e-6, 1e2)
    a.set_xlim(0, 2000)
    a.set_xlabel("time $t$")
    a.set_ylabel(r"$|\Delta R_1|$, outermost radius")
    legend = a.legend(loc="upper right", bbox_to_anchor=(1.04, 1.0), fontsize=6, handlelength=1.5)
    # Separate the legend symbols from the BCR read-off points to their left.
    fig.canvas.draw()
    legend_box = legend.get_window_extent().transformed(a.transAxes.inverted())
    a.plot(
        [legend_box.x0, legend_box.x0],
        [legend_box.y0, legend_box.y1],
        transform=a.transAxes,
        color=INK2,
        lw=0.6,
        clip_on=False,
    )
    for run in bs:
        e = run["eps"]
        tt2 = np.array(run["t"])
        env = np.array(run["envelope_outermost"])
        k = tt2 > 0
        b.loglog(
            tt2[k],
            env[k] / e,
            color=shades.get(e, MUTED),
            lw=1.0,
            label=rf"$\epsilon = {e/10**np.floor(np.log10(e)):.0f}\times10^{{{int(np.floor(np.log10(e)))}}}$",
        )
    for run in bd:
        e = run["eps"]
        tt2 = np.array(run["t"])
        env = np.array(run["envelope_outermost"])
        k = tt2 > 0
        b.loglog(tt2[k], env[k] / e, color=shades.get(e, MUTED), lw=0.7, ls=(0, (2, 2)))
    pc = os.path.join(RES, "paper", "bcr_lyapunov_C.json")
    C = json.load(open(pc)) if os.path.exists(pc) else None
    if C:
        rc = [r for r in C["clocks"]["between_shells"]["runs"] if abs(r["eps"] - 5e-5) < 1e-12][0]
        tc = np.array(rc["t"])
        k = tc > 0
        b.loglog(
            tc[k],
            np.array(rc["envelope_outermost"])[k] / rc["eps"],
            color=MUTED,
            lw=0.8,
            label=r"$\epsilon = 5\times10^{-5}$ (opposite)",
        )
    x = np.geomspace(10, 2000, 20)
    b.loglog(x, 0.15 * x, color=INK2, lw=0.8, ls=(0, (1, 1.5)), label=r"$\propto t$, boundary clock")
    xe = np.linspace(10, 200, 50)
    b.loglog(xe, np.exp(0.04 * xe), color=INK, lw=0.9, label=r"$e^{0.04t}$")
    b.set_xlim(1, 2000)
    b.set_ylim(0.5, 1e4)
    b.set_xlabel("time $t$")
    b.set_ylabel(r"envelope of $|\Delta R_1|$ / $\epsilon$")
    b.legend(loc="upper left", fontsize=6)
    b.text(
        0.97,
        0.04,
        "the three blue $\\epsilon$ coincide\n$\\epsilon$: clock between shells",
        transform=b.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=INK2,
    )
    panel(a, "a")
    panel(b, "b")
    save(fig, "fig4.pdf")
    # numbers quoted in Sec. 3.3 and App. B, computed at full sampling by bcr_lyapunov.py: log-log slopes, threshold
    # times, and exponential-rate fits of the envelope over growing windows 0 < t <= T, for both observables
    nums = {}
    for name, data in (("reconstruction", D), ("opposite", C)):
        if data is None:
            continue
        for clock, v in data["clocks"].items():
            for run in v["runs"]:
                nums[f'{name}|{clock}|{run["eps"]:g}'] = dict(
                    outermost=dict(
                        loglog_slope=run["loglog_slope_outermost"],
                        fit_start=run["fit_start_outermost"],
                        thresholds=run["thresholds_outermost"],
                        window_rates=run["window_rates_outermost"],
                    ),
                    outer_shell=dict(
                        loglog_slope=run["loglog_slope"],
                        fit_start=run["fit_start"],
                        thresholds=run["thresholds"],
                        window_rates=run["window_rates"],
                    ),
                )
    return nums


# ------------------------------------------------------------------ Fig. 2: return map at BCR's Fig. 5 parameters
def fig_bcr_section():
    d = os.path.join(RES, "bcr_section")
    Sx = json.load(open(os.path.join(d, "bcr_section.json")))
    Ck = json.load(open(os.path.join(d, "bcr_checks.json")))
    LG = json.load(open(os.path.join(d, "bcr_benettin_long.json")))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DW, 2.4), gridspec_kw=dict(wspace=0.42, width_ratios=[1.15, 0.9, 1]))
    for bg in Sx["background"]:
        p = np.array(bg["points"])
        if len(p):
            a.plot(p[:, 0], p[:, 1] * 100, color=MUTED, **DOT)
    L = np.array(Ck["long_points"])
    a.plot(L[:, 0], L[:, 1] * 100, color=C1, **DOT)
    for fp in Sx["fixed_points"]:
        a.plot(fp["x"][0], fp["x"][1] * 100, "x" if fp["kind"] == "hyperbolic" else "o", color=INK, ms=3.5, mew=0.8)
    a.add_patch(plt.Rectangle((0.97, 1.46), 0.07, 0.1, fill=False, ec=C2, lw=0.7))
    a.set_xlabel("crossing radius $R$")
    a.set_ylabel(r"$100\,M_{\rm mid}$")
    k = (L[:, 1] * 1e3 > 14.6) & (L[:, 1] * 1e3 < 15.6) & (L[:, 0] > 0.9)
    b.plot(L[k, 1] * 100, L[k, 0], color=C1, **DOT)
    x = L[k, 1] * 1e3
    y = L[k, 0]
    c0 = np.median(x)
    k0 = np.abs(x - c0) < 0.05
    pf = np.polyfit(x[k0] - c0, y[k0], 8)
    res = float(np.sqrt(np.mean((y[k0] - np.polyval(pf, x[k0] - c0)) ** 2)))
    b.set_xlabel(r"$100\,M_{\rm mid}$")
    b.set_ylabel("$R$")
    b.text(0.04, 0.95, "axes exchanged", transform=b.transAxes, ha="left", va="top", fontsize=6, color=INK2)
    seen = set()
    shades = {1e-8: SEQ[3], 1e-9: SEQ[2], 1e-10: SEQ[1]}
    for r in list(Ck["benettin"]) + LG:
        key = (r["name"], r["d0"])
        if key in seen or (r["name"] == "BCR" and r["d0"] > 5e-7) or r["name"] == "center far":
            continue
        if r["name"] == "BCR" and r["d0"] != 1e-9:
            continue
        seen.add(key)
        kk = np.array(r["lambda_N"])
        if r["name"] == "BCR":
            c.loglog(
                kk[:, 0],
                kk[:, 1],
                "o-",
                ms=2.2,
                lw=0.9,
                color=shades[r["d0"]],
                label=f"BCR-5 orbit, $d_0 = 10^{{{int(np.log10(r['d0']))}}}$",
            )
        else:
            c.loglog(kk[:, 0], kk[:, 1], "s--", ms=2.2, lw=0.9, color=INK2, label="orbit near an elliptic point")
    JC = json.load(open(os.path.join(d, "jacobian_check.json")))
    ex = [r for r in JC["exact"] if r["v0"] == [1.0, 1.0]][0]
    kk = np.array(ex["lambda_N"])
    kk = kk[kk[:, 0] >= 100]
    c.loglog(kk[:, 0], kk[:, 1], "-", lw=1.3, color=INK, label="BCR-5 orbit, variational tangent")
    fd = [r for r in JC["fd"] if r["h"] == 1e-7 and r["rtol"] == 1e-12][0]
    kk = np.array(fd["lambda_N"])
    kk = kk[kk[:, 0] >= 100]
    c.loglog(kk[:, 0], kk[:, 1], "x--", ms=2.6, lw=0.8, color=C2, label=r"finite differences, $h = 10^{-7}$")
    n = np.logspace(2, 4.35, 20)
    c.loglog(n, 1.2 * np.log(n) / n, ":", color=INK, lw=0.8, label=r"$\propto \ln N/N$")
    tr = 20000.0 / len(Sx["bcr_orbit"][0]["points"])
    c.axhline(0.04 * tr, color=C2, lw=0.8, ls=(0, (3, 2)))
    c.text(110, 0.04 * tr * 1.15, "BCR: 0.04 per unit time", color=C2, fontsize=6.5)
    c.set_ylim(1e-7, 0.4)
    c.set_xlabel("returns $N$")
    c.set_ylabel(r"$\lambda_N$ per return")
    hs, ls = c.get_legend_handles_labels()
    d0_exp = lambda lab: -int(re.search(r"10\^\{(-?\d+)\}", lab).group(1)) if "d_0" in lab else 0
    order = sorted(
        range(len(ls)), key=lambda i: ("variational" not in ls[i], "BCR-5 orbit" not in ls[i], d0_exp(ls[i]))
    )
    c.legend(
        [hs[i] for i in order],
        [ls[i] for i in order],
        loc="lower left",
        fontsize=6.3,
        handlelength=1.8,
        labelspacing=0.2,
        borderaxespad=0.2,
    )
    panel(a, "a")
    panel(b, "b")
    panel(c, "c")
    save(fig, "fig2.pdf")
    lam_end = [r["lambda_N"][-1] for r in LG]
    return dict(
        zoom_rms=res,
        zoom_n=int(k.sum()),
        zoom_strip_n=int(k0.sum()),
        return_time=tr,
        lambda_final=lam_end,
        exact_tangent={str(v["v0"]): v["lambda_N"] for v in JC["exact"]},  # 'exact' = variational tangent
        fd_lambda_3000={f"h={r['h']:g},rtol={r['rtol']:g}": (r["lambda_N"][-1], r["stopped"]) for r in JC["fd"]},
        validation=JC["validation"],
        fold=JC["fold_diagnosis"]["median_error"],
        fixed_points=[dict(x=fp["x"], kind=fp["kind"], rho=fp["rotation_number"]) for fp in Sx["fixed_points"]],
    )


# ------------------------------------------------------------------ Fig. 5: type (ii) family across delta
def fig_bcr_delta():
    D = json.load(open(os.path.join(RES, "bcr_section", "delta_scan.json")))
    F7 = json.load(open(os.path.join(RES, "bcr_section", "bcr_fig7.json")))
    runs = sorted(D["runs"], key=lambda r: r["delta"])
    fig, (f7, a, b, c) = plt.subplots(1, 4, figsize=(DW, 2.2), gridspec_kw=dict(wspace=0.5))
    # (a) BCR's own rule: dR_1/dt where the outermost radius crosses R_1 = 2 (both directions)
    f7stats = {}
    for r in sorted(F7["runs"], key=lambda r: r["delta"]):
        v = np.array(r["values"])
        dl = r["delta"]
        vp = v[v > 0]  # outward passages; inward ones are the same values negated
        f7.plot(np.full(len(v), dl), v, color=C1, **DOT)  # both directions, as in BCR's figure
        f7stats[f"{dl:g}"] = dict(n=int(len(v)), out_min=float(vp.min()), out_max=float(vp.max()))
    f7.set_ylim(-3.6, 3.6)
    f7.set_xlim(0, 0.035)
    f7.set_xticks([0.01, 0.02, 0.03])
    f7.set_xlabel(r"$\delta$")
    f7.set_ylabel(r"$dR_1/dt$ where $R_1 = 2$")
    for r in runs:
        p = np.array(r["points"])
        dl = r["delta"]
        a.plot(np.full(len(p), dl), 3 * p[:, 1] / dl, color=C1, **DOT)
    a.set_xlabel(r"$\delta$")
    a.set_ylabel(r"section values $3M_{\rm mid}/\delta$")
    pick = [0.006, 0.018, 0.030, 0.037]
    cols = [SEQ[0], SEQ[1], SEQ[2], SEQ[3]]
    for dl, col in zip(pick, cols):
        r = [x for x in runs if abs(x["delta"] - dl) < 1e-9][0]
        p = np.array(r["points"])
        b.plot(p[:, 0], 3 * p[:, 1] / dl, color=col, **DOT)
        b.plot([], [], "o", ms=3, color=col, label=f"{dl:g}")
    b.set_xlabel("crossing radius $R$")
    b.set_ylabel(r"$3M_{\rm mid}/\delta$")
    b.legend(
        loc="center", bbox_to_anchor=(0.47, 0.5), fontsize=5.4, handletextpad=0.1, title=r"$\delta$", title_fontsize=6
    )
    dl = np.array([r["delta"] for r in runs])
    lam = np.array([dict((k, l) for k, l in r["benettin"])[4000] for r in runs])
    slope = np.array([r["loglog_slope"] for r in runs])
    tr = np.array([r["return_time"] for r in runs])
    c.semilogy(dl, lam, "o", ms=2.6, color=C1, label=r"$\lambda_{4000}$, inner shell contracting")
    DC = json.load(open(os.path.join(RES, "bcr_section", "delta_scan_C.json")))
    rc = sorted(DC["runs"], key=lambda r: r["delta"])
    lamc = np.array([dict((k, l) for k, l in r["benettin"])[4000] for r in rc])
    c.semilogy(
        [r["delta"] for r in rc], lamc, "o", ms=2.6, mfc="none", mec=C1, mew=0.7, label="inner shell expanding"
    )
    pt = os.path.join(RES, "bcr_section", "layer_tangent_family.json")
    lamt = None
    if os.path.exists(pt):
        T = json.load(open(pt))["runs"]
        dt_ = np.array([r["case"]["delta"] for r in T])
        lamt = np.array([dict(r["lambda_N"])[4000] for r in T])
        c.semilogy(dt_, lamt, "x", ms=3.0, mew=0.8, color=INK, label="variational tangent")
    c.axhline(np.log(4000) / 4000, color=INK2, lw=0.8, ls=":", label=r"$\ln N/N$, $N = 4000$")
    c.semilogy(dl, 0.04 * tr, color=C2, lw=0.8, ls=(0, (3, 2)), label="BCR: 0.04 per unit time")
    c.set_ylim(5e-5, 0.6)
    c.set_xlabel(r"$\delta$")
    c.set_ylabel(r"$\lambda_{4000}$ per return")
    c.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=5.2, handletextpad=0.3)
    panel(f7, "a")
    panel(a, "b")
    panel(b, "c")
    panel(c, "d")
    save(fig, "fig5.pdf")
    slc = np.array([r["loglog_slope"] for r in rc])
    return dict(
        deltas=dl.tolist(),
        lambda_4000=lam.tolist(),
        loglog_slope=slope.tolist(),
        return_time=tr.tolist(),
        lambda_4000_range=[float(lam.min()), float(lam.max())],
        slope_range=[float(slope.min()), float(slope.max())],
        lambda_4000_range_C=[float(lamc.min()), float(lamc.max())],
        slope_range_C=[float(slc.min()), float(slc.max())],
        lambda_4000_tangent_range=None if lamt is None else [float(lamt.min()), float(lamt.max())],
        fig7=f7stats,
    )


# ------------------------------------------------------------------ Fig. 3: chaotic orbits at the example's parameters
THR = 3e-3  # late-block exponent per return above which an orbit counts as strongly chaotic (Sec. 3.2)


def late_block(run, n1=2000, n2=6000):
    c = run["cumulative_log"]
    e = run.get("every", 10)
    return float((c[n2 // e - 1] - c[n1 // e - 1]) / (n2 - n1))


def fig_bcr_layer():
    d = os.path.join(RES, "bcr_section")
    T = json.load(open(os.path.join(d, "layer_tangent_example.json")))["runs"]
    TL = json.load(open(os.path.join(d, "tangent_long.json")))
    Ck = json.load(open(os.path.join(d, "bcr_checks.json")))
    Sx = json.load(open(os.path.join(d, "bcr_section.json")))
    saddle = np.array([fp["x"] for fp in Sx["fixed_points"] if fp["kind"] == "hyperbolic"][0])
    runs = {(r["case"]["direction"], r["case"]["dist"]): r for r in T}
    for r in T:
        r["late"] = late_block(r)
    chaotic = [r for r in T if r["case"]["direction"] != "example" and r["late"] > THR]
    regular = [r for r in T if r["case"]["direction"] != "example" and r["late"] <= THR]
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 4.6), gridspec_kw=dict(wspace=0.38, hspace=0.42))
    a, z, b, c = axs.flat
    L = np.array(Ck["long_points"])
    P = np.vstack([np.array(r["points"]) for r in chaotic])
    overtaking_lines = []
    overtaking_path = os.path.join(d, "bcr_overtaking.json")
    if os.path.exists(overtaking_path):  # older archived reruns store head-on crossings only
        for run in json.load(open(overtaking_path))["runs"]:
            color = C1 if run["label"] == "BCR-5" else INK2 if run["label"] in ("R+ 0.07", "R- 0.07") else C2
            for branch in ("both_in", "both_out"):
                points = np.array(run[branch])
                if len(points) > 1:
                    points = points[np.argsort(points[:, 0])]
                    (line,) = a.plot(
                        points[:, 0],
                        100 * points[:, 1],
                        color=color,
                        ls=":",
                        lw=0.7,
                        zorder=1,
                        dash_capstyle="round",
                    )
                    overtaking_lines.append(line)
    for ax, regs in ((a, (("R+", 0.07), ("R-", 0.07))), (z, (("R+", 0.01), ("R-", 0.01)))):
        ax.plot(L[:, 0], L[:, 1] * 100, color=C1, **DOT)
        zo = 3  # regular curves on top of the chaotic points, so they stay visible in (a) too
        ax.plot(P[:, 0], P[:, 1] * 100, color=C2, zorder=2, **DOT)
        for key in regs:
            q = np.array(runs[key]["points"])
            ax.plot(q[:, 0], q[:, 1] * 100, color=C3 if ax is z else INK2, zorder=zo, **DOT)
        ax.plot(saddle[0], saddle[1] * 100, "x", color=INK, ms=3.5, mew=0.8)
    # Align dots across nearby curves so their overlapping dash patterns retain visible gaps.
    a.autoscale_view()
    for line in overtaking_lines:
        points = line.get_xydata()
        display = a.transData.transform(points) * 72 / fig.dpi
        length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(display, axis=0), axis=1))]
        anchor = np.interp(0.8, points[:, 0], length)
        line.set_linestyle(((-anchor / line.get_linewidth()) % 3.4, (1.0, 2.4)))
    elliptic = [fp["x"] for fp in Sx["fixed_points"] if fp["kind"] == "elliptic"]
    for e in elliptic:  # without rescaling, so the dot patterns aligned above keep their gaps
        a.plot(e[0], 100 * e[1], "o", color=INK, ms=2.2, mew=0, zorder=4, scalex=False, scaley=False)
    zx = (saddle[0] - 0.03, saddle[0] + 0.03)
    zy = (saddle[1] * 100 - 0.03, saddle[1] * 100 + 0.03)  # equal scales in (R, 100 M)
    a.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=INK, lw=0.6))
    z.set_xlim(*zx)
    z.set_ylim(*zy)
    z.set_aspect("equal", adjustable="box")  # one unit of R and of 100 M_mid have the same length
    z.axhline(saddle[1] * 100, color=INK2, lw=0.5, ls=":")
    a.set_xlabel("crossing radius $R$")
    a.set_ylabel(r"$100\,M_{\rm mid}$")
    z.set_xlabel("crossing radius $R$")
    # extent of the chaotic points along the cut M = M_saddle (scaled chart), outer and inner side
    near = np.abs(P[:, 1] - saddle[1]) * 100 < 1e-3
    ext = dict(
        outer=float((P[near, 0] - saddle[0]).max()), inner=float((saddle[0] - P[near, 0]).max()), n=int(near.sum())
    )
    # where the regular orbits started 1e-2 from the saddle (R+ and R-) cross the same strip
    reg_cut = {}
    for key in (("R+", 0.01), ("R-", 0.01)):
        q = np.array(runs[key]["points"])
        qn = np.abs(q[:, 1] - saddle[1]) * 100 < 1e-3
        reg_cut[key[0]] = [float(np.abs(q[qn, 0] - saddle[0]).min()), float(np.abs(q[qn, 0] - saddle[0]).max())]
    marks = {"R+": "o", "R-": "s", "M+": "^", "M-": "v"}
    cols = {"R+": SEQ[3], "R-": SEQ[2], "M+": C2, "M-": "#b8501f"}
    for dr in marks:
        pts = sorted([(ds, r) for (d2, ds), r in runs.items() if d2 == dr])
        b.semilogx(
            [ds for ds, _ in pts],
            [r["late"] for _, r in pts],  # signed: the symlog axis shows negative exponents
            marks[dr] + "-",
            ms=2.4,
            lw=0.7,
            color=cols[dr],
            label=dr,
        )
    b.set_yscale("symlog", linthresh=1e-3, linscale=0.4)
    b.axhline(THR, color=INK2, lw=0.6, ls=":")
    b.annotate(
        '“strongly chaotic” above',
        xy=(0.03, THR),
        xycoords=b.get_yaxis_transform(),
        xytext=(0, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=6,
        color=INK2,
    )
    b.axvspan(0.149, 0.174, color=C1, alpha=0.4, lw=0)
    b.text(0.105, 0.018, "BCR-5 orbit", rotation=90, ha="center", va="center", fontsize=6, color=C1)
    b.set_xlim(5e-10, 0.4)
    b.set_ylim(-2e-3, 0.1)
    b.set_yticks([-1e-3, 0, 1e-3, 1e-2, 1e-1])
    b.set_xlabel("start distance from the saddle")
    b.set_ylabel(r"$\lambda_{[2000,6000]}$ per return", labelpad=1)
    b.legend(loc="lower left", fontsize=6, ncol=2, handlelength=1.2, columnspacing=0.6)
    sty = {
        "R+ 0.0001": (C2, "-"),
        "M- 0.0001": ("#b8501f", "-"),
        "R+ 0.01": (C3, "-"),  # same color as these orbits in panel (b)
        "R- 0.02": (SEQ[1], "-"),
        "R+ 0.07": (MUTED, "-"),
        "BCR's example": (C1, "-"),
    }
    lab = {
        "R+ 0.0001": r"R+ $10^{-4}$",
        "M- 0.0001": r"M$-$ $10^{-4}$",
        "R+ 0.01": r"R+ $10^{-2}$",
        "R- 0.02": r"R$-$ 0.02",
        "R+ 0.07": "R+ 0.07",
        "BCR's example": "BCR-5 orbit",
    }
    for r in TL["runs"]:
        if r["label"] not in sty:
            continue
        cc = np.array(r["cumulative_log"])
        N = 10 * np.arange(1, len(cc) + 1)
        k = N >= 100
        c.loglog(
            N[k],
            cc[k] / N[k],
            color=sty[r["label"]][0],
            ls=sty[r["label"]][1],
            lw=0.9,
            label=lab[r["label"]],
        )
    n = np.logspace(2, 4.7, 10)
    c.loglog(n, 1.2 * np.log(n) / n, ":", color=INK, lw=0.8, label=r"$\propto \ln N/N$")
    c.set_ylim(1e-7, 0.3)
    c.set_xlabel("returns $N$")
    c.set_ylabel(r"$\lambda_N$ per return", labelpad=1)
    c.legend(loc="lower left", fontsize=6.5, handlelength=1.4, labelspacing=0.15, borderaxespad=0.1)
    panel(a, "a")
    panel(z, "b")
    panel(b, "c")
    panel(c, "d")
    save(fig, "fig3.pdf")
    core = [r for r in chaotic]
    out = dict(
        threshold=THR,
        extent_along_cut=ext,  # strip |100 (M_mid - M_saddle)| < 1e-3, i.e. |M_mid - M_saddle| < 1e-5
        regular_along_cut=reg_cut,
        chaotic=[
            (r["case"]["direction"], r["case"]["dist"], r["late"], dict(r["lambda_N"]).get(6000)) for r in chaotic
        ],
        regular=[
            (r["case"]["direction"], r["case"]["dist"], r["late"], dict(r["lambda_N"]).get(6000)) for r in regular
        ],
        example=[(r["late"], dict(r["lambda_N"]).get(6000)) for r in T if r["case"]["direction"] == "example"],
        chaotic_late_range=[min(r["late"] for r in core), max(r["late"] for r in core)],
        regular_late_range=[min(r["late"] for r in regular), max(r["late"] for r in regular)],
        return_time_chaotic=float(np.mean([r["return_time"] for r in core])),
        long={r["label"]: dict(lambda_N=r["lambda_N"], late=r["late"]) for r in TL["runs"]},
    )
    if os.path.exists(overtaking_path):
        out["overtaking_near_elliptic"] = overtaking_near_elliptic(
            Sx["params"], elliptic, json.load(open(overtaking_path))["runs"]
        )
    return out


def overtaking_near_elliptic(params, elliptic, runs):
    """The overtaking crossings of Fig. 3a nearest each elliptic point in (R, 100 M_mid), over all orbits and for the
    BCR-5 orbit. The junction conditions fix both speeds at a crossing from (R, M_mid), so an overtaking crossing and a
    head-on one at nearly the same (R, M_mid) differ mainly in the sign of one shell's velocity (B's when both move
    out, A's when both move in): its velocity gap to the elliptic state measures how far apart the two states are."""
    sys.path.insert(0, HERE)
    import n_shells as N

    S = N.Shell("A", params["m"], params["w"])
    _, _, tot = params["masses"]
    vel = lambda R, M: (np.sqrt(-N.potential(R, 0.0, M, S)), np.sqrt(-N.potential(R, M, tot, S)))
    out = []
    for e in elliptic:
        ea, eb = vel(*e)
        row = dict(x=e, Rdot=[float(ea), float(-eb)])
        for key, sel in (("all", lambda lab: True), ("BCR-5", lambda lab: lab == "BCR-5")):
            pts = [
                (R, M, br, r["label"])
                for r in runs
                if sel(r["label"])
                for br in ("both_in", "both_out")
                for R, M in r[br]
            ]
            R, M, br, lab = min(pts, key=lambda p: np.hypot(p[0] - e[0], 100 * (p[1] - e[1])))
            sa, sb = vel(R, M)
            va, vb = (sa, sb) if br == "both_out" else (-sa, -sb)
            row[key] = dict(
                orbit=lab,
                branch=br,
                x=[R, M],
                distance=float(np.hypot(R - e[0], 100 * (M - e[1]))),
                Rdot=[float(va), float(vb)],
                velocity_gap=[float(abs(va - ea)), float(abs(vb + eb))],
            )
        out.append(row)
    return out


# ------------------------------------------------------------------ Fig. 6: chaotic orbits across delta
def fig_bcr_layer_delta():
    D = json.load(open(os.path.join(RES, "bcr_section", "layer_delta.json")))
    T = json.load(open(os.path.join(RES, "bcr_section", "layer_tangent_delta.json")))["runs"]
    fig, axs = plt.subplots(1, 2, figsize=(DW, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    shades = ["#c6dbf4", "#8fb8ea", "#4f8fdc", "#2a78d6", "#174a8c"]
    out = {}
    for ax, side, name in (
        (axs[0], -1, "inner side ($R < R_{\\rm saddle}$)"),
        (axs[1], 1, "outer side ($R > R_{\\rm saddle}$)"),
    ):
        od = []
        for s_, col in zip(D["summary"], shades):
            dl = s_["delta"]
            rows = sorted(
                [r for r in T if abs(r["case"]["delta"] - dl) < 1e-12 and r["case"]["side"] == side],
                key=lambda r: r["case"]["dist"],
            )
            x = [r["case"]["dist"] for r in rows]
            y = [late_block(r) for r in rows]
            ax.semilogx(x, y, "o-", ms=2.6, lw=0.9, color=col, label=rf"$\delta = {dl:g}$")
            od.append(s_["orbit_distance"][str(side)])
            ext = None
            for xx, yy in zip(x, y):
                if yy > THR:
                    ext = xx
                else:
                    break
            nxt = [xx for xx in x if ext is None or xx > ext]
            if ext is not None and nxt:  # the first sampled start beyond the strongly chaotic ones
                ax.plot(nxt[0], y[x.index(nxt[0])], "o", ms=5.0, mfc="none", mew=0.8, color=col, zorder=4)
            out.setdefault(f"{dl:g}", {})[side] = dict(
                dist=x,
                late=y,
                extent=[ext, nxt[0] if nxt else None],
                lam6000=[dict(r["lambda_N"]).get(6000) for r in rows],
                return_time=[r["return_time"] for r in rows],
            )
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.4)
        ax.axvspan(min(od), max(od), color=C1, alpha=0.4, lw=0)
        ax.text(min(od) / 1.12, 0.03, "BCR-T2\norbits", ha="right", fontsize=6.5, color=C1)
        ax.axhline(THR, color=INK2, ls=":", lw=0.8)
        ax.set_xlim(7e-4, 1.0)
        ax.set_ylim(-3e-3, 0.1)
        ax.set_xlabel("start distance from the saddle along $R$")
        ax.set_title(name, fontsize=7)
    axs[0].set_ylabel(r"$\lambda_{[2000,6000]}$ per return")
    axs[0].legend(loc="lower right", fontsize=5.6, ncol=2, handlelength=1.6, columnspacing=0.8, frameon=False)
    panel(axs[0], "a")
    panel(axs[1], "b")
    save(fig, "fig6.pdf")
    return out


# ------------------------------------------------------------------ Table 3: convergence of the exponents
def table_convergence():
    d = os.path.join(RES, "bcr_section")
    C = json.load(open(os.path.join(d, "layer_convergence.json")))
    T = json.load(open(os.path.join(d, "layer_tangent_example.json")))["runs"]
    orbits = [
        ("layer", "R+", 1e-4, "R+", 1e-4, r"R+, $10^{-4}$"),
        ("layer", "M-", 1e-4, "M-", 1e-4, r"M$-$, $10^{-4}$"),
        ("edge", "R+", 1e-2, "R+", 1e-2, r"R+, $10^{-2}$"),
        ("edge", "R-", 2e-2, "R-", 2e-2, r"R$-$, 0.02"),
        ("outside", "R+", 7e-2, "R+", 7e-2, "R+, 0.07"),
        ("example", "BCR", None, "example", None, "BCR-5"),
    ]
    rows = (
        [("variational tangent", None)]
        + [
            (rf"$d_0 = 10^{{{int(round(np.log10(x)))}}}$" + (" (scans)" if x == 1e-10 else ""), ("d0", x))
            for x in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
        ]
        + [
            ("renormalized every return", ("renorm", 1)),
            ("every 50 returns", ("renorm", 50)),
            (r"rtol $= 10^{-11}$", ("rtol", 1e-11)),
            (r"rtol $= 10^{-13}$", ("rtol", 1e-13)),
            ("max step 0.01", ("max_step", 0.01)),
            (r"separation along $(1, -1)$", ("sep", [1.0, -1.0])),
        ]
    )
    base = C["base"]

    def find(kind, dr, dist, setting):
        want = dict(base)
        if setting is not None:
            want[setting[0]] = setting[1]
        for r in C["runs"]:
            c = r["case"]
            if (c["kind"], c["direction"], c["dist"], c["returns"]) == (kind, dr, dist, 6000) and all(
                c[k] == want[k] for k in base
            ):
                return r
        return None

    def cell(v):
        if v is None:
            return "--"
        x = v * 1e3
        a = abs(x)
        t = f"{a:.0f}" if a >= 10 else (f"{a:.1f}" if a >= 1 else f"{a:.2f}")
        return ("$-$" if x < 0 else "") + t

    vals = {}
    lines = []
    for name, setting in rows:
        row = [name]
        for kind, dr, dist, tdr, tdist, _ in orbits:
            if setting is None:
                r = [q for q in T if q["case"]["direction"] == tdr and q["case"]["dist"] == tdist][0]
                v = late_block(r)
            else:
                r = find(kind, dr, dist, setting)
                v = None
                if r is not None:
                    S = np.array(r["cumulative_log"])
                    B = r["block"]
                    v = float((S[6000 // B - 1] - S[2000 // B - 1]) / 4000)
            vals[(name, dr, dist)] = v
            row.append(cell(v))
        lines.append(" & ".join(row) + r" \\")
    head = " & ".join(["setting"] + [o[5] for o in orbits]) + r" \\"
    tex = (
        r"""\begin{table}[!tbp]
\centering\small
\caption{Late-block exponents $\lambda_{[2000,6000]}$ in units of $10^{-3}$ per return, at BCR-5 parameters.
Columns give the start direction and distance from the saddle, or the BCR-5 orbit.
The first row uses the variational tangent; subsequent rows vary one two-orbit setting from the scan baseline:
$d_0 = 10^{-10}$ in $(R,100\,\Mmid)$, renormalization every 10 returns, rtol $=10^{-12}$, maximum step 0.02, and
initial separation along $(1,1)$; --: not run.
Changing $d_0$ preserves the reference trajectory, allowing direct comparison.
Other settings and the variational method follow different numerical chaotic trajectories after a few hundred
returns, sampling different finite-time exponents.}
\label{tab:convergence}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l"""
        + "r" * len(orbits)
        + r"""}
\toprule
"""
        + head
        + "\n\\midrule\n"
        + "\n".join(lines[:1])
        + "\n\\midrule\n"
        + "\n".join(lines[1:8])
        + "\n\\midrule\n"
        + "\n".join(lines[8:])
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    open(os.path.join(OUT, "table_convergence.tex"), "w").write(tex)
    print("wrote table_convergence.tex")
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in vals.items()}


def table_refinement():
    """Table 2: the variational tangent with a tighter and a looser tolerance and a smaller maximum step."""
    path = os.path.join(RES, "bcr_section", "tangent_refinement.json")
    if not os.path.exists(path):  # e.g. a rerun made before this check was added
        print("no tangent_refinement.json: Table 2 skipped")
        return None
    R = json.load(open(path))
    orbits = [
        ("R+ 1e-4", r"R+, $10^{-4}$ (saddle layer)", 1),
        ("R+ 1e-2", r"R+, $10^{-2}$", 3),
        ("BCR's example", "BCR-5", 6),
    ]
    vals, lines = {}, []
    for rt, ms in R["settings"]:
        name = rf"rtol $= 10^{{{int(round(np.log10(rt)))}}}$, max step {ms:g}" + (
            " (scans)" if (rt, ms) == (1e-12, 0.02) else ""
        )
        row = [name]
        for orb, _, nd in orbits:
            r = [
                q
                for q in R["runs"]
                if q["case"]["orbit"] == orb and q["case"]["rtol"] == rt and q["case"]["max_step"] == ms
            ][0]
            v = r["late"]["2000-6000"]
            vals[f"{orb}|{rt:g}|{ms:g}"] = v
            x = v * 1e3
            row.append(("$-$" if x < 0 else "") + f"{abs(x):.{nd}f}")
        lines.append(" & ".join(row) + r" \\")
    head = " & ".join(["setting"] + [o[1] for o in orbits]) + r" \\"
    tex = (
        r"""\begin{table}[!tbp]
\centering\small
\caption{Variational-tangent refinement (\texttt{bcr\_tangent\_refinement.py}): $\lambda_{[2000,6000]}$ in units of
$10^{-3}$ per return at BCR-5 parameters.
Columns identify a chaotic start, a nearby regular start, and the BCR-5 orbit; rows vary tolerance and maximum step.
The chaotic trajectory changes with the settings, but every exponent remains well above the classification threshold
of 3 in these units.}
\label{tab:refinement}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrr}
\toprule
"""
        + head
        + "\n\\midrule\n"
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    open(os.path.join(OUT, "table_refinement.tex"), "w").write(tex)
    print("wrote table_refinement.tex")
    return vals


# ------------------------------------------------------------------ section components and clocks (Secs. 2, 4.3, App. B)
def domain_clock_numbers():
    """The section's components (bcr_section_domain.py), the collapse criterion (bcr_collapse_criterion.py), and rates
    in boundary time (bcr_boundary_rates.py); no figure."""
    d = os.path.join(RES, "bcr_section")
    paths = {k: os.path.join(d, f"{k}.json") for k in ("section_domain", "collapse_criterion", "boundary_rates")}
    if not all(os.path.exists(p) for p in paths.values()):
        print("no section_domain/collapse_criterion/boundary_rates.json: their numbers skipped")
        return None
    D, C, B = (json.load(open(p)) for p in paths.values())
    near = lambda c: c["components"][0] if c["components"][0]["R"][1] < 0.1 else None
    out = dict(
        components={
            f"{c['label']} {c['delta']:g}": dict(
                R_horizon=c["R_horizon"],
                near_horizon=near(c),
                near_horizon_at_horizon=c["lowest_component_at_horizon"],
                main=c["components"][-1],
            )
            for c in D["components"]
        },
        near_horizon_onset_between=D["onset_bracket"],
        survey_starts_failing_full_test={f"{g['delta']:g}": g["allowed_not_full"] for g in D["survey_grids"]},
        near_horizon_fate={
            f"{f['delta']:g}": dict(
                starts=f["starts"], outcomes=f["outcomes"], t_max=f["t_end_max"], crossings_max=f["crossings_max"]
            )
            for f in D["fate"]
        },
    )
    sh = [s for st in C["starts"] for v in st["shift_per_factor_100"].values() for s in v]
    bm = [st["boundary_over_middle"] for st in C["starts"]]
    out["collapse_cutoff"] = dict(
        shift_per_factor_100=[min(sh), max(sh)],
        crossings_unchanged=all(len(st["crossings"]) == 1 for st in C["starts"]),
        sequences_identical=all(
            v["kinds_identical"] and v["max_dt"] == 0 and v["max_dr"] == 0
            for st in C["starts"]
            for v in st["sequence_vs_cutoff"].values()
        ),
        all_collapse=all(st["statuses"] == ["horizon_approach"] for st in C["starts"]),
        boundary_over_middle_collapse_time=[min(bm), max(bm)],
    )
    out["threshold"] = {f"anchor {t['anchor']}": t["threshold"] for t in C["threshold"]}
    cuts = [r for r in B["runs"] if r["job"]["kind"] == "cuts"]
    tab = [r for r in B["runs"] if r["job"]["kind"] == "table" and "control" not in r["job"]["label"]]
    fc = [r["late"]["bdry_per_mid"] for r in cuts]
    ft = [b["bdry_per_mid"] for r in tab for b in r["blocks"]]
    out["boundary_time"] = dict(
        cuts_factor=[min(fc), max(fc)],
        cuts_rate_drop=1 - 1 / max(fc),
        table_block_factor=[min(ft), max(ft)],
        table_rate_drop=[1 - 1 / min(ft), 1 - 1 / max(ft)],
        cuts_max_rate={k: max(r["late"][f"rate_{k}"] for r in cuts) for k in ("mid", "bdry")},
        table_max_block_rate={k: max(b[f"rate_{k}"] for r in tab for b in r["blocks"]) for k in ("mid", "bdry")},
        reproduces_stored_tangent=max(r["max_deviation_from_stored"] for r in cuts),
    )
    return out


def family_long_numbers():
    """The type (ii) family followed with the variational tangent for F["returns"] returns (5e4 in the paper;
    bcr_family_long.py, Sec. 4.1): lambda_N at 4000 returns and at the end of the run, the late block, and rates per
    unit time; no figure."""
    path = os.path.join(RES, "bcr_section", "family_long.json")
    if not os.path.exists(path):
        print("no family_long.json: its numbers skipped")
        return None
    F = json.load(open(path))
    n = F["returns"]
    ok = [r for r in F["runs"] if r["status"] == "ok"]
    lam = lambda N: [r["lambda_N"][str(N)] for r in ok]
    late = [r["late_block"] for r in ok if r["late_block"]]
    return dict(
        returns=n,
        deltas=len(ok),
        stopped=[r["delta"] for r in F["runs"] if r["status"] != "ok"],
        lambda_4000=[min(lam(4000)), max(lam(4000))] if n >= 4000 else None,
        lambda_end=[min(lam(n)), max(lam(n))],
        lnN_over_N=ok[0]["lnN_over_N"],
        late_block=dict(
            returns=late[0]["returns"], exponent=[min(b["exponent"] for b in late), max(b["exponent"] for b in late)]
        )
        if late
        else None,
        per_unit_time_end=[
            min(r["lambda_N"][str(n)] / r["return_time"] for r in ok),
            max(r["lambda_N"][str(n)] / r["return_time"] for r in ok),
        ],
    )


def chart_check_numbers():
    """Table 1 rates in the chart and in a norm regular at the folds, and the fixed points' periods
    (bcr_chart_check.py; Secs. 3.1 and 4.3, App. C); no figure."""
    path = os.path.join(RES, "bcr_section", "chart_check.json")
    if not os.path.exists(path):
        print("no chart_check.json: its numbers skipped")
        return None
    C = json.load(open(path))
    rows = [r for r in C["rows"] if r["status"] == "ok"]
    chaotic = [r for r in rows if "control" not in r["label"]]
    return dict(
        max_rate_difference=max(r["max_rate_difference"] for r in rows),
        chaotic_block_rates_chart=[
            min(b["rate_chart"] for r in chaotic for b in r["blocks"]),
            max(b["rate_chart"] for r in chaotic for b in r["blocks"]),
        ],
        max_abs_log_distortion=max(r["max_abs_distortion"] for r in rows),
        fixed_points=C["fixed_points"],
    )


def ends_followup_numbers():
    """The strongest starts of the finer end grids that detect nothing, followed to T = 2e4 (bcr_ends_followup.py),
    and the rate that linear tangent growth gives over each classification window (Sec. 4.3, App. C); no figure."""
    path = os.path.join(RES, "bcr_section", "ends_followup.json")
    if not os.path.exists(path):
        print("no ends_followup.json: its numbers skipped")
        return None
    F = json.load(open(path))["runs"]
    decay = [r for r in F if r["rate"] < SURVEY_THR]
    keep = [r for r in F if r["rate"] >= SURVEY_THR]
    return dict(
        grids=sorted({f"{r['region']}|{r['delta']:g}" for r in F}),
        short_rate_max=max(r["short_rate"] for r in F),
        decaying=dict(n=len(decay), long_rate_max=max(r["rate"] for r in decay)),
        persisting=[
            dict(
                region=r["region"],
                delta=r["delta"],
                rate=r["rate"],
                blocks=r["blocks"],
                crossings_per_return=r["crossings_per_return"],
                distance_from_saddle=r["distance_from_saddle"],
            )
            for r in keep
        ],
        linear_growth=dict(
            per_return_2000_6000=float(np.log(3) / 4000),
            survey_T4_T=float(np.log(4) / 15000),
            ends_T4_T=float(np.log(4) / 1500),
            blocks_2_to_4=[float(np.log(k / (k - 1)) / 5000) for k in (2, 3, 4)],
        ),
    )


# ------------------------------------------------------------------ weighted Birkhoff averages (Secs. 3.3, 4.1, App. B)
def wba_numbers():
    """Agreement in digits of weighted Birkhoff averages (bcr_wba.py): the least over the three observables of d(N),
    over N vs 2N samples, at all crossings with A inside and on the head-on section; no figure."""
    path = os.path.join(RES, "bcr_section", "bcr_wba.json")
    if not os.path.exists(path):
        print("no bcr_wba.json: weighted Birkhoff numbers skipped")
        return None
    W = {r["case"]["label"]: r for r in json.load(open(path))["runs"]}
    d = lambda t, key="n_vs_2n": {n: min(v[key] for v in obs.values()) for n, obs in t.items()}
    out = {}
    for lab in ("example D", "example C"):
        r = W[lab]
        out[lab] = dict(
            all_crossings=d(r["all_a_inside"]),
            halves=d(r["all_a_inside"], "halves"),
            head_on=d(r["head_on"]),
            overtaking=1 - r["counts"]["head_on"] / r["a_inside"],
        )
    out["controls"] = {lab: d(r["all_a_inside"]) for lab, r in W.items() if lab.startswith("control")}
    out["tolerance_8000"] = {
        f"{r['case']['rtol']:g}|{r['case']['max_step']:g}": d(r["all_a_inside"]).get("8000")
        for lab, r in W.items()
        if lab.startswith("tolerance")
    }
    fam = [r for lab, r in W.items() if lab.startswith("family")]
    allc = [d(r["all_a_inside"]).get("8000") for r in fam]
    head = [d(r["head_on"])[max(r["head_on"], key=int)] for r in fam]
    ov = [1 - r["counts"]["head_on"] / r["a_inside"] for r in fam]
    out["family"] = dict(
        all_crossings_8000=[min(allc), max(allc)],
        head_on_at_largest_N=sorted(head),
        overtaking=[min(ov), max(ov)],
    )
    # the BCR-5 orbit to t = 2e4 (bcr_section.py, start D): the inner shell alternates, so half of the crossings have
    # A inside, and the head-on ones are the section points
    S = json.load(open(os.path.join(RES, "bcr_section", "bcr_section.json")))
    r = [x for x in S["bcr_orbit"] if x["start"] == "D"][0]
    out["bcr5_overtaking_t2e4"] = 1 - len(r["points"]) / (r["crossings"] / 2)
    return out


# ------------------------------------------------------------------ Figs. 7-8 and Table 1: chaos away from the saddle
C4 = "#b5267a"  # chaos at the section's ends; orange (C2) stays chaos near the saddle, as in Fig. 3a
TIP_THR = 1e-2  # rate per unit time that marks a strongly chaotic start on the short runs of the finer end grids
SURVEY_THR = 1e-3  # rate per unit time over [T/4, T] that flags a survey start (T = 20000)
FEW = 3.0  # mean crossings per return separating the two kinds of flagged survey start


def cut_band():
    """Late-block rates per unit time of the strongly chaotic starts along the cuts (Fig. 6, late >= 1e-2 per
    return), excluding the weaker starts at the edge of the chaotic region: the band near the saddle in Fig. 8f."""
    T = json.load(open(os.path.join(RES, "bcr_section", "layer_tangent_delta.json")))["runs"]
    r = [late_block(x) / x["return_time"] for x in T if late_block(x) >= 1e-2]
    return min(r), max(r)


def fig_bcr_overview():
    """Fig. 7: the whole head-on section at BCR's Fig. 5 parameters, in the chart of Fig. 2a."""
    d = os.path.join(RES, "bcr_section")
    O = json.load(open(os.path.join(d, "section_overview.json")))
    Sx = json.load(open(os.path.join(d, "bcr_section.json")))
    T = json.load(open(os.path.join(d, "layer_tangent_example.json")))["runs"]
    E = np.array(O["edges"])
    fig, ax = plt.subplots(figsize=(DW, 2.7))
    ax.fill_between(E[:, 0], 100 * E[:, 1], 100 * E[:, 2], color="#f3f2ee", lw=0)
    ax.plot(E[:, 0], 100 * E[:, 1], color=INK2, lw=0.7)  # lower edge: A at a turning point
    ax.plot(E[:, 0], 100 * E[:, 2], color=INK2, lw=0.7, ls="--")  # upper edge: B at a turning point
    allp = []
    for b in Sx["background"]:
        p = np.array(b["points"])
        if len(p):
            ax.plot(p[:, 0], 100 * p[:, 1], color=MUTED, **DOT)
            allp.append(p)
    p = np.array(Sx["bcr_orbit"][0]["points"])
    allp.append(p)
    ax.plot(p[:, 0], 100 * p[:, 1], color=C1, **DOT)
    chaotic = [r for r in T if r["case"]["direction"] != "example" and late_block(r) > THR]
    Pch = np.vstack([np.array(r["points"]) for r in chaotic])
    ax.plot(Pch[:, 0], 100 * Pch[:, 1], color=C2, zorder=2, **DOT)
    tip = O["orbits"][0]
    q = np.array(tip["headon"])
    ax.plot(q[:, 0], 100 * q[:, 1], **{**DOT, "ms": 0.9}, color=C4)
    for fp in Sx["fixed_points"]:
        if fp["kind"] == "hyperbolic":
            ax.plot(fp["x"][0], 100 * fp["x"][1], "x", color=INK, ms=3.5, mew=0.8)
        else:
            ax.plot(fp["x"][0], 100 * fp["x"][1], "o", color=INK, ms=2.2, mew=0)
    # range of Fig. 2a (data extent plus matplotlib's default 5% margins)
    A = np.vstack(allp)
    x0, x1, y0, y1 = A[:, 0].min(), A[:, 0].max(), 100 * A[:, 1].min(), 100 * A[:, 1].max()
    mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    ax.add_patch(
        plt.Rectangle((x0 - mx, y0 - my), x1 - x0 + 2 * mx, y1 - y0 + 2 * my, fill=False, ls=":", lw=0.7, ec=INK2)
    )
    ax.text(x1 + mx, y1 + my + 0.02, "Fig. 2a", fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$100\,M_{\rm mid}$")
    ax.set_xlim(0.1, 2.55)
    ax.set_ylim(1.05, 1.98)
    for text, col, off in (
        ("upper edge: B at a turning point", 2, -0.05),
        ("lower edge: A at a turning point", 1, 0.05),
    ):
        Rl = 1.85
        y = lambda R: 100 * np.interp(R, E[:, 0], E[:, col])
        p0, p1 = ax.transData.transform((Rl - 0.1, y(Rl - 0.1))), ax.transData.transform((Rl + 0.1, y(Rl + 0.1)))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(
            Rl,
            y(Rl) + off,
            text,
            fontsize=6.5,
            color=INK2,
            rotation=ang,
            rotation_mode="anchor",
            ha="center",
            va="center",
        )
    save(fig, "fig7.pdf")
    orbits = [{k: v for k, v in o.items() if k != "headon"} | dict(headon=len(o["headon"])) for o in O["orbits"]]
    return dict(
        ends=O["ends"],
        near_end_cuts=O["near_end_cuts"],
        orbits=orbits,
        near_saddle_chaotic=len(chaotic),
        fig1a_box=[x0 - mx, x1 + mx, y0 - my, y1 + my],
    )


def fig_bcr_survey():
    """Fig. 8: survey of the head-on section across the type (ii) family, and the rates near the section's ends."""
    d = os.path.join(RES, "bcr_section")
    Sv = json.load(open(os.path.join(d, "survey.json")))
    En = json.load(open(os.path.join(d, "survey_ends.json")))
    Ck = json.load(open(os.path.join(d, "survey_checks.json")))
    saddles = {s["delta"]: s["saddle"] for s in json.load(open(os.path.join(d, "layer_delta.json")))["summary"]}
    band = cut_band()
    fig = plt.figure(figsize=(DW, 3.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.55, wspace=0.9)
    axes = [fig.add_subplot(gs[i // 3, 2 * (i % 3) : 2 * (i % 3) + 2]) for i in range(6)]
    out = dict(T=Sv["T"], threshold=SURVEY_THR, few_crossings=FEW, near_saddle_band=list(band), survey={}, ends={})
    for k, (ax, g) in enumerate(zip(axes[:5], Sv["grids"])):
        dl = g["delta"]
        runs = g["runs"]
        R = np.array([r["x0"][0] for r in runs])
        m = np.array([3 * r["x0"][1] / dl for r in runs])
        lam = np.array([r.get("rate_per_time", np.nan) for r in runs])
        cpr = np.array([r["crossings_per_return_mean"] or 0 for r in runs])
        coll = np.array([r.get("event_status") == "horizon_approach" for r in runs])
        stop = np.array([r["status"] != "ok" for r in runs]) & ~coll  # stopped but not collapsing (none expected)
        below = ~coll & ~stop & ~(lam > SURVEY_THR)
        few = ~coll & (lam > SURVEY_THR) & (cpr <= FEW)
        many = ~coll & (lam > SURVEY_THR) & (cpr > FEW)
        ax.plot(R[below], m[below], "o", ms=1.1, mew=0, color=MUTED, rasterized=True)
        ax.plot(R[coll], m[coll], marker=(6, 2, 0), ls="none", ms=2.4, mew=0.45, color=INK2, rasterized=True)
        ax.plot(R[stop], m[stop], "s", ms=2.0, mew=0, color=INK)
        ax.plot(R[few], m[few], "o", ms=3.0, mew=0.35, mec="white", color=C2, zorder=4)
        ax.plot(R[many], m[many], "o", ms=3.0, mew=0.35, mec="white", color=C4, zorder=4)
        s = saddles[dl]
        ax.plot([s[0]], [3 * s[1] / dl], "x", ms=3.5, mew=0.8, color=INK, zorder=5)
        ax.set_xlim(0.1, 2.45)
        ax.set_ylim(1.08, 1.95)
        ax.set_title(rf"$\delta = {dl:g}$", pad=2)
        ax.set_xlabel("crossing radius $R$", labelpad=1)
        if k in (0, 3):
            ax.set_ylabel(r"$3M_{\rm mid}/\delta$")
        panel(ax, "abcde"[k])
        dist = np.hypot(R[few] - s[0], m[few] - 3 * s[1] / dl)
        tcol = np.array([r.get("event_t_end", np.nan) for r in runs])[coll]
        out["survey"][f"{dl:g}"] = dict(
            starts=len(runs),
            spacing_R=(g["box"][1] - g["box"][0]) / g["nr"],
            flagged_few=int(few.sum()),
            flagged_many=int(many.sum()),
            flagged_few_max_distance_from_saddle=float(dist.max()) if len(dist) else None,
            flagged_many_rates=sorted(lam[many].tolist()),
            collapse=int(coll.sum()),
            collapse_fraction=float(coll.mean()),
            collapse_time_max=float(tcol.max()) if len(tcol) else None,
            stopped_other=int(stop.sum()),
        )
    # key to the markers of panels a-e, above the top row
    from matplotlib.lines import Line2D

    key = [
        Line2D([], [], marker="o", ls="none", ms=2.2, mew=0, color=MUTED, label=r"rate $\leq 10^{-3}$"),
        Line2D([], [], marker="o", ls="none", ms=3.0, mew=0, color=C2, label=r"$> 10^{-3}$, $\leq 3$ crossings/return"),
        Line2D([], [], marker="o", ls="none", ms=3.0, mew=0, color=C4, label=r"$> 10^{-3}$, $> 3$ crossings/return"),
        Line2D([], [], marker=(6, 2, 0), ls="none", ms=4.5, mew=0.7, color=INK2, label="collapse"),
        Line2D([], [], marker="x", ls="none", ms=3.5, mew=0.8, color=INK, label="saddle"),
    ]
    fig.legend(
        handles=key,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        fontsize=6,
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.2,
    )
    # (f) rates near the ends against delta: maxima of the finer grids (hollow, muted) and the Table 1 starts (one bar
    # per start over the blocks of both integration settings, with a filled marker for the end: circle small R, square large R)
    ax = axes[5]
    for region, mk in (("small R", "o"), ("large R", "s")):
        gr = sorted([e for e in En["grids"] if e["region"] == region], key=lambda e: e["delta"])
        ds = [e["delta"] for e in gr]
        mx = [max(r.get("rate_per_time", 0.0) for r in e["runs"]) for e in gr]
        ax.plot(ds, mx, mk, ms=2.6, mfc="none", mew=0.6, color=MUTED, ls="none", label=f"{region[:-2]}-$R$ end, finer grid max")
        for e, x in zip(gr, mx):
            rates = [r.get("rate_per_time", 0.0) for r in e["runs"]]
            out["ends"][f"{region}|{e['delta']:g}"] = dict(
                starts=len(rates),
                above_1e2=sum(x > TIP_THR for x in rates),
                above_3e3=sum(x > 3e-3 for x in rates),
                max=x,
            )
    for row in Ck["rows"]:
        if row["control"]:
            continue
        b = [x for s in ("default", "tight") for x in row["tangent"][s]["blocks"] if x is not None]
        small = row["label"].startswith("small")
        x = row["delta"] + (-0.0005 if small else 0.0005)
        ax.plot([x, x], [min(b), max(b)], "-", lw=1.6, color=C4, solid_capstyle="butt")
        ax.plot([x], [np.sqrt(min(b) * max(b))], "o" if small else "s", ms=3.2, mfc=C4, mec="white", mew=0.5, color=C4)
    ax.plot([0], [1e9], "o", ms=3.2, mfc=C4, mec="white", mew=0.5, color=C4, lw=1.6, label="Table 1 start, range of blocks")
    ax.axhline(TIP_THR, ls=":", lw=0.7, color=MUTED)
    ax.axhspan(*band, color=C2, alpha=0.15, lw=0)
    ax.text(0.0385, np.sqrt(band[0] * band[1]), "saddle layer", fontsize=5.5, color=C2, va="center", ha="right")
    ax.set_yscale("log")
    ax.set_xlim(0.004, 0.039)
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel(r"$\delta$", labelpad=1)
    ax.set_ylabel(r"rate per unit $t$", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.0, handlelength=1.0, borderaxespad=0.1, labelspacing=0.2)
    panel(ax, "f")
    save(fig, "fig8.pdf")
    C = json.load(open(os.path.join(d, "collapse_delay.json")))
    out["collapse_delay"] = dict(C["summary"], r0=C["r0"], r1=C["r1"], md=C["md"], tmax=C["tmax"])
    return out


def table_edge():
    """Table 1: strongly chaotic starts near the section's ends, and two controls (bcr_survey_checks.py)."""
    Ck = json.load(open(os.path.join(RES, "bcr_section", "survey_checks.json")))
    f3 = lambda x: f"{x:.3f}"
    lines, vals = {False: [], True: []}, {}
    for row in Ck["rows"]:
        dl, ctl = row["delta"], row["control"]
        blk = {s: [x for x in row["tangent"][s]["blocks"] if x is not None] for s in ("default", "tight")}
        if ctl:  # blocks 2-4, after the initial transient
            rate = {
                s: r"$|\cdot| \le " + f"{max(abs(x) for x in b[1:]) * 1e4:.1f}" + r"\times10^{-4}$"
                for s, b in blk.items()
            }
        else:
            rate = {s: f"{f3(min(b))}--{f3(max(b))}" for s, b in blk.items()}
        name = row["label"].replace("small R", "small $R$").replace("large R", "large $R$")
        name = "control" if ctl else name
        R = f"{row['R']:.4f}".rstrip("0")
        lines[ctl].append(
            f"{dl:.3f} & {name} & ({R}, {3 * row['md']:.3f}) & {100 * row['crossings']['overtaking_share']:.0f}\\% & "
            f"{rate['default']} & {rate['tight']} \\\\"
        )
        vals[f"{dl:g}|{row['label']}"] = dict(
            overtaking=row["crossings"]["overtaking_share"],
            default=[min(blk["default"]), max(blk["default"])],
            tight=[min(blk["tight"]), max(blk["tight"])],
            headon_near_ends=[row["crossings"][k] for k in ("near_small_end", "middle", "near_large_end")],
            end_switches=row["crossings"]["end_switches"],
            crossings_per_return=row["tangent"]["default"]["crossings_per_return"],
            status=[row["tangent"]["default"]["status"], row["tangent"]["tight"]["status"], row["crossings"]["status"]],
        )
    tips = [v for k, v in vals.items() if "control" not in k]
    vals["tips"] = dict(
        overtaking=[min(v["overtaking"] for v in tips), max(v["overtaking"] for v in tips)],
        blocks=[
            min(min(v["default"][0], v["tight"][0]) for v in tips),
            max(max(v["default"][1], v["tight"][1]) for v in tips),
        ],
    )
    tex = (
        r"""\begin{table}[!hbp]
\centering\small
\caption{Strongly chaotic starts near the ends of the section, and two controls.}
\label{tab:edge}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llllcc}
\toprule
 & & & & \multicolumn{2}{c}{rate} \\
\cmidrule(lr){5-6}
$\delta$ & end & $(R, 3\Mmid/\delta)$ & overtaking & default & tighter \\
\midrule
"""
        + "\n".join(lines[False] + [r"\midrule"] + lines[True])
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    open(os.path.join(OUT, "table_edge.tex"), "w").write(tex)
    print("wrote table_edge.tex")
    return vals


# ------------------------------------------------------------------ Fig. 9: validation against BCR
# (collapse threshold, crossing ladder, and critical scaling)
def fig_bcr():
    V = json.load(open(os.path.join(RES, "paper", "bcr_validation.json")))
    Cc = json.load(open(os.path.join(RES, "paper", "bcr_critical.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(DW, 2.2))
    thr = 0.5 * sum(V["threshold"])
    a.axvspan(0.030, thr, color=GRID, lw=0)
    a.text(
        0.0322,
        6.5,
        r"no collapse by boundary time $10^3$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK2,
    )
    col = [s for s in V["scan"] if s["status"] == "horizon_approach"]
    a.plot([s["delta"] for s in col], [s["crossings"] for s in col], "o", ms=2.0, color=C1)
    a.axvline(0.0370, color=INK2, lw=0.8, ls=(0, (3, 2)))
    a.text(0.0352, 6.5, "BCR: 0.0370", rotation=90, ha="center", va="center", fontsize=6.5, color=INK2)
    a.set_xlim(0.030, 0.090)
    a.set_ylim(-0.8, 14)
    a.set_yticks([0, 1, 3, 5, 7, 9, 11, 13])
    a.set_xlabel(r"$\delta$ (BCR-T2 data)")
    a.set_ylabel("crossings before collapse")
    panel(a, "a")
    eps = np.array(Cc["eps"])
    R = Cc["runs"]
    sel = [(e, r["M_BH"]) for e, r in zip(eps, R) if r.get("crossings") == 3 and r.get("trapped_rank") == 0]
    e = np.array([q[0] for q in sel])
    M = np.array([q[1] for q in sel])
    f = lambda x, M0, A, g: M0 + A * x**g
    s3 = e <= 3e-6
    p, cv = curve_fit(f, e[s3], M[s3], p0=[M.min() - 1e-6, 1.0, 0.9], maxfev=50000)
    b.loglog(e, M - p[0], "o", ms=2.6, color=C1, label="inner-shell collapse after 3 crossings")
    xx = np.geomspace(e.min(), 3e-6, 50)
    b.loglog(
        xx,
        p[1] * xx ** p[2],
        color=INK2,
        lw=0.9,
        label=rf"fit, $\gamma = {p[2]:.2f}$ ($\varepsilon \leq 3\times10^{{-6}}$)",
    )
    b.set_xlabel(r"$\varepsilon = \delta_3^* - \delta$")
    b.set_ylabel(r"$M_{\rm BH} - M_c$")
    b.legend(loc="upper left")
    panel(b, "b")
    save(fig, "fig9.pdf")
    return dict(gamma=p[2], gamma_err=float(np.sqrt(cv[2, 2])), delta3=Cc["delta3_star"], thr=thr)


if __name__ == "__main__":
    nums = {}
    for f in (
        fig_crossings,
        fig_bcr_lyapunov,
        fig_bcr_section,
        fig_bcr_delta,
        fig_bcr_layer,
        fig_bcr_layer_delta,
        fig_bcr_overview,
        fig_bcr_survey,
        fig_bcr,
        table_edge,
        table_convergence,
        table_refinement,
        wba_numbers,
        domain_clock_numbers,
        ends_followup_numbers,
        chart_check_numbers,
        family_long_numbers,
    ):
        nums[f.__name__] = f()
    # the numbers quoted in the text, for checking against it
    json.dump(nums, open(os.path.join(OUT, "figure_numbers.json"), "w"), indent=1, default=float)
