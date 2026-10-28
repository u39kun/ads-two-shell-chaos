"""Addendum: BCR's Fig. 5 (Brito, Cardoso & Rocha, PRD 94, 024003) reproduced from the BCR-5 orbit (Fig. 10, App. B).

The orbit of Sec. 3.3: region masses 0, 0.015, 0.03, shells with m = 0.00815 and w = 0.2, both at R_i = 1, the inner
shell contracting and the outer expanding (the initially separating assignment, bcr_lyapunov.initial with start D).
Reference solver n_shells.py with relative tolerance 1e-12, absolute tolerance 1e-14 and maximum step 0.02, in the
static time t of the region between the shells (BCR's evolution variable), to t = 1000. Both shells are sampled every
0.02 from the dense output; dR/dt = Rdot f / sqrt(Rdot^2 + f) with Rdot = dR/dtau and f = 1 - 2 M_mid/R + R^2, as in
independent_check.py. As in BCR's figure, R1 and R2 are the outermost and innermost radius by position, not by
material shell.
(a) (R1, R2). (b) (dR/dt, R) for both, with BCR's two inset windows of the outermost shell, drawn as lines.
The opposite assignment is this run backward in time and gives the same point set (Sec. 3.3 and App. B), so it is
not run here.

Prints the numbers quoted in App. B.
TMAX (default 1000) sets the run length; App. B also quotes a run to 5000, by which the points fill the region.
usage (from code/): python addendum/bcr_fig5.py OUT.pdf [TMAX]"""

import os, sys, time, platform, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bcr_lyapunov as BL  # sets SOLVER=reference: dense output
import numpy as np
import scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# embedding the Computer Modern fonts makes fontTools warn about their header timestamps; harmless
logging.getLogger("fontTools").setLevel(logging.ERROR)
DT = 0.02
KW = dict(anchor=1, rtol=1e-12, atol=1e-14, max_step=0.02, max_events=10**6, collect=True)
INSETS = ((1.85, 2.15, 2.0, 2.3), (2.0, 2.2, 2.0, 2.05))  # BCR's windows: dR/dt range, R range

# style of make_figures_bcr.py (not imported: it creates its output folder on import)
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9893"
DW = 7.0
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
        "savefig.dpi": 600,
    }
)
DOT = dict(marker="o", ms=0.35, mew=0, ls="none", rasterized=True)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right", color=INK)


def orbit(tmax):
    """Rows t, R1, dR1/dt, R2, dR2/dt (outermost, innermost) every DT, and the run."""
    masses, y0, order = BL.initial(1.0, "D")
    run = BL.N.simulate([BL.A, BL.B], masses, y0, order=order, tmax=tmax, **KW)
    rows = []
    for seg in run["segments"]:
        t = np.arange(np.ceil(seg["t"][0] / DT - 1e-9) * DT, seg["t"][-1], DT)
        if not len(t):
            continue
        y, mid = seg["dense"](t), seg["masses"][1]
        R, Rdot = np.stack([y[0], y[2]]), np.stack([y[1], y[3]])
        f = 1 - 2 * mid / R + R**2
        X = Rdot * f / np.sqrt(Rdot**2 + f)
        out, k = np.argmax(R, axis=0), np.arange(len(t))
        rows.append(np.column_stack([t, R[out, k], X[out, k], R[1 - out, k], X[1 - out, k]]))
    return np.concatenate(rows), run


def figure(d, path):
    t, R1, X1, R2, X2 = d.T
    fig, (a, b) = plt.subplots(1, 2, figsize=(DW, 2.6), gridspec_kw=dict(wspace=0.28))
    a.plot([0, 1.5], [0, 1.5], ":", color=MUTED, lw=0.7)
    a.plot(R1, R2, color=INK, **DOT)
    a.set(xlim=(0, 4), ylim=(0, 1.5), xlabel=r"$R_1$, outermost radius", ylabel=r"$R_2$, innermost radius")
    panel(a, "a")
    b.plot(X1, R1, color=C1, label="outermost", **DOT)
    b.plot(X2, R2, color=C2, label="innermost", **DOT)
    b.set(xlim=(-5, 5), ylim=(0, 4), xlabel=r"$dR/dt$", ylabel=r"radius $R$")
    b.legend(loc="upper left", markerscale=8, handlelength=0.8)
    for (x0, x1, y0, y1), box in zip(INSETS, ([0.08, 0.07, 0.2, 0.25], [0.75, 0.07, 0.2, 0.25])):
        ins = b.inset_axes(box)
        # the outermost shell as lines, as in BCR's insets; broken where the curve leaves the window or the
        # outermost shell changes at a crossing (a jump in dR/dt)
        keep = (X1 > x0 - 0.1) & (X1 < x1 + 0.1) & (R1 > y0 - 0.1) & (R1 < y1 + 0.1)
        cut = ~keep | np.r_[False, np.abs(np.diff(X1)) > 0.5]
        ins.plot(np.where(cut, np.nan, X1), np.where(cut, np.nan, R1), color=C1, lw=0.25, rasterized=True)
        ins.set(xlim=(x0, x1), ylim=(y0, y1))
        ins.set_xticks([x0, x1])
        ins.set_yticks([y0, y1])
        ins.tick_params(labelsize=5.5, length=2, pad=1)
        for s in ins.spines.values():
            s.set_visible(True)
    panel(b, "b")
    fig.savefig(path, metadata={"CreationDate": None})  # no timestamp: reruns give identical files
    plt.close(fig)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    path = sys.argv[1]
    tmax = float(sys.argv[2]) if len(sys.argv) == 3 else 1000.0
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
          f"matplotlib {matplotlib.__version__}")
    print(f"masses {BL.MASSES}, m = {BL.M}, w = {BL.W}, R_i = 1, inner shell contracting, outer expanding")
    print(f"rtol {KW['rtol']:g}, atol {KW['atol']:g}, max_step {KW['max_step']:g}, middle-region time to {tmax:g}, "
          f"sampled every {DT:g}")
    t0 = time.time()
    d, run = orbit(tmax)
    t_run = time.time() - t0
    t, R1, X1, R2, X2 = d.T
    print(f"status {run['status']}, t_end {run['t_end']:g}, crossings {len(run['events'])}, samples {len(t)}")
    print(f"R1 (outermost)      {R1.min():.4f} .. {R1.max():.4f}")
    print(f"R2 (innermost)      {R2.min():.4f} .. {R2.max():.4f}")
    print(f"dR1/dt (outermost) {X1.min():+.4f} .. {X1.max():+.4f}")
    print(f"dR2/dt (innermost) {X2.min():+.4f} .. {X2.max():+.4f}")
    figure(d, path)
    print(f"wrote {os.path.basename(path)}; integration {t_run:.0f} s, total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
