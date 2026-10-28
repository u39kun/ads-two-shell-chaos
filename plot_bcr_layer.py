"""Chaotic layer of the two-shell map at BCR's Fig. 5 parameters. usage: python plot_bcr_layer.py DIR OUT.png"""

import sys, os, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
A = json.load(open(os.path.join(d, "layer_scan.json")))
B = json.load(open(os.path.join(d, "layer_refine.json")))
C = json.load(open(os.path.join(d, "bcr_checks.json")))
LG = json.load(open(os.path.join(d, "bcr_benettin_long.json")))
runs = {}
for r in A["runs"] + B["runs"]:
    key = (r["direction"], round(r["dist"], 12))
    if key not in runs or r["lambda_N"][-1][0] > runs[key]["lambda_N"][-1][0]:
        runs[key] = r
plt.rcParams.update({"font.size": 8})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw=dict(wspace=0.32))
L = np.array(C["long_points"])
a.plot(L[:, 0], L[:, 1] * 1e3, ".", ms=0.3, color="tab:blue", label="BCR's example")
for (dr, ds), r in runs.items():
    p = np.array(r["points"])
    if ds == 1e-4 and dr in ("R+", "M+"):
        a.plot(
            p[:, 0],
            p[:, 1] * 1e3,
            ".",
            ms=0.4,
            color="tab:red",
            label="started $10^{-4}$ from saddle" if dr == "R+" else None,
        )
    if ds == 0.07 and dr in ("R+", "R-"):
        a.plot(
            p[:, 0],
            p[:, 1] * 1e3,
            ".",
            ms=0.4,
            color="tab:green",
            label="started 0.07 from saddle" if dr == "R+" else None,
        )
sad = A["saddle"]
a.plot(sad[0], sad[1] * 1e3, "kx", ms=5)
a.set_xlabel("crossing radius $R$")
a.set_ylabel(r"$M_{\rm mid}$ ($10^{-3}$)")
a.legend(fontsize=6.5, loc="upper center", markerscale=8)
a.set_title("(a) section: chaotic layer (red), regular orbits", fontsize=8)
marks = {"R+": "o", "R-": "s", "M+": "^", "M-": "v"}
for dr in marks:
    pts = sorted([(ds, r) for (d2, ds), r in runs.items() if d2 == dr])
    xs = [ds for ds, _ in pts]
    ys = [r["lambda_N"][-1][1] for _, r in pts]
    b.loglog(xs, ys, marks[dr] + "-", ms=3, lw=0.8, label=dr)
b.axhline(np.log(5000) / 5000, color="k", ls=":", lw=0.8)
b.text(1.5e-9, np.log(5000) / 5000 * 1.15, "regular: ~ln N/N", fontsize=6.5)
rt = 4.9
b.axhline(0.04 * rt, color="tab:red", ls="--", lw=0.8)
b.text(1.5e-9, 0.04 * rt * 1.1, "BCR: 0.04 per unit time", fontsize=6.5, color="tab:red")
b.axvspan(0.145, 0.18, color="tab:blue", alpha=0.2)
b.text(0.16, 0.012, "BCR's example", fontsize=6.5, color="tab:blue", ha="center", rotation=90)
b.set_xlabel("start distance from the saddle (chart units)")
b.set_ylabel(r"$\lambda_N$ per return, $N$ = 4000 or 6000")
b.set_ylim(3e-4, 0.5)
b.set_xlim(5e-10, 0.4)
b.legend(fontsize=6.5, loc="center left")
b.set_title("(b) exponent against distance", fontsize=8)
sel = [
    (("R+", 1e-4), "tab:red", "layer, $10^{-4}$"),
    (("R+", 0.01), "tab:orange", "near saddle, $10^{-2}$"),
    (("R+", 0.07), "tab:green", "regular, 0.07"),
]
for key, col, lab in sel:
    k = np.array(runs[key]["lambda_N"])
    c.loglog(k[:, 0], k[:, 1], "o-", ms=3, color=col, label=lab)
k = np.array([r for r in LG if r["d0"] == 1e-9][0]["lambda_N"])
c.loglog(k[:, 0], k[:, 1], "o-", ms=3, color="tab:blue", label="BCR's example")
n = np.logspace(2, 4.3, 10)
c.loglog(n, 1.2 * np.log(n) / n, "k:", lw=0.8, label=r"$\propto \ln N/N$")
c.set_xlabel("returns $N$")
c.set_ylabel(r"$\lambda_N$ per return")
c.legend(fontsize=6.5)
c.set_title("(c) exponent against $N$", fontsize=8)
plt.savefig(sys.argv[2], dpi=130, bbox_inches="tight")
print("wrote", sys.argv[2])
