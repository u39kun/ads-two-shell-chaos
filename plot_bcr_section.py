"""Figure for the BCR section study: (a) section with background orbits, (b) zoom on the BCR curve at its right
waist, (c) two-orbit finite-time exponents. usage: python plot_bcr_section.py DIR OUT.png"""

import sys, json, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
S = json.load(open(os.path.join(d, "bcr_section.json")))
C = json.load(open(os.path.join(d, "bcr_checks.json")))
LG = json.load(open(os.path.join(d, "bcr_benettin_long.json")))
plt.rcParams.update({"font.size": 8})
fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(wspace=0.32))
for bg in S["background"]:
    p = np.array(bg["points"])
    if len(p):
        a.plot(p[:, 0], p[:, 1] * 1e3, ",", color="0.6")
L = np.array(C["long_points"])
a.plot(L[:, 0], L[:, 1] * 1e3, ".", ms=0.5, color="tab:blue")
for fp in S["fixed_points"]:
    a.plot(fp["x"][0], fp["x"][1] * 1e3, "x" if fp["kind"] == "hyperbolic" else "o", color="k", ms=5)
a.add_patch(plt.Rectangle((0.97, 14.6), 0.06, 1.0, fill=False, ec="tab:red", lw=0.8))
a.set_xlabel("crossing radius $R$")
a.set_ylabel(r"$M_{\rm mid}$ ($10^{-3}$)")
a.set_title("(a) section; blue: BCR Fig. 5 orbit, $t\\leq10^5$", fontsize=8)
k = (L[:, 1] * 1e3 > 14.6) & (L[:, 1] * 1e3 < 15.6) & (L[:, 0] > 0.9)
b.plot(L[k, 1] * 1e3, L[k, 0], ".", ms=1.2, color="tab:blue")
x = L[k, 1] * 1e3
y = L[k, 0]
c0 = np.median(x)
kk0 = np.abs(x - c0) < 0.05
p = np.polyfit(x[kk0] - c0, y[kk0], 8)
r = y[kk0] - np.polyval(p, x[kk0] - c0)
b.set_xlabel(r"$M_{\rm mid}$ ($10^{-3}$)")
b.set_ylabel("$R$")
b.set_title(
    f"(b) zoom: {k.sum()} crossings; in a 0.1-wide strip\nthey lie within {np.sqrt((r**2).mean()):.0e} of a smooth curve",
    fontsize=8,
)
runs = [r for r in C["benettin"]] + LG
seen = set()
for r in runs:
    key = (r["name"], r["d0"])
    if key in seen:
        continue
    seen.add(key)
    kk = np.array(r["lambda_N"])
    if r["name"] == "BCR":
        if r["d0"] > 5e-7:
            continue
        c.loglog(kk[:, 0], kk[:, 1], "o-", ms=2.5, lw=0.9, label=f"BCR orbit, $d_0={r['d0']:.0e}$")
    elif r["name"] == "center":
        c.loglog(kk[:, 0], kk[:, 1], "s--", ms=2.5, lw=0.9, color="0.4", label="regular orbit near a center")
n = np.logspace(2, 4.4, 20)
c.loglog(n, 1.2 * np.log(n) / n, "k:", lw=0.8, label=r"$\propto \ln N/N$")
c.axhline(0.04 * 4.91, color="tab:red", lw=0.8, ls="--")
c.text(110, 0.04 * 4.91 * 1.1, "BCR rate 0.04, per return", color="tab:red", fontsize=7)
c.set_xlabel("returns $N$")
c.set_ylabel(r"$\lambda_N$ per return")
c.legend(fontsize=6.5, loc="lower left")
c.set_title("(c) two-orbit finite-time exponent", fontsize=8)
plt.savefig(sys.argv[2], dpi=130, bbox_inches="tight")
print("wrote", sys.argv[2])
