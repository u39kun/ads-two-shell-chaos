"""Machine check of every algebraic step in the crossing part of the invariant-measure proof.
Model (Israel junction for spherical shells, transparent crossing): in the 2D radial tangent space at the
crossing event, grad R jumps by -(mu/R) n across each shell; u_A, u_B (hence n_A, n_B) are continuous.
Light-cone components, metric ds^2 = -dx+ dx-. Shell A inner before the crossing, B outer."""

import sympy as sp, random

p, q, b = sp.symbols("p q b", real=True)  # grad R in the inner region; rapidity of B (A's set to 0)
al, be = sp.symbols("alpha beta", positive=True)  # mu_A/R, mu_B/R
dot = lambda w, z: -(w[0] * z[1] + w[1] * z[0]) / 2
add = lambda w, z, c=1: (w[0] + c * z[0], w[1] + c * z[1])
uA, nA = (1, 1), (1, -1)
uB, nB = (sp.exp(b), sp.exp(-b)), (sp.exp(b), -sp.exp(-b))
g_in = (p, q)
g_mid = add(g_in, nA, -al)  # across A (outward)
g_out = add(g_mid, nB, -be)  # across B
g_new = add(g_in, nB, -be)  # after: B borders in and mid'
f = lambda g: dot(g, g)
fin, fmid, fout, fnew = map(f, (g_in, g_mid, g_out, g_new))
# shell quantities in the middle region before the crossing
vA, vB = dot(uA, g_mid), dot(uB, g_mid)
bA, bB = dot(nA, g_mid), dot(nB, g_mid)
gam = (bA * bB - vA * vB) / fmid
S = (bB * vA - bA * vB) / fmid
# after the crossing (middle region mid')
vA2, vB2 = dot(uA, g_out), dot(uB, g_in)
bA2, bB2 = dot(nA, g_new), dot(nB, g_new)
S2 = (bB2 * vA2 - bA2 * vB2) / fnew

checks = {}
checks["junction A: beta(in) - beta(mid) = mu_A/R"] = dot(nA, g_in) - dot(nA, g_mid) - al
checks["junction B: beta(mid) - beta(out) = mu_B/R"] = dot(nB, g_mid) - dot(nB, g_out) - be
checks["v continuous across own shell (A)"] = dot(uA, g_in) - vA
checks["mass update: f_new = f_in + f_out - f_mid - 2 (mu_A mu_B/R^2) gamma"] = fnew - (
    fin + fout - fmid - 2 * al * be * gam
)
checks["velocity update A: v_A' = v_A - (mu_B/R) S"] = vA2 - (vA - be * S)
checks["velocity update B: v_B' = v_B - (mu_A/R) S"] = vB2 - (vB - al * S)
checks["S unchanged by the crossing"] = S2 - S
X = lambda v, bb, ff: v * ff / bb
checks["g_in = 1/X_B - 1/X_A = S/(v_A v_B)"] = (1 / X(vB, bB, fmid) - 1 / X(vA, bA, fmid)) - S / (vA * vB)
checks["g_out = 1/X_B' - 1/X_A' = S/(v_A' v_B')"] = (1 / X(vB2, bB2, fnew) - 1 / X(vA2, bA2, fnew)) - S2 / (vA2 * vB2)
# dM'/dM at fixed R, M_in, M_out  (df = -(2/R) dM in every region, so dM'/dM = df_new/df_mid along the constraint curve)
Xs = [p, q, b]
grad = lambda h: sp.Matrix([sp.diff(h, x) for x in Xs])
T = grad(fin).cross(grad(fout))
dMdM = T.dot(grad(fnew)) / T.dot(grad(fmid))
checks["dM'/dM = -v_A' v_B' / (v_A v_B)"] = dMdM + vA2 * vB2 / (vA * vB)
# Ida & Nakao, Prog. Theor. Phys. 101, 989 (1999), Eq. (3.19): the post-crossing metric function f4, equivalent to
# their Eq. (3.18) for the mass m4 (nu = mu_A/R for the inner shell, nu* = mu_B/R for the outer)
ni, no = al, be
f1, f2, f3 = fin, fmid, fout
IN = (
    (ni**2 + no**2) / 2
    + (f1 + f3 - f2) / 2
    + (ni**2 - f1) * (no**2 - f3) / (2 * f2)
    + (2 * ni * vA) * (2 * no * vB) / (2 * f2)
)  # eps*sqrt(...) = 2 nu v (checked separately below)
checks["Ida-Nakao f4 equals f_new"] = IN - fnew
checks["Ida-Nakao radicand = (2 nu_i v_A)^2"] = (ni**4 - 2 * (f1 + f2) * ni**2 + (f1 - f2) ** 2) - (2 * ni * vA) ** 2
checks["Ida-Nakao radicand = (2 nu_o v_B)^2"] = (no**4 - 2 * (f2 + f3) * no**2 + (f2 - f3) ** 2) - (2 * no * vB) ** 2

allok = True
sample = random.Random(0)  # seeded, so the numeric residuals printed below are the same on every run
for name, expr in checks.items():
    simp = sp.simplify(sp.together(sp.expand(expr)))
    worst = 0
    for _ in range(50):
        vals = {
            p: sample.uniform(0.5, 3),
            q: -sample.uniform(0.5, 3),
            b: sample.uniform(-2, 2),
            al: sample.uniform(0.01, 0.3),
            be: sample.uniform(0.01, 0.3),
        }
        if min(fmid.subs(vals), fout.subs(vals), fnew.subs(vals)) <= 0:
            continue
        worst = max(worst, abs(sp.N(expr.subs(vals), 40)))
    ok = simp == 0
    allok &= ok
    print(
        f"[{'OK ' if ok else 'FAIL'}] {name}   (symbolic: {simp if not ok else 0}; numeric max |residual| {sp.N(worst, 3)})"
    )
print("ALL SYMBOLIC CHECKS PASS" if allok else "SOME CHECK FAILED")

# The hand derivation of dM'/dM in App. A of the paper, step by step, at fixed R, M_in, and M_out. Variables: f = f(R; M_mid)
# in the middle region, nu_i = mu_i/R, and f_in, f_out; head-on crossing with A inside moving out (v_A > 0 > v_B).
# These steps involve square roots, so they are checked numerically at 40 significant digits.
F, nA, nB, Fi, Fo = sp.symbols("f nu_A nu_B f_in f_out", positive=True)
bA_ = (Fi - F - nA**2) / (2 * nA)  # junction condition of A: sqrt(v_A^2 + f_in) - sqrt(v_A^2 + f) = nu_A
bB_ = (F - Fo + nB**2) / (2 * nB)  # junction condition of B: sqrt(v_B^2 + f) - sqrt(v_B^2 + f_out) = nu_B
vA_, vB_ = sp.sqrt(bA_**2 - F), -sp.sqrt(bB_**2 - F)
gam_, S_ = (bA_ * bB_ - vA_ * vB_) / F, (bB_ * vA_ - bA_ * vB_) / F
steps = {
    "App. A: dv_A/df = -(beta_A + nu_A)/(2 nu_A v_A)": sp.diff(vA_, F) + (bA_ + nA) / (2 * nA * vA_),
    "App. A: dv_B/df = (beta_B - nu_B)/(2 nu_B v_B)": sp.diff(vB_, F) - (bB_ - nB) / (2 * nB * vB_),
    "App. A: f S^2 = v_A^2 + v_B^2 - 2 gamma v_A v_B": F * S_**2 - (vA_**2 + vB_**2 - 2 * gam_ * vA_ * vB_),
    "App. A: dgamma/df = -S (nu_A v_A + nu_B v_B - nu_A nu_B S)/(2 nu_A nu_B v_A v_B)": sp.diff(gam_, F)
    + S_ * (nA * vA_ + nB * vB_ - nA * nB * S_) / (2 * nA * nB * vA_ * vB_),
    "App. A: df'/df = -1 - 2 nu_A nu_B dgamma/df = -v_A' v_B'/(v_A v_B)": sp.diff(Fi + Fo - F - 2 * nA * nB * gam_, F)
    + (vA_ - nB * S_) * (vB_ - nA * S_) / (vA_ * vB_),
}
rng = random.Random(1)
pts = []
while len(pts) < 50:
    q = lambda lo, hi: sp.Rational(rng.randint(int(lo * 10**6), int(hi * 10**6)), 10**6)  # exact inputs
    v = {F: q(0.3, 2.0), nA: q(0.01, 0.3), nB: q(0.01, 0.3)}
    v[Fi], v[Fo] = v[F] + q(0.1, 1.5), v[F] - q(0.05, 0.25)
    if v[Fo] > 0 and bA_.subs(v) > 0 and bB_.subs(v) > 0 and (bA_**2 - F).subs(v) > 0 and (bB_**2 - F).subs(v) > 0:
        pts.append(v)
stepok = True
for name, expr in steps.items():
    worst = max(abs(sp.N(expr.subs(v), 40)) for v in pts)
    ok = worst < 1e-30
    stepok &= ok
    print(f"[{'OK ' if ok else 'FAIL'}] {name}   (numeric max |residual| at 40 digits {sp.N(worst, 3)})")
print("ALL STEPS OF THE HAND DERIVATION PASS" if stepok else "SOME STEP FAILED")
