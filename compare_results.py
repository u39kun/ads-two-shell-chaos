"""Compare a fresh run of the pipeline (run_all.sh) with the shipped results.

1. Result files: for every JSON under REF (per-run caches excluded), compare all numbers with the same file under NEW
   and report the largest relative difference and any change of structure (keys, list lengths). Run times
   (fields named ...seconds) are ignored.
2. Quoted numbers: draw the figures from each results folder with make_figures_bcr.py (into OUTDIR/reference and
   OUTDIR/rerun, so the figures can be compared by eye) and compare the two figure_numbers.json files, which hold
   the numbers the paper quotes with its figures and tables. The numbers that differ are sorted by kind (chaotic
   orbits, two-orbit estimates, exponents of regular orbits, everything else), starts that change classification are
   listed, and the largest differences are labeled with where they appear in the paper and shown in its units.

Along regular orbits (BCR's example, the type (ii) family, the nudge test, and the validation) a different machine or
library build changes results only slightly, so these should agree closely. Their finite-time exponents are close to
zero, though, so a small absolute change can be a large relative one, and two-orbit estimates near their noise floor
change more. Orbits in the chaotic layer follow different trajectories after a few hundred returns, so their exponents,
and anything chosen from them (such as the orbits that tangent_extend picks), agree only statistically.
usage: python compare_results.py REF NEW [--out figs_compare] [--top 40]"""

import os, sys, json, math, re, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(0.0, "identical"), (1e-9, "<1e-9"), (1e-6, "<1e-6"), (1e-3, "<1e-3"), (1e-2, "<1e-2"), (math.inf, ">=1e-2")]


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


FLOOR = 1e-12  # relative differences are taken against max(|a|, |b|, FLOOR): rounding-level values (residuals
# of order 1e-15) then do not show up as large relative differences
TIMING = re.compile(r"(^|/|_)seconds$")  # run times (e.g. runs[3]/seconds) are not results


def compare(a, b):
    """Returns ({path: (ref, new, rel)} for differing numbers, [paths only in one of them], number of shared leaves).
    Run-time fields are ignored."""
    A = {p: v for p, v in leaves(a) if not TIMING.search(p)}
    B = {p: v for p, v in leaves(b) if not TIMING.search(p)}
    diffs, n = {}, 0
    for p, x in A.items():
        if p not in B:
            continue
        n += 1
        y = B[p]
        num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        if num(x) and num(y):
            if x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                continue
            diffs[p] = (x, y, abs(x - y) / max(abs(x), abs(y), FLOOR))
        elif x != y:
            diffs[p] = (x, y, math.inf)
    only = sorted(set(A) ^ set(B))
    return diffs, only, n


def result_files(top):
    """Paths, relative to top, of all JSON result files, without the per-case cache folders (*_parts)."""
    out = set()
    for root, dirs, files in os.walk(top):
        dirs[:] = [d for d in dirs if not d.endswith("_parts")]
        out.update(os.path.relpath(os.path.join(root, f), top) for f in files if f.endswith(".json"))
    return out


THR = 3e-3  # late-block exponent per return above which an orbit counts as strongly chaotic (Sec. 3.2 of the paper)
KINDS = [
    ("chaotic", "exponents, reach, and return times of chaotic orbits (expected to differ)"),
    ("diff", "two-orbit and finite-difference estimates on regular orbits (noise floor)"),
    ("var", "variational-tangent exponents of regular orbits (close to zero)"),
    ("other", "everything else (fixed points, nudge test, geometry, validation)"),
]


def plain(s):
    """A LaTeX table label as plain text: '$d_0 = 10^{-8}$' -> 'd0 = 1e-8'."""
    s = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", s)
    for a, b in (("$-$", "-"), ("$", ""), ("d_0", "d0"), ("\\,", " ")):
        s = s.replace(a, b)
    return s


def describe(p, R, N):
    """For a path in figure_numbers.json: (where the number appears in the paper, kind, scale, unit). Values are shown
    multiplied by scale: table cells in their printed units (1e-3 per return), other exponents per return."""
    TAB, PR = "units of 1e-3 per return, as printed", "per return"
    sec, *rest = p.split("/")[1:]
    try:
        if sec == "table_convergence":
            name, col, dist = rest[0].split("|")
            where = f'Table 3, row "{plain(name)}", column "' + (
                "example" if col == "BCR" else f"{col}, {float(dist):g}"
            )
            var = R[sec].get(f"variational tangent|{col}|{dist}")
            if var is not None and var > THR:
                return where + '" (chaotic orbit)', "chaotic", 1e3, TAB
            if name == "variational tangent":
                return where + '" (regular orbit, variational tangent)', "var", 1e3, TAB
            return where + '" (regular orbit, two-orbit estimate)', "diff", 1e3, TAB
        if sec == "table_refinement":
            orb, rt, ms = rest[0].split("|")
            col = {"R+ 1e-4": "R+, 1e-4 (layer)", "R+ 1e-2": "R+, 1e-2", "BCR's example": "example"}.get(orb, orb)
            where = f'Table 2, row "rtol = {rt}, max step {ms}", column "{col}"'
            if orb == "R+ 1e-4":
                return where + " (chaotic orbit)", "chaotic", 1e3, TAB
            return where + " (regular orbit, variational tangent)", "var", 1e3, TAB
        if sec == "fig_bcr_layer":
            m = re.fullmatch(r"(chaotic|regular)\[(\d+)\]\[([23])\]", rest[0])
            if m:
                row = R[sec][m.group(1)][int(m.group(2))]
                what = "late-block exponent" if m.group(3) == "2" else "exponent after 6000 returns"
                where = f"Fig. 3c, start {row[0]} {row[1]:g} from the saddle: {what}"
                if m.group(1) == "chaotic":
                    return where + " (chaotic orbit)", "chaotic", 1, PR
                return where + " (regular orbit, variational tangent)", "var", 1, PR
            if rest[0] == "long":
                L = R[sec]["long"][rest[1]]
                if rest[2] == "late":
                    what = f"exponent over returns {rest[3]}"
                else:
                    k = int(re.match(r"lambda_N\[(\d+)\]", rest[2]).group(1))
                    what = f"exponent after {L['lambda_N'][k][0]:g} returns"
                where = f"Fig. 3d, long run of the start {rest[1]}: {what}"
                if L["late"]["2000-6000"] > THR:
                    return where + " (chaotic orbit)", "chaotic", 1, PR
                return where + " (regular orbit, variational tangent)", "var", 1, PR
            if re.match(r"(extent_along_cut|chaotic_late_range|return_time_chaotic)", rest[0]):
                return f"Sec. 3.2, chaotic orbits: {rest[0].replace('_', ' ')}", "chaotic", 1, ""
            if rest[0] == "regular_along_cut":
                return (
                    f"Sec. 3.2, regular starts 1e-2 from the saddle ({rest[1].split('[')[0]}): distance of crossings from the saddle near its line",
                    "other",
                    1,
                    "",
                )
            if rest[0].startswith("regular_late_range"):
                return "Sec. 3.2, range of late-block exponents of the regular starts", "var", 1, PR
            if rest[0].startswith("example"):
                return "Sec. 3.2, BCR's example: variational-tangent exponent", "var", 1, PR
        if sec == "fig_bcr_layer_delta":
            dl, side, key = rest
            m = re.fullmatch(r"(late|lam6000)\[(\d+)\]", key)
            if m:
                i, A, B = int(m.group(2)), R[sec][dl][side], N[sec][dl][side]
                what = "late-block exponent" if m.group(1) == "late" else "exponent after 6000 returns"
                where = (
                    f"Fig. 6{'a' if side == '-1' else 'b'}, delta = {dl}, start {A['dist'][i]:g} from the saddle on the "
                    f"{'inner' if side == '-1' else 'outer'} side: {what}"
                )
                if max(A["late"][i], B["late"][i]) > THR:
                    return where + " (chaotic orbit)", "chaotic", 1, PR
                return where + " (regular orbit, variational tangent)", "var", 1, PR
            side_txt = (
                f"Fig. 6{'a' if side == '-1' else 'b'}, delta = {dl}, {'inner' if side == '-1' else 'outer'} side"
            )
            m = re.fullmatch(r"extent\[([01])\]", key)
            if m:
                what = "farthest strongly chaotic start" if m.group(1) == "0" else "next start outward"
                return f"{side_txt}: {what} (reach of strong chaos)", "chaotic", 1, ""
            m = re.fullmatch(r"return_time\[(\d+)\]", key)
            if m:
                i, A, B = int(m.group(1)), R[sec][dl][side], N[sec][dl][side]
                where = f"{side_txt}, start {A['dist'][i]:g}: mean time per return"
                if max(A["late"][i], B["late"][i]) > THR:
                    return where + " (chaotic orbit)", "chaotic", 1, ""
                return where + " (regular orbit)", "other", 1, ""
        if sec == "fig_bcr_section" and rest[0].startswith("fixed_points"):
            return "Sec. 3.1, fixed points of the section map", "other", 1, ""
        if sec == "fig_bcr_section":
            m = re.fullmatch(r"lambda_final\[(\d+)\]\[1\]", rest[0])
            if m:
                return (
                    f"Fig. 2c, BCR's example, long two-orbit run {int(m.group(1)) + 1}: final exponent (regular orbit, two-orbit estimate)",
                    "diff",
                    1,
                    PR,
                )
            if rest[0] == "exact_tangent":
                v0 = rest[1].split("]")[0] + "]"
                return f"Fig. 2c, BCR's example, variational tangent from {v0} (regular orbit)", "var", 1, PR
            if rest[0] == "fd_lambda_3000":
                return (
                    f"App. C, finite-difference product ({rest[1].split('[')[0]}), 3000 returns (regular orbit, finite-difference estimate)",
                    "diff",
                    1,
                    PR,
                )
        if sec == "fig_bcr_delta":
            m = re.fullmatch(r"loglog_slope\[(\d+)\]", rest[0])
            if m:
                dl = R[sec]["deltas"][int(m.group(1))]
                return (
                    f"Sec. 4.1, type (ii) at delta = {dl:g}: log-log slope of the twin runs' separation",
                    "other",
                    1,
                    "",
                )
            if rest[0].startswith("slope_range"):
                which = "inner shell expanding" if "_C" in rest[0] else "inner shell contracting"
                return f"Sec. 4.1, type (ii): range of log-log slopes ({which})", "other", 1, ""
            m = re.fullmatch(r"lambda_4000\[(\d+)\]", rest[0])
            if m:
                dl = R[sec]["deltas"][int(m.group(1))]
                return (
                    f"Fig. 5d, type (ii) at delta = {dl:g}: two-orbit exponent after 4000 returns (regular orbit, two-orbit estimate)",
                    "diff",
                    1,
                    PR,
                )
            if rest[0].startswith("lambda_4000_range"):
                which = "inner shell expanding" if "_C" in rest[0] else "inner shell contracting"
                return (
                    f"App. B, type (ii): range of two-orbit exponents after 4000 returns ({which}; regular orbits, two-orbit estimates)",
                    "diff",
                    1,
                    PR,
                )
            if rest[0].startswith("lambda_4000_tangent_range"):
                return (
                    "App. B, type (ii): range of variational-tangent exponents after 4000 returns (regular orbits)",
                    "var",
                    1,
                    PR,
                )
        if sec == "fig_bcr_lyapunov":
            name, clock, eps = rest[0].split("|")
            how = "reconstructed velocities" if name == "reconstruction" else "opposite velocities"
            what = " ".join(rest[1:]).replace("_", " ")
            return f"Fig. 4 (nudge test), {how}, {clock.replace('_', ' ')} clock, eps = {eps}: {what}", "other", 1, ""
    except (KeyError, IndexError, ValueError, AttributeError, TypeError):
        pass
    return p, "other", 1, ""


def classification_changes(R, N):
    """Starts whose late-block exponent lies above the threshold in one run and below it in the other."""
    out = []
    try:
        a = {(r[0], r[1]) for r in R["fig_bcr_layer"]["chaotic"]}
        b = {(r[0], r[1]) for r in N["fig_bcr_layer"]["chaotic"]}
        for k in sorted(a ^ b):
            out.append(
                f"Fig. 3c, start {k[0]} {k[1]:g}: strongly chaotic only in the {'reference' if k in a else 'rerun'}"
            )
        for dl, sides in R["fig_bcr_layer_delta"].items():
            for side, A in sides.items():
                B = N["fig_bcr_layer_delta"][dl][side]
                for i, (x, y) in enumerate(zip(A["late"], B["late"])):
                    if (x > THR) != (y > THR):
                        out.append(
                            f"Fig. 6, delta = {dl}, start {A['dist'][i]:g} on the {'inner' if side == '-1' else 'outer'}"
                            f" side: {x:.2e} in the reference, {y:.2e} in the rerun"
                        )
    except (KeyError, IndexError, TypeError):
        pass
    return out


def band(r):
    for lim, name in BANDS:
        if r <= lim:
            return name
    return BANDS[-1][1]


def figure_numbers(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, RESULTS_DIR=os.path.abspath(results))
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, "make_figures_bcr.py"), outdir], env=env, capture_output=True, text=True
    )
    if p.returncode != 0:
        tail = [l for l in p.stderr.strip().splitlines() if "timestamp seems very low" not in l][-6:]
        print(f"  make_figures_bcr.py failed on {results} (expected if that run is incomplete):")
        print("\n".join("    " + l for l in tail))
        return None
    return json.load(open(os.path.join(outdir, "figure_numbers.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--out", default="figs_compare")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    print(
        "How to read this: processor-dependent compilation and numerical libraries can introduce small floating-point"
    )
    print(
        "differences between machines or versions. Along regular orbits they stay small, so those values agree closely."
    )
    print(
        "Along orbits in the chaotic layer they grow until the runs follow different trajectories, so their exponents"
    )
    print("differ (layer_*, tangent_long, layer_convergence, noise_trace), and so do two-orbit estimates near their")
    print("noise floor. Exponents of regular orbits are close to zero, so a small absolute change there can be a large")
    print(
        "relative one. Section 2 sorts the quoted numbers that differ by kind and labels each with where it appears in"
    )
    print("the paper. See App. C of the paper and the README.\n")
    print(f"1. Result files ({a.ref} vs {a.new})")
    print(
        "  numbers: values compared (run times ignored); differ: how many differ; max rel. diff: |a - b|/max(|a|, |b|, 1e-12);"
    )
    print('  structure: "same", or how many entries (keys or list items) exist in only one of the two files, e.g. a')
    print(
        "  field that the current scripts add but an older shipped file lacks, or a different number of stored points."
    )
    print("  only in: a result file present in one folder only, e.g. a check added after the rerun.")
    RF, NF = result_files(a.ref), result_files(a.new)
    print(f'  {"file":46s} {"numbers":>9s} {"differ":>8s} {"max rel. diff":>14s}  structure')
    for rel in sorted(RF | NF):
        if rel not in NF or rel not in RF:
            print(f"  {rel:46s}  only in {a.ref if rel in RF else a.new}")
            continue
        d, only, n = compare(json.load(open(os.path.join(a.ref, rel))), json.load(open(os.path.join(a.new, rel))))
        mx = max((r for _, _, r in d.values() if r != math.inf), default=0.0)
        text = sum(1 for _, _, r in d.values() if r == math.inf)
        note = ("same" if not only else f"{len(only)} entries") + (f"; {text} text values differ" if text else "")
        print(f"  {rel:46s} {n:9d} {len(d):8d} {mx:14.2e}  {note}")
    print(f"  {len(RF & NF)} result files in both; {len(RF - NF)} only in {a.ref}; {len(NF - RF)} only in {a.new}.")

    print(
        f"\n2. Numbers quoted with the figures and tables (figure_numbers.json; figures in {a.out}/reference and "
        f"{a.out}/rerun)"
    )
    R = figure_numbers(a.ref, os.path.join(a.out, "reference"))
    N = figure_numbers(a.new, os.path.join(a.out, "rerun"))
    if R is None or N is None:
        sys.exit(1)
    d, only, n = compare(R, N)
    print(f'  {"section":22s} {"numbers":>8s} ' + " ".join(f"{b:>9s}" for _, b in BANDS))
    NL = dict(leaves(N))
    for sec in R:
        keys = [p for p, _ in leaves(R[sec], "/" + sec) if p in NL]
        counts = {b: 0 for _, b in BANDS}
        for p in keys:
            counts[band(d[p][2]) if p in d else "identical"] += 1
        print(f"  {sec:22s} {len(keys):8d} " + " ".join(f"{counts[b]:9d}" for _, b in BANDS))
    if only:
        print(
            f"  {len(only)} quoted entries exist in only one of the two (e.g. a different set of orbits chosen, or a table whose input is missing):"
        )
        for p in only[:10]:
            print(f"    {p}")
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    D = {p: (x, y, r, *describe(p, R, N)) for p, (x, y, r) in d.items() if num(x) and num(y)}
    print("\n  What differs, by kind (quoted numbers differing by more than a relative 1e-6):")
    for kind, text in KINDS:
        sel = [v for v in D.values() if v[4] == kind and v[2] > 1e-6]
        note = ""
        if sel and kind in ("diff", "var"):
            note = f"largest absolute difference {max(abs(v[0] - v[1]) for v in sel):.1e} per return"
        elif sel and kind == "other":
            note = f"largest relative difference {max(v[2] for v in sel):.1e}"
        print(f"    {text:76s} {len(sel):5d}  {note}")
    ch = classification_changes(R, N)
    print(f"  Starts classified differently (late-block exponent above or below {THR:g} per return): {len(ch)}")
    for c in ch:
        print(f"    {c}")
    big = sorted(D.items(), key=lambda kv: -kv[1][2])[: a.top]
    if big:
        print(
            f"\n  Largest relative differences (top {len(big)}). Relative difference: |reference - rerun| / max(|reference|,"
            "\n  |rerun|); values from the reference (the paper) and the rerun, with where each appears in the paper."
            "\n  Exponents are per return; Tables 2 and 3 are shown in their printed units of 1e-3 per return."
        )
        for p, (x, y, r, where, kind, scale, unit) in big:
            u = f" ({unit})" if unit else ""
            print(
                f"    {r:8.2e}  {where}\n              reference {x * scale:.3g}, rerun {y * scale:.3g}, "
                f"difference {abs(x - y) * scale:.2g}{u}"
            )


if __name__ == "__main__":
    main()
