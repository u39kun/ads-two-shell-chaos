"""Generate Table 4 from the numbers exported by make_figures_bcr.py.

usage (from code/): python addendum/make_table_criteria.py figs/figure_numbers.json figs/table_criteria.tex
"""

import argparse
import json
from pathlib import Path


def scientific(value):
    """Two significant figures, without trailing zeros in the mantissa."""
    mantissa, exponent = f"{value:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    power = rf"10^{{{int(exponent)}}}"
    return power if mantissa == "1" else rf"{mantissa}\times{power}"


def table_criteria(numbers):
    growth = numbers["ends_followup_numbers"]["linear_growth"]
    radial_threshold = scientific(numbers["fig_bcr_layer"]["threshold"])
    survey_threshold = scientific(numbers["fig_bcr_survey"]["threshold"])
    minimum_rate = scientific(numbers["table_edge"]["tips"]["blocks"][0])
    radial_growth = scientific(growth["per_return_2000_6000"])
    block_growth = growth["blocks_2_to_4"]
    block_range = rf"$({min(block_growth) * 1e4:.1f}$--${max(block_growth) * 1e4:.1f})\times10^{{-4}}$ (blocks 2--4)"

    # The regular-like and fine-grid thresholds are analysis choices, not measured values.
    regular_threshold, end_threshold = 1e-3, 1e-2
    rows = [
        (
            r"Radial cuts, strongly chaotic (Secs.~\ref{sec:layer},~\ref{sec:layerdelta})",
            "exponent per return, returns 2000--6000",
            rf"$> {radial_threshold}$",
            rf"${radial_growth}$",
        ),
        (
            r"Radial cuts, regular-like (Sec.~\ref{sec:layerdelta})",
            "exponent per return, returns 2000--6000",
            rf"$|\lambda| < {scientific(regular_threshold)}$",
            rf"${radial_growth}$",
        ),
        (
            r"Survey, flagged (Sec.~\ref{sec:survey})",
            r"rate per unit time, $[5000, 2\times10^4]$",
            rf"$> {survey_threshold}$",
            rf"${scientific(growth['survey_T4_T'])}$",
        ),
        (
            r"Fine grids at the ends, strongly chaotic (Sec.~\ref{sec:survey})",
            r"rate per unit time, $[500, 2000]$",
            rf"$> {scientific(end_threshold)}$",
            rf"${scientific(growth['ends_T4_T'])}$",
        ),
        (
            r"Table~\ref{tab:edge}, chaotic starts",
            "rate per unit time, four successive 5000-unit blocks, default and tighter settings",
            rf"positive in every block at both settings (observed $\ge {minimum_rate}$)",
            block_range,
        ),
    ]
    return (
        r"""\begin{table}[!tbp]
\centering\small
\caption{Orbit-classification criteria and supporting checks.
Exponents and rates come from the variational tangent, with the Euclidean norm in $(R, 3\Mmid/\delta)$;
time windows use middle-region Schwarzschild time.
Linear growth: reference value for a tangent norm growing linearly over the stated window.}
\label{tab:criteria}
\begin{tabular}{>{\raggedright}p{0.24\textwidth}>{\raggedright}p{0.29\textwidth}>{\raggedright}p{0.17\textwidth}>{\raggedright\arraybackslash}p{0.16\textwidth}}
\toprule
Use & Quantity and window & Criterion & Linear growth \\
\midrule
"""
        + "\n".join(" & ".join(row) + r" \\" for row in rows)
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("numbers", type=Path, help="figure_numbers.json from make_figures_bcr.py")
    parser.add_argument("output", type=Path, help="output LaTeX table")
    args = parser.parse_args()
    numbers = json.loads(args.numbers.read_text())
    table = table_criteria(numbers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
