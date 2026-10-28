#!/usr/bin/env bash
# Addendum checks, Figure 10, and Table 4, from the code/ directory's environment.
# Orbit integrations were run only on the reference i9-14900K (AVX2), separately from run_all.sh and the
# cross-machine comparison (see ../README.md). Table 4 uses the saved figs/figure_numbers.json.
# usage: ./addendum/run_addendum.sh      (from code/ or from code/addendum/)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p addendum/results figs
$PY addendum/make_table_criteria.py figs/figure_numbers.json figs/table_criteria.tex > addendum/results/table_criteria.log 2>&1
$PY addendum/nudge_lowprec.py addendum/results/nudge_lowprec.json --tmax 2000 --start D > addendum/results/nudge_lowprec.log 2>&1
$PY addendum/long_orbits.py results/bcr_section/bcr_section.json addendum/results/long_orbits.json > addendum/results/long_orbits.log 2>&1
$PY addendum/bcr_fig5.py figs/fig10.pdf > addendum/results/bcr_fig5.log 2>&1
tail -n 3 addendum/results/long_orbits.log
