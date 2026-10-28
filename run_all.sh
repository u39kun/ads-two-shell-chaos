#!/usr/bin/env bash
# Run the whole pipeline from scratch, in dependency order, writing every result file under results/ and a log next
# to it, then draw the figures and write the quoted numbers to figs_rerun/.
#
# usage: ./run_all.sh [WORKERS]          (default: all cores)
#   STEPS="name ..."     run only these steps (names as printed below); their inputs must already exist
#   FORCE=1              rerun steps whose result file already exists (default: skip them, so a run can resume)
#   SKIP_SUPERSEDED=1    skip the superseded two-orbit layer scans and finite-difference product (App. C)
#   DRY=1                print the commands without running them
#
# To check a fresh copy against the shipped results:
#   mv results results_reference && ./run_all.sh 32 2>&1 | tee run_all.log
#   python compare_results.py results_reference results
set -uo pipefail
cd "$(dirname "$0")"
W=${1:-$(nproc)}
PY=${PYTHON:-python}
S=results/bcr_section; P=results/paper
mkdir -p "$S" "$P"
T0=$(date +%s); FAILED=()

# step NAME OUTPUT LOG COMMAND...   (OUTPUT "-" = always run)
step() {
  local name=$1 out=$2 log=$3; shift 3
  if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $name "* ]]; then return; fi
  if [ "$out" != "-" ] && [ -s "$out" ] && [ -z "${FORCE:-}" ]; then echo "[skip] $name ($out exists)"; return; fi
  if [ -n "${DRY:-}" ]; then echo "[dry]  $name: $* > $log"; return; fi
  echo "[$(date +%H:%M:%S)] $name: $*"
  local t=$(date +%s)
  if "$@" > "$log" 2>&1; then
    echo "[$(date +%H:%M:%S)] $name done in $(( $(date +%s) - t ))s"
  else
    echo "[$(date +%H:%M:%S)] $name FAILED (see $log)"; FAILED+=("$name")
  fi
}

# Fig. 1: crossing types and ordinary and overtaking returns (short reference trace)
step crossing_trace $P/bcr_crossings.json       $P/bcr_crossings.log       $PY bcr_crossings.py $P/bcr_crossings.json
# Fig. 2 and everything at the example's parameters (needs the section first)
step section       $S/bcr_section.json         $S/bcr_section.log         $PY bcr_section.py $S/bcr_section.json --workers $W
step checks        $S/bcr_checks.json          $S/bcr_checks.log          $PY bcr_section_checks.py $S/bcr_section.json $S/bcr_checks.json
step smoothness    $S/curve_smoothness.json    $S/curve_smoothness.log    $PY bcr_curve_smoothness.py $S/bcr_checks.json $S/curve_smoothness.json
step benettin_long $S/bcr_benettin_long.json   $S/bcr_benettin_long.log   $PY bcr_benettin_long.py
step jacobian      $S/jacobian_check.json      $S/jacobian_check.log      $PY bcr_jacobian_check.py $S/jacobian_check.json --workers $W
# Fig. 3 and Table 2
step tangent_example $S/layer_tangent_example.json $S/layer_tangent_example.log $PY bcr_layer_tangent.py example $S/layer_tangent_example.json --workers $W
step overtaking    $S/bcr_overtaking.json      $S/bcr_overtaking.log      $PY bcr_overtaking.py $S/bcr_overtaking.json --workers $W
step refinement    $S/tangent_refinement.json $S/tangent_refinement.log $PY bcr_tangent_refinement.py $S/bcr_section.json $S/tangent_refinement.json --workers $W
step tangent_long  $S/tangent_long.json        $S/tangent_long.log        $PY bcr_tangent_long.py $S/bcr_section.json $S/tangent_long.json --workers $W
# Fig. 4
step lyapunov      $P/bcr_lyapunov.json        $P/bcr_lyapunov.log        $PY bcr_lyapunov.py $P/bcr_lyapunov.json --tmax 2000 --workers $W
step lyapunov_C    $P/bcr_lyapunov_C.json      $P/bcr_lyapunov_C.log      $PY bcr_lyapunov.py $P/bcr_lyapunov_C.json --tmax 2000 --start C --workers $W
# Fig. 5 (the delta scan also feeds Fig. 6)
step delta_scan    $S/delta_scan.json          $S/delta_scan.log          $PY bcr_delta_scan.py $S/delta_scan.json --workers $W
step delta_scan_C  $S/delta_scan_C.json        $S/delta_scan_C.log        env BCR_START=C $PY bcr_delta_scan.py $S/delta_scan_C.json --workers $W
step fig7          $S/bcr_fig7.json            $S/bcr_fig7.log            $PY bcr_fig7.py $S/bcr_fig7.json --workers $W
step fig7_short    $S/bcr_fig7_t1000.json      $S/bcr_fig7_t1000.log      $PY bcr_fig7.py $S/bcr_fig7_t1000.json --tmax 1000 --workers $W
step tangent_family $S/layer_tangent_family.json $S/layer_tangent_family.log $PY bcr_layer_tangent.py family $S/layer_tangent_family.json --returns 4000 --workers $W
step family_long   $S/family_long.json         $S/family_long.log         $PY bcr_family_long.py $S/delta_scan.json $S/family_long.json --workers $W
# Fig. 6 (saddles from layer_delta; the extension picks the orbits with intermediate exponents from tangent_delta)
step layer_delta   $S/layer_delta.json         $S/layer_delta.log         $PY bcr_layer_delta.py $S/delta_scan.json $S/layer_delta.json --workers $W
step tangent_delta $S/layer_tangent_delta.json $S/layer_tangent_delta.log $PY bcr_layer_tangent.py delta $S/layer_tangent_delta.json --workers $W
step tangent_extend $S/layer_tangent_extend.json $S/layer_tangent_extend.log $PY bcr_layer_tangent.py extend $S/layer_tangent_extend.json --returns 20000 --workers $W
# Figs. 7-8 and Table 1 (chaos away from the saddle; Fig. 8 also reads layer_delta and tangent_delta)
step survey        $S/survey.json              $S/survey.log              $PY bcr_survey.py section $S/survey.json --workers $W
step survey_ends   $S/survey_ends.json         $S/survey_ends.log         $PY bcr_survey.py ends $S/survey_ends.json --workers $W
step survey_checks $S/survey_checks.json       $S/survey_checks.log       $PY bcr_survey_checks.py $S/survey_checks.json --workers $W
step chart_check   $S/chart_check.json         $S/chart_check.log         $PY bcr_chart_check.py $S/bcr_section.json $S/chart_check.json --workers $W
step ends_followup $S/ends_followup.json      $S/ends_followup.log       $PY bcr_ends_followup.py $S/survey_ends.json $S/layer_delta.json $S/ends_followup.json --workers $W
step overview      $S/section_overview.json    $S/section_overview.log    $PY bcr_section_overview.py $S/section_overview.json
step collapse_delay $S/collapse_delay.json     $S/collapse_delay.log      $PY bcr_collapse_delay.py $S/collapse_delay.json --workers $W
# Secs. 2 and 4.3: the section's components, the collapse criterion, and rates in boundary time
step section_domain $S/section_domain.json     $S/section_domain.log      $PY bcr_section_domain.py $S/section_domain.json --workers $W
step collapse_criterion $S/collapse_criterion.json $S/collapse_criterion.log $PY bcr_collapse_criterion.py $S/survey.json $S/collapse_criterion.json --workers $W
step boundary_rates $S/boundary_rates.json     $S/boundary_rates.log      $PY bcr_boundary_rates.py $S/layer_tangent_delta.json $S/boundary_rates.json --workers $W
# Fig. 9
step validation    $P/bcr_validation.json      $P/bcr_validation.log      $PY bcr_validation.py $P/bcr_validation.json
step critical      $P/bcr_critical.json        $P/bcr_critical.log        $PY bcr_critical.py $P/bcr_critical.json
# Table 3 and App. C
step convergence   $S/layer_convergence.json   $S/layer_convergence.log   $PY bcr_layer_convergence.py $S/bcr_section.json $S/layer_convergence.json --workers $W
step noise_trace   $S/noise_trace.json         $S/noise_trace.log         $PY bcr_noise_trace.py $S/noise_trace.json
# Sec. 2, Sec. 3.3, and Apps. A and B
step independent   $P/independent_check.json   $P/independent_check.log   $PY independent_check.py $P/independent_check.json
step reversal      $S/bcr_reversal.json        $S/bcr_reversal.log        $PY bcr_reversal.py $S/bcr_reversal.json
step wba           $S/bcr_wba.json             $S/bcr_wba.log             $PY bcr_wba.py $S/bcr_section.json $S/bcr_wba.json --workers $W
step measure       $P/two_shell_measure.json   $P/two_shell_measure.log   $PY two_shell_measure.py $P/two_shell_measure.json
step sympy_proof   -                           $P/sympy_crossing_proof.log $PY sympy_crossing_proof.py
# superseded by the variational tangent, kept for App. C and Table 3's comparison
if [ -z "${SKIP_SUPERSEDED:-}" ]; then
step layer_scan    $S/layer_scan.json          $S/layer_scan.log          $PY bcr_layer_scan.py $S/bcr_section.json $S/layer_scan.json --workers $W
step layer_refine  $S/layer_refine.json        $S/layer_refine.log        $PY bcr_layer_scan.py $S/bcr_section.json $S/layer_refine.json --returns 6000 --dists 1e-4,1e-2,2e-2,3e-2,5e-2,7e-2 --workers $W
step fd_product    $S/bcr_ftle.json            $S/bcr_ftle.log            $PY bcr_section_lyapunov.py $S/bcr_section.json $S/bcr_ftle.json
fi
# figures, Tables 1-3, and the numbers quoted with them, from the results just computed
step figures       -                           figs_rerun.log             $PY make_figures_bcr.py figs_rerun

echo "total $(( ($(date +%s) - T0) / 60 )) min"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED steps: ${FAILED[*]}"; exit 1; fi
echo "all steps finished"
