#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CRIT_AID_ROOT="$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/run_acs_analysis.py"
python "$ROOT/scripts/run_acs_postprocessing.py"
python "$ROOT/scripts/run_oulad_analysis.py"
python "$ROOT/scripts/run_oulad_bootstrap.py"
python "$ROOT/scripts/run_external_domain_analyses.py"
python "$ROOT/scripts/run_model_family_sensitivity.py"
"$ROOT/run_postprocessing.sh"
