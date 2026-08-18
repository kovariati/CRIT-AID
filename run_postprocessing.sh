#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CRIT_AID_ROOT="$ROOT"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/build_result_registry.py"
python "$ROOT/scripts/build_tables_figures.py"
python "$ROOT/scripts/build_acs_tables_figures.py"
python "$ROOT/scripts/make_protocol_figure.py"
python "$ROOT/scripts/run_acs_target_definition_decomposition.py"
python "$ROOT/scripts/run_conformal_tradeoff_analysis.py"
python "$ROOT/scripts/run_cross_domain_variance_analysis.py"
python "$ROOT/scripts/build_diagnostic_result_registry.py"
python "$ROOT/scripts/audit_repository.py" --mode results --skip-checksums
