#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export CRIT_AID_ROOT="$ROOT"
export CRIT_AID_DATA_ROOT="${CRIT_AID_DATA_ROOT:-$ROOT/data_raw}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/scripts/prepare_data_from_raw.py"
"$ROOT/run_all_from_prepared.sh"
