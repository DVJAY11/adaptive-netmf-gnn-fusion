#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.matplotlib-cache}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python

"$PYTHON_BIN" run_experiments.py \
  --dataset Cora \
  --budgets 20 \
  --seeds 42 43 44 45 46 \
  --edge-drop-rates 0.0 0.2 0.4 \
  --epochs 200 \
  --patience 30 \
  --output-dir results_missing_edges \
  --cache-dir cache_missing_edges \
  --device auto

"$PYTHON_BIN" plot_results.py --results-dir results_missing_edges
