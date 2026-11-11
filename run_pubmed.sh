#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.matplotlib-cache}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python

# PubMed uses the eigen (large-window) NetMF approximation: the exact
# walk-power version needs far more memory on a graph of this size.
"$PYTHON_BIN" run_experiments.py \
  --dataset PubMed \
  --budgets 5 10 20 \
  --seeds 42 43 44 45 46 \
  --epochs 200 \
  --patience 30 \
  --netmf-method eigen \
  --netmf-rank 256 \
  --output-dir results_pubmed \
  --cache-dir cache_pubmed \
  --device auto

"$PYTHON_BIN" plot_results.py --results-dir results_pubmed
