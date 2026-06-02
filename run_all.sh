#!/usr/bin/env bash
#
# run_all.sh — end-to-end driver for the JPEG-confound control dataset.
#
# Runs generate -> evaluate -> analyze with the venv, logging each stage to
# logs/. Safe to re-run: generate.py is idempotent (existing files are skipped).
#
# Usage:
#   ./run_all.sh                 # full default scope (config.yaml: N=500, 15 corruptions)
#   ./run_all.sh 100             # override n_images for a quicker real run
#   ./run_all.sh 500 --fresh     # wipe previously generated arms first (clean slate)
#
# Notes:
#   * The ImageNette download (~342 MB) and torchvision model weights are cached
#     and NOT removed by --fresh.
#   * Full scope is ~2-4h on Apple-silicon MPS and ~3-4 GB of disk.
#   * The progress bars (download, tqdm) are verbose; per-stage tail summaries are
#     printed below and full logs land in logs/.

set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
N="${1:-}"                       # optional n_images override
FRESH=""
for a in "$@"; do [ "$a" = "--fresh" ] && FRESH=1; done

mkdir -p logs
NIMG_ARG=()
if [ -n "$N" ] && [ "$N" != "--fresh" ]; then NIMG_ARG=(--n-images "$N"); fi

if [ -n "$FRESH" ]; then
  echo "[run_all] --fresh: removing previously generated arms (keeping downloads/caches)"
  rm -rf data/base data/clean_saves data/corrupted results manifest.csv
fi

echo "[run_all] 1/3 generate  -> logs/generate.log"
$PY src/generate.py "${NIMG_ARG[@]}" 2>&1 | tee logs/generate.log | grep -aE \
  "^\[|SUMMARY|PNG files|JPEG files|skipped|corruptions OK|corruptions FAILED|disk usage" || true

echo "[run_all] 2/3 evaluate  -> logs/evaluate.log"
$PY src/evaluate.py 2>&1 | tee logs/evaluate.log | grep -aE "^\[model\]|^\[done\]|^\[labels\]" || true

echo "[run_all] 3/3 analyze   -> logs/analyze.log"
$PY src/analyze.py 2>&1 | tee logs/analyze.log | grep -aE "^\[" || true

echo "[run_all] complete. Results in results/ (CSVs + figures/). Logs in logs/."
