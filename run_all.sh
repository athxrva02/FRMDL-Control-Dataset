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

# Run a stage: capture full output to a log (so its exit code is checked directly,
# not masked by a pipe into grep), then print a filtered summary. ${arr[@]+...} is
# the bash-3.2-safe way to expand a possibly-empty array under `set -u`.
run_stage() {  # $1=label  $2=logfile  $3=summary-regex ; remaining args = command
  local label="$1" log="$2" re="$3"; shift 3
  echo "[run_all] $label -> $log"
  if ! "$@" > "$log" 2>&1; then
    echo "[run_all] FAILED: $label  (last 25 lines of $log)"
    tail -25 "$log"
    exit 1
  fi
  grep -aE "$re" "$log" || true
}

run_stage "1/3 generate" logs/generate.log \
  "^\[|SUMMARY|PNG files|JPEG files|skipped|corruptions OK|corruptions FAILED|disk usage" \
  $PY src/generate.py ${NIMG_ARG[@]+"${NIMG_ARG[@]}"}

run_stage "2/3 evaluate" logs/evaluate.log \
  "^\[model\]|^\[done\]|^\[labels\]" \
  $PY src/evaluate.py

run_stage "3/3 analyze" logs/analyze.log \
  "^\[" \
  $PY src/analyze.py

echo "[run_all] complete. Results in results/ (CSVs + figures/). Logs in logs/."
