#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python run_experiments.py all \
  --capacity-checkpoints 100 100000 \
  --capacity-preset fast \
  --capacity-sample-size 64 \
  --ablation-seeds 0 1 2 3 4 5 6 7 8 9 \
  --ablation-corpus-size 64 \
  --ablation-preset tuned \
  "$@"
