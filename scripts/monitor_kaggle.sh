#!/bin/bash
# Monitor the DiffuChat Kaggle kernel and fetch artifacts when COMPLETE.
# Usage: bash scripts/monitor_kaggle.sh [poll_interval_s]
set -u
export KAGGLE_CONFIG_DIR=/home/z/.kaggle
REF="kishore00afk/diffuchat-kaggle-training"
INTERVAL="${1:-120}"
ART_DIR="/home/z/my-project/download/diffuchat_artifacts"

while true; do
  STATUS=$(kaggle kernels status "$REF" 2>&1 | tail -1)
  echo "[$(date +%H:%M:%S)] $STATUS"
  if echo "$STATUS" | grep -q "COMPLETE"; then
    echo "run complete — fetching artifacts"
    mkdir -p "$ART_DIR"
    kaggle kernels output "$REF" -p "$ART_DIR" 2>&1 | tail -2
    ls -la "$ART_DIR"
    break
  fi
  if echo "$STATUS" | grep -qE "ERROR|CANCEL"; then
    echo "run failed/canceled — inspect logs"
    break
  fi
  sleep "$INTERVAL"
done
