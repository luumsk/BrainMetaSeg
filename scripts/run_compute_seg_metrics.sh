#!/usr/bin/env bash
# Configure below, then run: ./scripts/run_compute_seg_metrics.sh (paths are relative to your cwd)

GT_DIR="path/to/ground_truth"
SEG_DIR="path/to/segresnet_preds"
OUTPUT_CSV="./seg_metrics.csv"
PATTERN="*.nii.gz"
NAMING_SCHEME="identical"                      # identical or braintracking

CONNECTIVITY=26                                # 6, 18, or 26
MIN_VOLUME_MM3=20                              # noise filter for instance counting

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=(
  --gt-dir "$GT_DIR"
  --seg-dir "$SEG_DIR"
  --output-csv "$OUTPUT_CSV"
  --pattern "$PATTERN"
  --naming-scheme "$NAMING_SCHEME"
  --connectivity "$CONNECTIVITY"
  --min-volume-mm3 "$MIN_VOLUME_MM3"
  --log-level "$LOG_LEVEL"
)

python3 "$REPO_ROOT/utils/compute_seg_metrics.py" "${ARGS[@]}"
