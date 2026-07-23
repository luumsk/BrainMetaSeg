#!/usr/bin/env bash
# Configure the variables below, then run: ./scripts/run_compute_seg_metrics.sh
# (GT_DIR/SEG_DIR/OUTPUT_CSV are resolved relative to the directory you run
# this from, not this script's location -- run it from the repo root, or use
# absolute paths.)

# ---- Configuration ----------------------------------------------------
GT_DIR="path/to/ground_truth"
SEG_DIR="path/to/segresnet_preds"
OUTPUT_CSV="./seg_metrics.csv"
PATTERN="*.nii.gz"                             # glob (relative to GT_DIR) for gt files to evaluate
NAMING_SCHEME="identical"                      # identical, or braintracking (gt "tumor_2016-11.nii.gz" <-> seg "flair_2016_11.nii.gz")

CONNECTIVITY=26                                # 6, 18, or 26 -- used for instance counting
MIN_VOLUME_MM3=20                              # drop components smaller than this when counting instances (noise filter)

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

# ---- Run ---------------------------------------------------------------
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

python3 "$REPO_ROOT/compute_seg_metrics.py" "${ARGS[@]}"
