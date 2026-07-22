#!/usr/bin/env bash
# Configure the variables below, then run: ./scripts/run_check_tumor_data.sh
# (SEG_DIR/OUTPUT_PREFIX are resolved relative to the directory you run this
# from, not this script's location -- run it from the repo root, or use
# absolute paths.)
#
# Run this BEFORE run_tumor_tracking.sh to catch mismatched shapes, scans
# that aren't co-registered, non-binary masks, empty masks, and corrupt
# files.

# ---- Configuration ----------------------------------------------------
SEG_DIR="tumor_volume"                         # folder with tumor_YYYY-MM.nii.gz files
FILENAME_PATTERN='(?P<date>\d{4}[-_]\d{2})\.nii\.gz$'   # matches YYYY-MM or YYYY_MM before .nii.gz

CONNECTIVITY=26                                # 6, 18, or 26 -- should match run_tumor_tracking.sh
MIN_VOLUME_MM3=8                               # noise threshold -- should match run_tumor_tracking.sh
AFFINE_ATOL=0.01                               # max per-element affine difference to call two timepoints co-registered
MAX_GAP_DAYS=400                               # flag consecutive visits further apart than this many days

OUTPUT_PREFIX=""                               # empty -> defaults to "$SEG_DIR/data_quality_report"

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

# ---- Run ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=(
  --seg-dir "$SEG_DIR"
  --filename-pattern "$FILENAME_PATTERN"
  --connectivity "$CONNECTIVITY"
  --min-volume-mm3 "$MIN_VOLUME_MM3"
  --affine-atol "$AFFINE_ATOL"
  --max-gap-days "$MAX_GAP_DAYS"
  --log-level "$LOG_LEVEL"
)

if [ -n "$OUTPUT_PREFIX" ]; then
  ARGS+=(--output-prefix "$OUTPUT_PREFIX")
fi

python3 "$REPO_ROOT/check_tumor_data.py" "${ARGS[@]}"
