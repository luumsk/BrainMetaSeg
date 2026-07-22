#!/usr/bin/env bash
# Configure the variables below, then run: ./scripts/run_tumor_tracking.sh
# (SEG_DIR/OUTPUT_CSV are resolved relative to the directory you run this
# from, not this script's location -- run it from the repo root, or use
# absolute paths.)

# ---- Configuration ----------------------------------------------------
SEG_DIR="tumor_volume"
OUTPUT_CSV="./tumor_volumes.csv"
PATIENT_ID=""                                  
FILENAME_PATTERN='(?P<date>\d{4}[-_]\d{2})\.nii\.gz$'   # matches YYYY-MM or YYYY_MM before .nii.gz

CONNECTIVITY=26                                # 6, 18, or 26
MIN_VOLUME_MM3=8                               # drop components smaller than this (noise filter)
MATCH_CENTROID_THRESHOLD_MM=10
MATCH_SURFACE_THRESHOLD_MM=5

SAVE_CC_MAPS=true                              # true/false -- persist labeled component maps per timepoint
CC_MAPS_DIR=""                                 # empty -> defaults next to OUTPUT_CSV

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

# ---- Run ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=(
  --seg-dir "$SEG_DIR"
  --output-csv "$OUTPUT_CSV"
  --filename-pattern "$FILENAME_PATTERN"
  --connectivity "$CONNECTIVITY"
  --min-volume-mm3 "$MIN_VOLUME_MM3"
  --match-centroid-threshold-mm "$MATCH_CENTROID_THRESHOLD_MM"
  --match-surface-threshold-mm "$MATCH_SURFACE_THRESHOLD_MM"
  --log-level "$LOG_LEVEL"
)

if [ -n "$PATIENT_ID" ]; then
  ARGS+=(--patient-id "$PATIENT_ID")
fi

if [ "$SAVE_CC_MAPS" = true ]; then
  ARGS+=(--save-cc-maps)
  if [ -n "$CC_MAPS_DIR" ]; then
    ARGS+=(--cc-maps-dir "$CC_MAPS_DIR")
  fi
fi

python3 "$REPO_ROOT/tumor_tracking.py" "${ARGS[@]}"
