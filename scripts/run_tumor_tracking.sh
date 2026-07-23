#!/usr/bin/env bash
# Configure below, then run: ./scripts/run_tumor_tracking.sh (paths are relative to your cwd)

SEG_DIR="tumor_volume"
OUTPUT_CSV="./tumor_volumes.csv"
PATIENT_ID=""
FILENAME_PATTERN='(?P<date>\d{4}[-_]\d{2})\.nii\.gz$'

CONNECTIVITY=26                                # 6, 18, or 26
MIN_VOLUME_MM3=20                              # noise filter
MATCH_CENTROID_THRESHOLD_MM=10
MATCH_SURFACE_THRESHOLD_MM=5
DROP_SINGLE_TIMEPOINT_TRACKS=false              # drop noise-like single-timepoint tracks

SAVE_CC_MAPS=true
CC_MAPS_DIR=""                                 # empty -> defaults next to OUTPUT_CSV

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

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

if [ "$DROP_SINGLE_TIMEPOINT_TRACKS" = true ]; then
  ARGS+=(--drop-single-timepoint-tracks)
fi

if [ "$SAVE_CC_MAPS" = true ]; then
  ARGS+=(--save-cc-maps)
  if [ -n "$CC_MAPS_DIR" ]; then
    ARGS+=(--cc-maps-dir "$CC_MAPS_DIR")
  fi
fi

python3 "$REPO_ROOT/utils/tumor_tracking.py" "${ARGS[@]}"
