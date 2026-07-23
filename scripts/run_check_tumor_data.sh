#!/usr/bin/env bash
# Configure below, then run: ./scripts/run_check_tumor_data.sh (paths are relative to your cwd)
# Run this BEFORE run_tumor_tracking.sh.

SEG_DIR="tumor_volume"
FILENAME_PATTERN='(?P<date>\d{4}[-_]\d{2})\.nii\.gz$'

CONNECTIVITY=26                                # should match run_tumor_tracking.sh
MIN_VOLUME_MM3=20                              # should match run_tumor_tracking.sh
AFFINE_ATOL=0.01
MAX_GAP_DAYS=400                               # flag visits further apart than this

OUTPUT_PREFIX=""                               # empty -> defaults to "$SEG_DIR/data_quality_report"

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

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

python3 "$REPO_ROOT/utils/check_tumor_data.py" "${ARGS[@]}"
