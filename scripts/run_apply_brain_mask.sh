#!/usr/bin/env bash

LABELS_DIR="registered_labels"
MASKS_DIR="brats_preprocessed"                 # HD-BET's --save_bet_mask output dir
OUTPUT_DIR="brats_preprocessed_labels"
PATTERN="*.nii.gz"
RECURSIVE=true

NAMING_SCHEME="braintracking"                  # identical, or braintracking (label "tumor_2016-11.nii.gz" <-> scan "flair_2016_11.nii.gz")

MANIFEST_CSV=""                                # empty -> defaults next to OUTPUT_DIR

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  --labels-dir "$LABELS_DIR"
  --masks-dir "$MASKS_DIR"
  --output-dir "$OUTPUT_DIR"
  --pattern "$PATTERN"
  --naming-scheme "$NAMING_SCHEME"
  --log-level "$LOG_LEVEL"
)

if [ "$RECURSIVE" = true ]; then
  ARGS+=(--recursive)
else
  ARGS+=(--no-recursive)
fi

if [ -n "$MANIFEST_CSV" ]; then
  ARGS+=(--manifest-csv "$MANIFEST_CSV")
fi

"$REPO_ROOT/venv/bin/python3" "$REPO_ROOT/utils/apply_brain_mask.py" "${ARGS[@]}"
