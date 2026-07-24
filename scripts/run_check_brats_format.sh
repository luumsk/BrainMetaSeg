#!/usr/bin/env bash

INPUT_DIR="./nifti_native"
PATTERN="*.nii.gz"
RECURSIVE=true                                 # search INPUT_DIR's subfolders too

SKULL_STRIPPED_THRESHOLD=0.40                  # nonzero-voxel-frac cutoff (see utils/check_brats_format.py docstring)
TEMPLATE_CHANNEL="spgr_unstrip"                # must match whatever channel you register to -- spgr/spgr_unstrip disagree on L/R sign (see check_brats_format.py docstring)

OUTPUT_PREFIX=""                               # empty -> defaults to "$INPUT_DIR/brats_format_report"

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  --input-dir "$INPUT_DIR"
  --pattern "$PATTERN"
  --skull-stripped-nonzero-threshold "$SKULL_STRIPPED_THRESHOLD"
  --template-channel "$TEMPLATE_CHANNEL"
  --log-level "$LOG_LEVEL"
)

if [ "$RECURSIVE" = true ]; then
  ARGS+=(--recursive)
else
  ARGS+=(--no-recursive)
fi

if [ -n "$OUTPUT_PREFIX" ]; then
  ARGS+=(--output-prefix "$OUTPUT_PREFIX")
fi

python "$REPO_ROOT/utils/check_brats_format.py" "${ARGS[@]}"
