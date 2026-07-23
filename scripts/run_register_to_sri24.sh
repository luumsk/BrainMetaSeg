#!/usr/bin/env bash
# Configure below, then run: ./scripts/run_register_to_sri24.sh (paths are relative to your cwd)

INPUT_DIR="nifti_native"
OUTPUT_DIR="registered"
PATTERN="*.nii.gz"
RECURSIVE=true                                 # search INPUT_DIR's subfolders too
OUTPUT_SUFFIX=""                               # e.g. "_sri24", inserted before .nii.gz

TEMPLATE_CHANNEL="spgr_unstrip"                # nifti_native scans still have skull on -- use spgr instead if you skull-strip first
TEMPLATE_PATH=""                               # empty -> auto-download TEMPLATE_CHANNEL
TEMPLATE_CACHE_DIR=""                          # empty -> defaults to <repo>/templates/sri24

TRANSFORM_TYPE="Affine"                        # Rigid, Affine, SyN, SyNRA -- Affine avoids deforming lesion volume
N4_CORRECT=true
INTERPOLATOR="linear"                          # linear, bSpline, nearestNeighbor

SAVE_TRANSFORMS=true
TRANSFORMS_DIR=""                              # empty -> defaults next to OUTPUT_DIR
MANIFEST_CSV=""                                # empty -> defaults next to OUTPUT_DIR

OVERWRITE=false

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"  # so INPUT_DIR/OUTPUT_DIR above resolve consistently, no matter where you ran this from

ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --pattern "$PATTERN"
  --output-suffix "$OUTPUT_SUFFIX"
  --template-channel "$TEMPLATE_CHANNEL"
  --transform-type "$TRANSFORM_TYPE"
  --interpolator "$INTERPOLATOR"
  --log-level "$LOG_LEVEL"
)

if [ "$RECURSIVE" = true ]; then
  ARGS+=(--recursive)
else
  ARGS+=(--no-recursive)
fi

if [ "$N4_CORRECT" = true ]; then
  ARGS+=(--n4-correct)
else
  ARGS+=(--no-n4-correct)
fi

if [ "$SAVE_TRANSFORMS" = true ]; then
  ARGS+=(--save-transforms)
else
  ARGS+=(--no-save-transforms)
fi

if [ -n "$TEMPLATE_PATH" ]; then
  ARGS+=(--template-path "$TEMPLATE_PATH")
fi

if [ -n "$TEMPLATE_CACHE_DIR" ]; then
  ARGS+=(--template-cache-dir "$TEMPLATE_CACHE_DIR")
fi

if [ -n "$TRANSFORMS_DIR" ]; then
  ARGS+=(--transforms-dir "$TRANSFORMS_DIR")
fi

if [ -n "$MANIFEST_CSV" ]; then
  ARGS+=(--manifest-csv "$MANIFEST_CSV")
fi

if [ "$OVERWRITE" = true ]; then
  ARGS+=(--overwrite)
fi

python3 "$REPO_ROOT/utils/register_to_sri24.py" "${ARGS[@]}"
