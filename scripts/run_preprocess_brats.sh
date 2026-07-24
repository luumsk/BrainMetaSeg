#!/usr/bin/env bash
# Single-entry-point BraTS-like preprocessing: one input folder, one output
# folder. Run scripts/setup_hdbet_venv.sh once first (needed for SKULL_STRIP).
# See utils/preprocess_brats.py's docstring for the full step order/rationale.

INPUT_DIR="nifti_native"
OUTPUT_DIR="brats_preprocessed"
PATTERN="*.nii.gz"
RECURSIVE=true

LABELS_DIR="tumor_volume"                      # empty -> scans only, no label preprocessing
LABELS_NAMING_SCHEME="braintracking"           # identical, or braintracking (scan "flair_2016_11.nii.gz" <-> label "tumor_2016-11.nii.gz")

# ---- Steps (each independently toggleable) -----------------------------
N4_CORRECT=true                                # step 1
REGISTER=true                                  # step 2 -- also yields isotropic spacing as a side effect
TEMPLATE_CHANNEL="spgr_unstrip"                # skull-on SRI24 channel -- use spgr if your inputs are already skull-stripped
TRANSFORM_TYPE="Rigid"                         # BraTS uses rigid, not affine -- preserves true volume
INTERPOLATOR="linear"
RESAMPLE=true                                  # step 3 -- only takes effect if REGISTER=false
SKULL_STRIP=true                               # step 4 (HD-BET)
NORMALIZE=true                                 # step 5 -- z-score within the brain mask

HDBET_DEVICE=""                                # empty -> auto-detect (cuda if available, else cpu); or "cpu"/"cuda"/"mps"
HDBET_DISABLE_TTA=false                        # true trades a little quality for speed -- consider it if still on cpu
HDBET_VENV_DIR=""                              # empty -> defaults to "$REPO_ROOT/hdbet_venv"

SAVE_TRANSFORMS=true
VERIFY=true                                    # independent post-hoc check of the saved outputs
KEEP_INTERMEDIATE=false
OVERWRITE=false

LOG_LEVEL="INFO"                               # DEBUG, INFO, WARNING, ERROR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --pattern "$PATTERN"
  --template-channel "$TEMPLATE_CHANNEL"
  --transform-type "$TRANSFORM_TYPE"
  --interpolator "$INTERPOLATOR"
  --hdbet-device "$HDBET_DEVICE"
  --log-level "$LOG_LEVEL"
)

if [ "$RECURSIVE" = true ]; then ARGS+=(--recursive); else ARGS+=(--no-recursive); fi
if [ -n "$LABELS_DIR" ]; then
  ARGS+=(--labels-dir "$LABELS_DIR" --labels-naming-scheme "$LABELS_NAMING_SCHEME")
fi
if [ "$N4_CORRECT" = true ]; then ARGS+=(--n4-correct); else ARGS+=(--no-n4-correct); fi
if [ "$REGISTER" = true ]; then ARGS+=(--register); else ARGS+=(--no-register); fi
if [ "$RESAMPLE" = true ]; then ARGS+=(--resample); else ARGS+=(--no-resample); fi
if [ "$SKULL_STRIP" = true ]; then ARGS+=(--skull-strip); else ARGS+=(--no-skull-strip); fi
if [ "$HDBET_DISABLE_TTA" = true ]; then ARGS+=(--hdbet-disable-tta); fi
if [ -n "$HDBET_VENV_DIR" ]; then ARGS+=(--hdbet-venv-dir "$HDBET_VENV_DIR"); fi
if [ "$NORMALIZE" = true ]; then ARGS+=(--normalize); else ARGS+=(--no-normalize); fi
if [ "$SAVE_TRANSFORMS" = true ]; then ARGS+=(--save-transforms); else ARGS+=(--no-save-transforms); fi
if [ "$VERIFY" = true ]; then ARGS+=(--verify); else ARGS+=(--no-verify); fi
if [ "$KEEP_INTERMEDIATE" = true ]; then ARGS+=(--keep-intermediate); fi
if [ "$OVERWRITE" = true ]; then ARGS+=(--overwrite); fi

"$REPO_ROOT/venv/bin/python3" "$REPO_ROOT/utils/preprocess_brats.py" "${ARGS[@]}"
