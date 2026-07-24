#!/usr/bin/env bash
# One-time setup: create an isolated venv for HD-BET (skull-stripping).
# HD-BET pins nnunetv2>=2.5.1, which conflicts with this repo's own nnunetv2
# fork (setup.py pins 2.1.1) -- an isolated venv avoids that clash entirely
# instead of risking your training/inference environment. Run this once,
# then run_preprocess_brats_like.sh uses it automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HDBET_VENV_DIR="$REPO_ROOT/hdbet_venv"

python3 -m venv "$HDBET_VENV_DIR"
"$HDBET_VENV_DIR/bin/pip" install --upgrade pip
# On Linux, this pulls a CUDA-enabled torch build automatically (PyPI's
# default manylinux wheel bundles CUDA -- no special --index-url needed) as
# long as the machine has an NVIDIA driver; on macOS there's no CUDA wheel,
# so this becomes a CPU (or mps) build instead. Either way, the same
# requirement works unmodified on both -- see the GPU check below for which
# one you actually got.
"$HDBET_VENV_DIR/bin/pip" install HD-BET

echo ""
echo "HD-BET installed in $HDBET_VENV_DIR"
echo "First real run downloads pretrained weights automatically (needs network access)."
echo ""
echo "GPU check:"
"$HDBET_VENV_DIR/bin/python3" -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = ', '.join(torch.cuda.get_device_name(i) for i in range(n))
    print(f'  CUDA available -- {n} GPU(s): {names}')
elif torch.backends.mps.is_available():
    print('  No CUDA GPU, but Apple MPS is available (not auto-selected by run_preprocess_brats_like.sh -- set HDBET_DEVICE=mps explicitly to try it).')
else:
    print('  No GPU detected -- HD-BET will run on CPU (slow, ~1-2h/scan for typical volumes).')
"
