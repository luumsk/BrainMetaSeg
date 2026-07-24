#!/usr/bin/env bash
# One-time setup: create an isolated venv for HD-BET (skull-stripping).
# HD-BET pins nnunetv2>=2.5.1, which conflicts with this repo's own nnunetv2
# fork (setup.py pins 2.1.1) -- an isolated venv avoids that clash entirely
# instead of risking your training/inference environment. Run this once,
# then run_preprocess_brats.sh uses it automatically.
#
# Exits non-zero if an NVIDIA GPU is physically present (nvidia-smi works)
# but PyTorch can't actually use it -- that's a real misconfiguration, not a
# valid "no GPU" environment, and would otherwise have HD-BET silently fall
# back to CPU (~1-2h/scan) on a machine you specifically set up for GPU.
# Exits 0 (with a clear informational banner, not a failure) on a genuine
# CPU-only machine, e.g. for local dev.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HDBET_VENV_DIR="/media/storage/luu/hdbet_venv"

python3 -m venv "$HDBET_VENV_DIR"
"$HDBET_VENV_DIR/bin/pip" install --upgrade pip
# On Linux, this pulls a CUDA-enabled torch build automatically (PyPI's
# default manylinux wheel bundles CUDA -- no special --index-url needed) as
# long as the machine has an NVIDIA driver; on macOS there's no CUDA wheel,
# so this becomes a CPU (or mps) build instead. Either way, the same
# requirement works unmodified on both -- the GPU check below verifies which
# one you actually got, rather than assuming.
"$HDBET_VENV_DIR/bin/pip" install HD-BET

echo ""
echo "HD-BET installed in $HDBET_VENV_DIR"
echo "First real run downloads pretrained weights automatically (needs network access)."
echo ""

"$HDBET_VENV_DIR/bin/python3" -c "
import shutil
import sys

import torch

BAR = '=' * 70
nvidia_smi_present = shutil.which('nvidia-smi') is not None

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    try:
        # A real op, not just the availability flag -- catches driver/CUDA
        # version mismatches that torch.cuda.is_available() alone can miss.
        x = torch.randn(256, 256, device='cuda')
        (x @ x).sum().item()
    except Exception as exc:
        print(BAR, file=sys.stderr)
        print('GPU CHECK FAILED', file=sys.stderr)
        print(f'torch reports CUDA available but a test GPU operation raised: {exc}', file=sys.stderr)
        print('HD-BET may crash or silently misbehave on this device -- fix before running the real batch.', file=sys.stderr)
        print(BAR, file=sys.stderr)
        sys.exit(1)
    names = ', '.join(torch.cuda.get_device_name(i) for i in range(n))
    print(BAR)
    print(f'GPU READY -- {n} device(s): {names}')
    print('run_preprocess_brats.sh will auto-select this (HDBET_DEVICE=\"\" -> cuda).')
    print(BAR)
    sys.exit(0)

if nvidia_smi_present:
    print(BAR, file=sys.stderr)
    print('GPU CHECK FAILED', file=sys.stderr)
    print('nvidia-smi found an NVIDIA GPU, but PyTorch cannot see it (torch.cuda.is_available() is False).', file=sys.stderr)
    print('This usually means a CPU-only torch build got installed, or a driver/CUDA version mismatch.', file=sys.stderr)
    print('Left as-is, HD-BET will SILENTLY fall back to CPU (~1-2h/scan) on a machine meant to have a GPU.', file=sys.stderr)
    print('Try reinstalling torch in hdbet_venv with a CUDA build matching this machine\'s driver, e.g.:', file=sys.stderr)
    print(f'  {sys.executable} -m pip install torch --index-url https://download.pytorch.org/whl/cu121', file=sys.stderr)
    print(BAR, file=sys.stderr)
    sys.exit(1)

if torch.backends.mps.is_available():
    print(BAR)
    print('No NVIDIA GPU found (this looks like Apple Silicon). MPS is available but NOT')
    print('auto-selected by run_preprocess_brats.sh -- set HDBET_DEVICE=mps explicitly to try it')
    print('(not verified compatible with this model; cpu is the safe default).')
    print(BAR)
    sys.exit(0)

print(BAR)
print('No GPU detected -- HD-BET will run on CPU (slow: ~1-2h/scan for typical volumes).')
print('Expected on a CPU-only machine (e.g. a laptop). If this is meant to be a GPU server,')
print('something is wrong upstream of PyTorch -- no NVIDIA GPU was found at all (nvidia-smi missing).')
print(BAR)
sys.exit(0)
"
