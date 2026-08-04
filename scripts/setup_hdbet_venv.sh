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
source "$SCRIPT_DIR/hdbet_venv_path.sh"

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
import re
import shutil
import subprocess
import sys

import torch

BAR = '=' * 70
nvidia_smi_present = shutil.which('nvidia-smi') is not None


def best_cuda_wheel_tag():
    # torch.pytorch.org only publishes wheels for specific CUDA versions, not
    # every point release -- pick the highest one that's still <= the
    # driver's max-supported CUDA version (a driver can run code built for
    # its own CUDA version or older, never newer). Extend this list as new
    # tags get published (check https://download.pytorch.org/whl/).
    known_tags = [(11, 8), (12, 1), (12, 4), (12, 6), (12, 8)]
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', out)
        if not m:
            return None
        driver_max = (int(m.group(1)), int(m.group(2)))
    except Exception:
        return None
    candidates = [t for t in known_tags if t <= driver_max]
    if not candidates:
        return None
    major, minor = max(candidates)
    return f'cu{major}{minor}'

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
    print('This usually means the default pip install resolved a torch build compiled for a newer CUDA', file=sys.stderr)
    print('runtime than this driver supports (a driver runs its own CUDA version or OLDER, never newer --', file=sys.stderr)
    print('note this is independent of HD-BET/nnU-Net\'s own torch>=2.0.0 Python-API requirement, which any', file=sys.stderr)
    print('CUDA-tagged build still satisfies).', file=sys.stderr)
    print('Left as-is, HD-BET will SILENTLY fall back to CPU (~1-2h/scan) on a machine meant to have a GPU.', file=sys.stderr)
    tag = best_cuda_wheel_tag()
    if tag:
        print(f'Detected driver supports up to CUDA matching wheel tag \'{tag}\'. Reinstall torch/torchvision for it:', file=sys.stderr)
        print(f'  {sys.executable} -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/{tag}', file=sys.stderr)
    else:
        print('Could not auto-detect your driver\'s max CUDA version from nvidia-smi -- check it manually', file=sys.stderr)
        print('(\"CUDA Version: X.Y\" in the nvidia-smi header) and reinstall a matching build, e.g.:', file=sys.stderr)
        print(f'  {sys.executable} -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121', file=sys.stderr)
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
