#!/usr/bin/env python3
"""Apply a skull-strip brain mask (e.g. from HD-BET) to label/segmentation files.

Skull-stripping tools like HD-BET are neural networks trained on MRI
intensity contrast -- they can't run on a label/segmentation map, so
"skull-strip the labels too" has to happen differently: reuse the brain
mask HD-BET already computed for the matching scan (HD-BET's --save_bet_mask
writes "<scan_stem>_bet.nii.gz" next to its skull-stripped output) and zero
out every label voxel outside it.

Label and mask are expected to already share one grid (true if both trace
back to the same SRI24-registered scan via register_to_sri24.py --
this script does not resample, only checks and masks).

Usage
-----
    python utils/apply_brain_mask.py \\
        --labels-dir registered_labels \\
        --masks-dir brats_preprocessed \\
        --output-dir brats_preprocessed_labels \\
        --naming-scheme braintracking
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from compute_seg_metrics import NAMING_SCHEMES


logger = logging.getLogger("apply_brain_mask")

MASK_SUFFIX = "_bet.nii.gz"  # HD-BET's own convention for --save_bet_mask output


def strip_nifti_suffix(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


@dataclass
class MaskResult:
    label: str
    label_path: Path
    mask_path: Optional[Path] = None
    output_path: Optional[Path] = None
    status: str = "ok"
    error: str = ""
    label_voxels_before: int = 0
    label_voxels_after: int = 0
    voxels_removed_outside_brain: int = 0


def discover_labels(labels_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    glob_fn = labels_dir.rglob if recursive else labels_dir.glob
    paths = sorted(p for p in glob_fn(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {labels_dir}")
    return paths


def apply_mask_to_label(label_path: Path, mask_path: Path, output_path: Path) -> MaskResult:
    result = MaskResult(label=label_path.name, label_path=label_path, mask_path=mask_path)
    try:
        label_img = nib.load(str(label_path))
        mask_img = nib.load(str(mask_path))
        label_data = np.asarray(label_img.dataobj)
        mask_data = np.asarray(mask_img.dataobj)

        if label_data.shape != mask_data.shape:
            raise ValueError(f"Shape mismatch: label {label_data.shape} vs mask {mask_data.shape} -- not on the same grid")

        brain = mask_data > 0
        masked = np.where(brain, label_data, 0).astype(label_data.dtype)

        result.label_voxels_before = int((label_data > 0).sum())
        result.label_voxels_after = int((masked > 0).sum())
        result.voxels_removed_outside_brain = result.label_voxels_before - result.label_voxels_after

        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(masked, label_img.affine, label_img.header), str(output_path))
        result.output_path = output_path

    except Exception as exc:  # one bad label shouldn't kill the whole batch
        result.status = "failed"
        result.error = str(exc)
        logger.error("%s: %s", label_path.name, exc)

    return result


def run(
    labels_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    pattern: str,
    recursive: bool,
    naming_scheme: str,
    manifest_csv: Path,
) -> pd.DataFrame:
    labels = discover_labels(labels_dir, pattern, recursive)
    logger.info("Found %d label(s) in %s", len(labels), labels_dir)

    map_label_to_scan = NAMING_SCHEMES[naming_scheme]

    results = []
    for i, label_path in enumerate(labels, start=1):
        try:
            scan_filename = map_label_to_scan(label_path.name)
        except ValueError as exc:
            logger.warning("%s: could not derive scan filename -- %s", label_path.name, exc)
            results.append(MaskResult(label=label_path.name, label_path=label_path, status="failed", error=str(exc)))
            continue

        mask_path = masks_dir / f"{strip_nifti_suffix(scan_filename)}{MASK_SUFFIX}"
        if not mask_path.is_file():
            logger.warning("%s: no matching brain mask at %s -- skipping", label_path.name, mask_path)
            results.append(
                MaskResult(label=label_path.name, label_path=label_path, status="no_mask_found")
            )
            continue

        output_path = output_dir / label_path.name
        logger.info("[%d/%d] %s: applying brain mask from %s", i, len(labels), label_path.name, mask_path.name)
        results.append(apply_mask_to_label(label_path, mask_path, output_path))

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(manifest_csv, index=False)

    n_ok = int((df["status"] == "ok").sum())
    n_failed = int((df["status"] == "failed").sum())
    n_no_mask = int((df["status"] == "no_mask_found").sum())
    logger.info("Done: %d ok, %d failed, %d had no matching mask. Manifest: %s", n_ok, n_failed, n_no_mask, manifest_csv)

    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a skull-strip brain mask (e.g. from HD-BET) to label/segmentation files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels-dir", required=True, type=Path, help="Folder of registered label/segmentation files")
    parser.add_argument(
        "--masks-dir",
        required=True,
        type=Path,
        help="Folder containing HD-BET's '<scan_stem>_bet.nii.gz' brain masks (HD-BET's --save_bet_mask output dir)",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Where masked labels are written")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --labels-dir) for label files")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search --labels-dir recursively",
    )
    parser.add_argument(
        "--naming-scheme",
        choices=sorted(NAMING_SCHEMES),
        default="identical",
        help="How to derive a scan filename from its label's filename: 'identical' (same name), or "
        "'braintracking' (label 'tumor_2016-11.nii.gz' -> scan 'flair_2016_11.nii.gz')",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Path to write the per-label manifest CSV (default: 'brain_mask_manifest.csv' next to --output-dir)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    manifest_csv = args.manifest_csv or (args.output_dir.parent / "brain_mask_manifest.csv")

    run(
        labels_dir=args.labels_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        naming_scheme=args.naming_scheme,
        manifest_csv=manifest_csv,
    )


if __name__ == "__main__":
    main()
