#!/usr/bin/env python3
"""Compute segmentation metrics (Dice, HD95, volume, instance count) between
ground-truth and predicted masks, without a specialized third-party metrics
package (no medpy, MONAI, SimpleITK, or seg-metrics) -- just numpy/scipy/nibabel.

Ground truth is expected to be binary (a single "tumor" class). If the
matching prediction has multiple labels instead (e.g. 0/1/2/3 for
background/necrotic/edema/enhancing), they're merged into one binary
foreground before any metric is computed. If gt itself isn't binary, both
sides are binarized on a whole-foreground basis instead -- logged either way.

Usage
-----
    python utils/compute_seg_metrics.py \\
        --gt-dir /data/patient1/ground_truth \\
        --seg-dir /data/patient1/segresnet_preds \\
        --output-csv /data/patient1/seg_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from tumor_tracking import connectivity_structure, dice_score, extract_components

logger = logging.getLogger("compute_seg_metrics")


@dataclass
class MetricSettings:
    connectivity: int = 26
    min_volume_mm3: float = 20.0


# --------------------------------------------------------------------------
# NIfTI IO
# --------------------------------------------------------------------------


def load_nifti_raw(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """Load a NIfTI file's raw integer label map (not yet binarized)."""
    image = nib.load(str(path))
    data = np.asarray(image.dataobj).round().astype(np.int32)
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing, image.affine


# --------------------------------------------------------------------------
# Label handling: merge seg labels into one binary mask to match a binary gt
# --------------------------------------------------------------------------


def to_binary_masks(
    gt_raw: np.ndarray, seg_raw: np.ndarray, case_id: str
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], bool]:
    """Binarize gt/seg into comparable foreground masks, merging seg labels if gt is binary but seg isn't."""
    gt_labels = sorted(int(v) for v in np.unique(gt_raw))
    seg_labels = sorted(int(v) for v in np.unique(seg_raw))

    gt_is_binary = set(gt_labels) <= {0, 1}
    seg_is_binary = set(seg_labels) <= {0, 1}
    seg_merged = False

    if not gt_is_binary:
        logger.warning(
            "%s: ground truth is not binary (labels=%s) -- comparing gt and seg "
            "on a binarized whole-foreground basis instead of per-label",
            case_id,
            gt_labels,
        )
        seg_merged = not seg_is_binary
    elif not seg_is_binary:
        logger.info(
            "%s: seg has multiple labels %s but gt is binary -- merging seg "
            "labels into one binary foreground to match gt",
            case_id,
            seg_labels,
        )
        seg_merged = True

    return gt_raw > 0, seg_raw > 0, gt_labels, seg_labels, seg_merged


# --------------------------------------------------------------------------
# HD95 (Dice is reused as-is from tumor_tracking.dice_score)
# --------------------------------------------------------------------------


def surface_voxels(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Boolean mask of foreground voxels that touch the background."""
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def hausdorff_distance_95mm(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing: tuple[float, float, float],
    structure: np.ndarray,
) -> float:
    """95th-percentile symmetric Hausdorff distance (mm): max of the two directed
    95th-percentile surface distances -- robust to a single outlier voxel, unlike
    the true (100th percentile) Hausdorff distance. NaN if only one mask is
    empty (undefined), 0.0 if both are (perfect agreement).
    """
    a_empty, b_empty = not mask_a.any(), not mask_b.any()
    if a_empty and b_empty:
        return 0.0
    if a_empty or b_empty:
        return float("nan")

    surf_a = surface_voxels(mask_a, structure)
    surf_b = surface_voxels(mask_b, structure)

    dist_to_a = ndimage.distance_transform_edt(~mask_a, sampling=spacing)
    dist_to_b = ndimage.distance_transform_edt(~mask_b, sampling=spacing)

    d_b_to_a = dist_to_a[surf_b]
    d_a_to_b = dist_to_b[surf_a]

    return float(max(np.percentile(d_b_to_a, 95), np.percentile(d_a_to_b, 95)))


# --------------------------------------------------------------------------
# Per-case metrics
# --------------------------------------------------------------------------


def compute_case_metrics(case_id: str, gt_path: Path, seg_path: Path, settings: MetricSettings) -> dict:
    gt_raw, gt_spacing, gt_affine = load_nifti_raw(gt_path)
    seg_raw, _seg_spacing, seg_affine = load_nifti_raw(seg_path)

    if gt_raw.shape != seg_raw.shape:
        raise ValueError(
            f"{case_id}: gt shape {gt_raw.shape} != seg shape {seg_raw.shape} "
            "-- both must be on the same voxel grid to compute voxelwise metrics"
        )
    if not np.allclose(gt_affine, seg_affine, atol=1e-2):
        logger.warning("%s: gt/seg affines differ slightly; proceeding using gt's spacing for physical units", case_id)

    gt_mask, seg_mask, gt_labels, seg_labels, seg_merged = to_binary_masks(gt_raw, seg_raw, case_id)
    spacing = gt_spacing
    voxel_vol = float(np.prod(spacing))
    structure = connectivity_structure(settings.connectivity)

    _, gt_instance_count = extract_components(gt_mask, settings.connectivity, settings.min_volume_mm3, spacing)
    _, seg_instance_count = extract_components(seg_mask, settings.connectivity, settings.min_volume_mm3, spacing)

    gt_voxel_count = int(gt_mask.sum())
    seg_voxel_count = int(seg_mask.sum())
    gt_volume_mm3 = gt_voxel_count * voxel_vol
    seg_volume_mm3 = seg_voxel_count * voxel_vol

    return {
        "case_id": case_id,
        "gt_path": str(gt_path),
        "seg_path": str(seg_path),
        "gt_labels_raw": gt_labels,
        "seg_labels_raw": seg_labels,
        "seg_labels_merged": seg_merged,
        "dice": dice_score(gt_mask, seg_mask),
        "hd95_mm": hausdorff_distance_95mm(gt_mask, seg_mask, spacing, structure),
        "gt_voxel_count": gt_voxel_count,
        "seg_voxel_count": seg_voxel_count,
        "gt_volume_mm3": gt_volume_mm3,
        "seg_volume_mm3": seg_volume_mm3,
        "volume_diff_mm3": seg_volume_mm3 - gt_volume_mm3,
        "volume_diff_pct": (seg_volume_mm3 - gt_volume_mm3) / gt_volume_mm3 if gt_volume_mm3 > 0 else float("nan"),
        "gt_instance_count": gt_instance_count,
        "seg_instance_count": seg_instance_count,
        "instance_count_diff": seg_instance_count - gt_instance_count,
    }


# --------------------------------------------------------------------------
# Discovery: match gt/seg files by filename, via a pluggable naming scheme
# --------------------------------------------------------------------------


def identical_gt_to_seg_filename(gt_filename: str) -> str:
    """Default scheme: seg file has the exact same filename as gt."""
    return gt_filename


def braintracking_gt_to_seg_filename(gt_filename: str) -> str:
    """Map gt "tumor_2016-11.nii.gz" to its matching seg "flair_2016_11.nii.gz"."""
    if gt_filename.endswith(".nii.gz"):
        stem, suffix = gt_filename[: -len(".nii.gz")], ".nii.gz"
    elif gt_filename.endswith(".nii"):
        stem, suffix = gt_filename[: -len(".nii")], ".nii"
    else:
        raise ValueError(f"Not a NIfTI filename: {gt_filename!r}")

    gt_prefix = "tumor_"
    if not stem.startswith(gt_prefix):
        raise ValueError(f"Expected gt filename to start with '{gt_prefix}', got {gt_filename!r}")

    date_part = stem[len(gt_prefix):].replace("-", "_")
    return f"flair_{date_part}{suffix}"


NAMING_SCHEMES = {
    "identical": identical_gt_to_seg_filename,
    "braintracking": braintracking_gt_to_seg_filename,
}


def discover_pairs(gt_dir: Path, seg_dir: Path, pattern: str, naming_scheme: str = "identical") -> list[tuple[str, Path, Path]]:
    gt_files = sorted(gt_dir.glob(pattern))
    if not gt_files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {gt_dir}")

    map_gt_to_seg_filename = NAMING_SCHEMES[naming_scheme]

    pairs = []
    for gt_path in gt_files:
        seg_path = seg_dir / map_gt_to_seg_filename(gt_path.name)
        if not seg_path.is_file():
            logger.warning("%s: no matching seg file at %s -- skipping", gt_path.name, seg_path)
            continue
        case_id = gt_path.name.removesuffix(".nii.gz").removesuffix(".nii")
        pairs.append((case_id, gt_path, seg_path))

    if not pairs:
        raise FileNotFoundError(f"None of the {len(gt_files)} gt file(s) in {gt_dir} had a matching file in {seg_dir}")

    return pairs


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    gt_dir: Path,
    seg_dir: Path,
    output_csv: Path,
    pattern: str,
    settings: MetricSettings,
    naming_scheme: str = "identical",
) -> pd.DataFrame:
    pairs = discover_pairs(gt_dir, seg_dir, pattern, naming_scheme)
    logger.info("Found %d matched gt/seg case(s)", len(pairs))

    rows = []
    for case_id, gt_path, seg_path in pairs:
        try:
            rows.append(compute_case_metrics(case_id, gt_path, seg_path, settings))
        except Exception:
            logger.exception("%s: failed to compute metrics -- skipping", case_id)

    if not rows:
        logger.warning("No cases produced metrics; nothing written.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_csv)

    n_merged = int(df["seg_labels_merged"].sum())
    logger.info("%d/%d case(s) had seg labels merged to match a binary gt", n_merged, len(df))

    summary_cols = ["dice", "hd95_mm", "gt_volume_mm3", "seg_volume_mm3", "gt_instance_count", "seg_instance_count"]
    logger.info(
        "Summary (mean over %d case(s), ignoring NaN):\n%s",
        len(df),
        df[summary_cols].mean(numeric_only=True, skipna=True).to_string(),
    )

    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Dice/HD95/volume/instance-count metrics between ground-truth and predicted segmentation masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt-dir", required=True, type=Path, help="Folder of ground-truth NIfTI masks")
    parser.add_argument("--seg-dir", required=True, type=Path, help="Folder of predicted masks, matched to gt files by identical filename")
    parser.add_argument("--output-csv", required=True, type=Path, help="Path to write the per-case metrics CSV")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --gt-dir) for ground-truth files to evaluate")
    parser.add_argument(
        "--naming-scheme",
        choices=list(NAMING_SCHEMES),
        default="identical",
        help="How to map a gt filename to its matching seg filename. 'identical': same filename in both dirs. "
        "'braintracking': gt 'tumor_2016-11.nii.gz' <-> seg 'flair_2016_11.nii.gz'",
    )

    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26, help="Connectivity used for instance counting")
    parser.add_argument("--min-volume-mm3", type=float, default=20.0, help="Drop components smaller than this when counting instances (noise filter)")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    settings = MetricSettings(connectivity=args.connectivity, min_volume_mm3=args.min_volume_mm3)

    run(
        gt_dir=args.gt_dir,
        seg_dir=args.seg_dir,
        output_csv=args.output_csv,
        pattern=args.pattern,
        settings=settings,
        naming_scheme=args.naming_scheme,
    )


if __name__ == "__main__":
    main()
