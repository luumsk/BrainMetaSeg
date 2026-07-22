#!/usr/bin/env python3
"""Track tumor instances across timepoints and measure their volume over time.

Each file is a binary segmentation mask for one scan date (may contain
multiple disconnected tumor instances, e.g. several metastases). The scan
date is parsed from the filename (default pattern expects "YYYY-MM" or
"YYYY_MM" right before ".nii.gz", e.g. "tumor_2021-03.nii.gz" or
"flair_2021_03.nii.gz"; override --filename-pattern if yours differs) so
timepoints are ordered chronologically and real day-deltas can be computed
between them.

Per-timepoint volume is always computed from that file's own header, so
timepoints do NOT need to share a grid. Cross-timepoint matching, however,
auto-detects whether two consecutive timepoints are actually co-registered
(same shape and affine):
* If they are, instances are matched with a Hungarian assignment on voxel
  overlap/Dice/centroid/surface-distance -- accurate spatial matching.
* If they aren't (common for real longitudinal scans: different FOV, slice
  thickness, or scanner between visits), it falls back to matching by
  descending-volume rank (see `match_by_volume_rank`), which is a much
  weaker assumption and can misfire if the number or relative size of
  tumors changes between visits. Register scans to a common grid first if
  you need reliable identity tracking with multiple simultaneous tumors.

What this does now
-------------------
1. Extracts 3D connected components ("tumor instances") per timepoint.
2. Matches instances across consecutive timepoints and assigns a stable
   `tumor_id` to each track (see matching behavior above).
3. Measures volume (+ basic centroid/bbox geometry) per instance per
   timepoint and writes one long-format CSV, ready to group by `tumor_id` and
   plot volume vs. date.

Extension points (not implemented yet, by design)
--------------------------------------------------
* Boundary contrast against a co-registered intensity image (e.g. T1c):
  add an `image` argument to `compute_lesion_metrics()` and sample a ring
  around `mask` (e.g. via scipy.ndimage.distance_transform_edt) the same way
  volume is sampled now.
* Boundary/shape tracking over time: `--save-cc-maps` already persists the
  labeled connected-component volume for every timepoint, so a later script
  can load `{timepoint}_cc.nii.gz` + this CSV's `tumor_id`/`component_id`
  mapping without recomputing connected components from scratch.

Usage
-----
    python tumor_tracking.py \\
        --seg-dir /data/patient1/segmentations \\
        --output-csv /data/patient1/tumor_volumes.csv \\
        --save-cc-maps
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger("tumor_tracking")

DEFAULT_FILENAME_PATTERN = r"(?P<date>\d{4}[-_]\d{2})\.nii\.gz$"


@dataclass
class Settings:
    connectivity: int = 26
    min_volume_mm3: float = 8.0
    match_centroid_threshold_mm: float = 10.0
    match_surface_threshold_mm: float = 5.0


@dataclass
class TimepointFile:
    label: str  # normalized "YYYY-MM", regardless of "-" or "_" in the filename
    date: pd.Timestamp
    path: Path


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover_timepoints(seg_dir: Path, filename_pattern: str) -> list[TimepointFile]:
    pattern = re.compile(filename_pattern)
    timepoints = []

    for path in seg_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if not match:
            continue
        label = match.group("date").replace("_", "-")
        timepoints.append(TimepointFile(label=label, date=pd.to_datetime(label), path=path))

    if not timepoints:
        raise FileNotFoundError(f"No files matching pattern '{filename_pattern}' found in {seg_dir}")

    timepoints.sort(key=lambda tp: tp.date)
    return timepoints


# --------------------------------------------------------------------------
# NIfTI IO
# --------------------------------------------------------------------------


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj) > 0
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return data, spacing, image.affine


# --------------------------------------------------------------------------
# Connected components
# --------------------------------------------------------------------------


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(f"connectivity must be 6, 18, or 26. Got: {connectivity}")


def extract_components(
    mask: np.ndarray,
    connectivity: int,
    min_volume_mm3: float,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, int]:
    """Label connected tumor instances, drop tiny ones (segmentation noise).

    Remaining instances are renumbered 1..k, ordered by descending volume so
    component 1 is always the largest tumor at that timepoint.
    """
    structure = connectivity_structure(connectivity)
    raw_map, n_raw = ndimage.label(mask, structure=structure)
    if n_raw == 0:
        return raw_map.astype(np.int32), 0

    voxel_vol = float(np.prod(spacing))
    sizes = ndimage.sum_labels(mask, raw_map, index=np.arange(1, n_raw + 1))
    volumes_mm3 = sizes * voxel_vol

    keep_ids = np.where(volumes_mm3 >= min_volume_mm3)[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(raw_map, dtype=np.int32), 0

    order = np.argsort(-volumes_mm3[keep_ids - 1])
    ordered_old_ids = keep_ids[order]

    remap = np.zeros(n_raw + 1, dtype=np.int32)
    for new_id, old_id in enumerate(ordered_old_ids, start=1):
        remap[old_id] = new_id

    return remap[raw_map].astype(np.int32), int(ordered_old_ids.size)


def component_ids(cc_map: np.ndarray) -> list[int]:
    ids = np.unique(cc_map)
    return [int(i) for i in ids if i != 0]


def bbox_slices(mask: np.ndarray, pad: int, shape: tuple[int, ...]) -> tuple[slice, ...]:
    coords = np.argwhere(mask)
    mins = np.clip(coords.min(axis=0) - pad, 0, None)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, np.array(shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))


# --------------------------------------------------------------------------
# Per-instance metrics
#
# This is the extension point for future per-lesion measurements (contrast,
# shape, boundary descriptors, ...): add more key/value pairs to the dict
# returned here, threading through whatever extra inputs (e.g. an intensity
# image) they need.
# --------------------------------------------------------------------------


def compute_lesion_metrics(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    voxel_vol = float(np.prod(spacing))
    voxel_count = int(mask.sum())
    volume_mm3 = voxel_count * voxel_vol

    coords = np.argwhere(mask)
    centroid_mm = coords.mean(axis=0) * np.array(spacing)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)

    return {
        "voxel_count": voxel_count,
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_mm3 / 1000.0),
        "centroid_z_mm": float(centroid_mm[0]),
        "centroid_y_mm": float(centroid_mm[1]),
        "centroid_x_mm": float(centroid_mm[2]),
        "bbox_z_min": int(bbox_min[0]),
        "bbox_y_min": int(bbox_min[1]),
        "bbox_x_min": int(bbox_min[2]),
        "bbox_z_max": int(bbox_max[0]),
        "bbox_y_max": int(bbox_max[1]),
        "bbox_x_max": int(bbox_max[2]),
    }


def compute_timepoint_metrics(cc_map: np.ndarray, spacing: tuple[float, float, float]) -> pd.DataFrame:
    rows = []
    for cid in component_ids(cc_map):
        row = {"component_id": cid}
        row.update(compute_lesion_metrics(cc_map == cid, spacing))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cross-timepoint matching (Hungarian assignment)
# --------------------------------------------------------------------------


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    size_sum = mask_a.sum() + mask_b.sum()
    if size_sum == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(2.0 * intersection / size_sum)


def min_surface_distance_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Min distance (mm) between any voxel of A and any voxel of B.

    Uses a distance transform on a padded shared bounding box, which stays
    fast even for large tumors (avoids O(n*m) pairwise point distances).
    """
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("inf")

    sl = bbox_slices(mask_a | mask_b, pad=2, shape=mask_a.shape)
    crop_a, crop_b = mask_a[sl], mask_b[sl]
    dist_from_a = ndimage.distance_transform_edt(~crop_a, sampling=spacing)
    return float(dist_from_a[crop_b].min())


def build_cost_matrix(
    cc_a: np.ndarray,
    cc_b: np.ndarray,
    ids_a: list[int],
    ids_b: list[int],
    spacing: tuple[float, float, float],
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Cost matrix for Hungarian assignment, plus a boolean candidate mask.

    A pair is only a valid candidate if it overlaps or is close by
    centroid/surface distance -- this keeps far-apart tumors from ever being
    matched even when the assignment would otherwise be unbalanced.
    """
    n_a, n_b = len(ids_a), len(ids_b)
    cost = np.full((n_a, n_b), np.inf)
    is_candidate = np.zeros((n_a, n_b), dtype=bool)

    for i, id_a in enumerate(ids_a):
        mask_a = cc_a == id_a
        centroid_a = np.argwhere(mask_a).mean(axis=0) * np.array(spacing)

        for j, id_b in enumerate(ids_b):
            mask_b = cc_b == id_b
            intersection = int(np.logical_and(mask_a, mask_b).sum())
            centroid_b = np.argwhere(mask_b).mean(axis=0) * np.array(spacing)
            centroid_distance = float(np.linalg.norm(centroid_a - centroid_b))

            has_overlap = intersection > 0
            close_centroid = centroid_distance <= settings.match_centroid_threshold_mm

            surface_distance = np.inf
            if has_overlap or close_centroid:
                surface_distance = 0.0 if has_overlap else min_surface_distance_mm(mask_a, mask_b, spacing)
                close_surface = surface_distance <= settings.match_surface_threshold_mm
            else:
                close_surface = False

            if not (has_overlap or close_centroid or close_surface):
                continue

            dice = dice_score(mask_a, mask_b)
            is_candidate[i, j] = True
            cost[i, j] = (
                -1000.0 * intersection
                - 10.0 * dice
                + 1.0 * centroid_distance
                + 0.5 * (surface_distance if np.isfinite(surface_distance) else 0.0)
            )

    return cost, is_candidate


def match_timepoint_pair(
    cc_a: np.ndarray, cc_b: np.ndarray, spacing: tuple[float, float, float], settings: Settings
) -> dict[int, int]:
    """One-to-one match of components in cc_a to components in cc_b."""
    ids_a, ids_b = component_ids(cc_a), component_ids(cc_b)
    if not ids_a or not ids_b:
        return {}

    cost, is_candidate = build_cost_matrix(cc_a, cc_b, ids_a, ids_b, spacing, settings)
    if not is_candidate.any():
        return {}

    large_cost = 1e12
    cost_for_solver = np.where(is_candidate, cost, large_cost)
    row_ind, col_ind = linear_sum_assignment(cost_for_solver)

    return {ids_a[i]: ids_b[j] for i, j in zip(row_ind, col_ind) if is_candidate[i, j]}


def is_same_grid(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...], affine_a: np.ndarray, affine_b: np.ndarray
) -> bool:
    """Whether two timepoints share one voxel grid (so array indices line up).

    Longitudinal clinical scans are frequently *not* resliced to a common
    grid between visits (different FOV/matrix, slice thickness, or even
    scanner). Voxel-overlap/Dice/surface-distance are only meaningful when
    this is true; otherwise fall back to `match_by_volume_rank`.
    """
    return shape_a == shape_b and np.allclose(affine_a, affine_b, atol=1e-2)


def match_by_volume_rank(ids_a: list[int], ids_b: list[int]) -> dict[int, int]:
    """Fallback match for timepoints that aren't on a common grid.

    `extract_components` already numbers components 1..k by descending
    volume within each timepoint, so pairing id i in A with id i in B is
    equivalent to matching largest-to-largest, 2nd-largest-to-2nd-largest,
    etc. This is a much weaker assumption than true spatial correspondence
    (it can misfire if the number or size-ranking of tumors changes between
    visits) but it's the best available signal without registering the
    scans to a shared grid first.
    """
    return dict(zip(ids_a, ids_b))


# --------------------------------------------------------------------------
# Global track construction across all timepoints
# --------------------------------------------------------------------------


def build_global_tracks(
    timepoint_labels: list[str],
    cc_maps: dict[str, np.ndarray],
    spacings: dict[str, tuple[float, float, float]],
    affines: dict[str, np.ndarray],
    settings: Settings,
    id_prefix: str,
) -> pd.DataFrame:
    """Chain consecutive-timepoint matches into global tumor tracks.

    Only directly consecutive timepoints are matched (no gap-bridging across
    a timepoint where a tumor was missed) -- simple and robust; revisit if
    you need to bridge dropouts.
    """
    next_id = 1
    active_tracks: dict[int, str] = {}
    records = []  # (timepoint, component_id, tumor_id)

    first_tp = timepoint_labels[0]
    for cid in component_ids(cc_maps[first_tp]):
        tumor_id = f"{id_prefix}{next_id:04d}"
        next_id += 1
        active_tracks[cid] = tumor_id
        records.append((first_tp, cid, tumor_id))

    for prev_tp, cur_tp in zip(timepoint_labels[:-1], timepoint_labels[1:]):
        cc_a, cc_b = cc_maps[prev_tp], cc_maps[cur_tp]

        if is_same_grid(cc_a.shape, cc_b.shape, affines[prev_tp], affines[cur_tp]):
            matches = match_timepoint_pair(cc_a, cc_b, spacings[prev_tp], settings)
        else:
            logger.warning(
                "%s -> %s: timepoints are not on a common grid (different shape/affine); "
                "falling back to matching by descending-volume rank instead of spatial overlap. "
                "Register scans to a shared grid first for reliable identity-based tracking "
                "when multiple tumors can be present at once.",
                prev_tp,
                cur_tp,
            )
            matches = match_by_volume_rank(component_ids(cc_a), component_ids(cc_b))

        new_active_tracks: dict[int, str] = {}
        matched_cur_ids = set()

        for prev_cid, tumor_id in active_tracks.items():
            cur_cid = matches.get(prev_cid)
            if cur_cid is not None:
                new_active_tracks[cur_cid] = tumor_id
                matched_cur_ids.add(cur_cid)
                records.append((cur_tp, cur_cid, tumor_id))
            # else: track ends here (tumor disappeared / was not matched)

        for cid in component_ids(cc_maps[cur_tp]):
            if cid not in matched_cur_ids:
                tumor_id = f"{id_prefix}{next_id:04d}"
                next_id += 1
                new_active_tracks[cid] = tumor_id
                records.append((cur_tp, cid, tumor_id))

        active_tracks = new_active_tracks

    return pd.DataFrame(records, columns=["timepoint", "component_id", "tumor_id"])


# --------------------------------------------------------------------------
# Growth deltas (safe to compute from volume alone; no extra inputs needed)
# --------------------------------------------------------------------------


def add_growth_columns(df: pd.DataFrame, dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    df = df.copy()
    df["_date"] = df["timepoint"].map(dates)
    df = df.sort_values(["tumor_id", "_date"]).reset_index(drop=True)

    prev_volume = df.groupby("tumor_id")["volume_mm3"].shift(1)
    prev_date = df.groupby("tumor_id")["_date"].shift(1)

    df["days_since_previous"] = (df["_date"] - prev_date).dt.total_seconds() / 86400.0
    df["volume_change_mm3"] = df["volume_mm3"] - prev_volume
    df["volume_change_pct"] = np.where(prev_volume > 0, df["volume_change_mm3"] / prev_volume, np.nan)
    df["is_new_tumor"] = prev_volume.isna()

    return df.drop(columns=["_date"])


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    seg_dir: Path,
    output_csv: Path,
    patient_id: str,
    filename_pattern: str,
    settings: Settings,
    save_cc_maps: bool,
    cc_maps_dir: Path,
) -> pd.DataFrame:
    timepoints = discover_timepoints(seg_dir, filename_pattern)
    logger.info("Found %d timepoints in %s", len(timepoints), seg_dir)

    cc_maps: dict[str, np.ndarray] = {}
    per_tp_metrics: dict[str, pd.DataFrame] = {}
    dates: dict[str, pd.Timestamp] = {}
    spacings: dict[str, tuple[float, float, float]] = {}
    affines: dict[str, np.ndarray] = {}

    for tp in timepoints:
        mask, tp_spacing, tp_affine = load_mask(tp.path)

        # Timepoints are NOT required to share one grid: real longitudinal
        # scans routinely differ in shape/spacing/position between visits
        # (different FOV, slice thickness, or scanner). Each timepoint's
        # volume is always computed with its own spacing; cross-timepoint
        # matching auto-detects whether voxel-overlap is meaningful (see
        # `is_same_grid` / `build_global_tracks`).
        cc_map, n_components = extract_components(mask, settings.connectivity, settings.min_volume_mm3, tp_spacing)
        cc_maps[tp.label] = cc_map
        dates[tp.label] = tp.date
        spacings[tp.label] = tp_spacing
        affines[tp.label] = tp_affine
        logger.info("%s: %d tumor instance(s)", tp.label, n_components)

        per_tp_metrics[tp.label] = (
            compute_timepoint_metrics(cc_map, tp_spacing) if n_components else pd.DataFrame()
        )

        if save_cc_maps:
            cc_maps_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(cc_map.astype(np.int16), tp_affine), str(cc_maps_dir / f"{tp.label}_cc.nii.gz"))

    timepoint_labels = [tp.label for tp in timepoints]
    if not any(not per_tp_metrics[label].empty for label in timepoint_labels):
        logger.warning("No tumor instances found in any timepoint above min-volume threshold.")
        return pd.DataFrame()

    tracks_df = build_global_tracks(
        timepoint_labels, cc_maps, spacings, affines, settings, id_prefix=f"{patient_id}_T"
    )

    metrics_long = pd.concat(
        (df.assign(timepoint=label) for label, df in per_tp_metrics.items() if not df.empty),
        ignore_index=True,
    )

    merged = tracks_df.merge(metrics_long, on=["timepoint", "component_id"], how="left")
    merged = add_growth_columns(merged, dates)
    merged.insert(0, "date", merged["timepoint"].map(dates))
    merged.insert(0, "patient_id", patient_id)
    merged = merged.sort_values(["tumor_id", "date"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info("Wrote %d rows to %s", len(merged), output_csv)

    return merged


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track tumor instances across dated segmentation files and measure volume over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seg-dir", required=True, type=Path, help="Folder containing dated segmentation files, e.g. tumor_YYYY-MM.nii.gz or flair_YYYY_MM.nii.gz")
    parser.add_argument("--output-csv", required=True, type=Path, help="Path to write the long-format tumor volume/tracking CSV")
    parser.add_argument("--patient-id", default=None, help="Label for the patient_id column (defaults to the --seg-dir folder name)")
    parser.add_argument(
        "--filename-pattern",
        default=DEFAULT_FILENAME_PATTERN,
        help="Regex with a named group 'date' (YYYY-MM or YYYY_MM) used to find and order segmentation files",
    )

    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--min-volume-mm3", type=float, default=8.0, help="Drop components smaller than this (filters segmentation noise)")
    parser.add_argument("--match-centroid-threshold-mm", type=float, default=10.0, help="Max centroid distance to consider two tumors a candidate match")
    parser.add_argument("--match-surface-threshold-mm", type=float, default=5.0, help="Max surface-to-surface distance to consider two tumors a candidate match")

    parser.add_argument(
        "--save-cc-maps",
        action="store_true",
        help="Also save labeled connected-component NIfTI volumes per timepoint (reusable input for later contrast/boundary analysis)",
    )
    parser.add_argument(
        "--cc-maps-dir",
        type=Path,
        default=None,
        help="Where to save labeled component maps if --save-cc-maps is set (defaults to a 'connected_components' folder next to --output-csv)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    settings = Settings(
        connectivity=args.connectivity,
        min_volume_mm3=args.min_volume_mm3,
        match_centroid_threshold_mm=args.match_centroid_threshold_mm,
        match_surface_threshold_mm=args.match_surface_threshold_mm,
    )

    patient_id = args.patient_id or args.seg_dir.name
    cc_maps_dir = args.cc_maps_dir or (args.output_csv.parent / "connected_components")

    run(
        seg_dir=args.seg_dir,
        output_csv=args.output_csv,
        patient_id=patient_id,
        filename_pattern=args.filename_pattern,
        settings=settings,
        save_cc_maps=args.save_cc_maps,
        cc_maps_dir=cc_maps_dir,
    )


if __name__ == "__main__":
    main()
