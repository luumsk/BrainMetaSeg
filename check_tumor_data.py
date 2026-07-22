#!/usr/bin/env python3
"""Pre-flight data-quality report for a folder of dated tumor segmentation files.

Run this BEFORE tumor_tracking.py to surface issues that would otherwise
silently produce wrong or low-confidence results: mismatched shapes, scans
that aren't co-registered to each other, unexpected (non-binary) label
values, empty masks, corrupt files, duplicate/unparsable timepoints, and
unusually large gaps between visits.

Expected input
--------------
    seg_dir/
        tumor_2021-03.nii.gz
        tumor_2021-07.nii.gz
        ...
    (or "YYYY_MM" with underscores, e.g. flair_2021_03.nii.gz)
Same file/date convention as tumor_tracking.py (override --filename-pattern
if yours differs).

Output
------
Writes two files next to --output-prefix:
* <prefix>.csv -- one row per input file, all per-file diagnostics.
* <prefix>.txt -- human-readable report: one section per file, one section
  per consecutive-timepoint comparison, and a final summary.

Usage
-----
    python check_tumor_data.py --seg-dir /path/to/tumor_volume
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


logger = logging.getLogger("check_tumor_data")

DEFAULT_FILENAME_PATTERN = r"(?P<date>\d{4}[-_]\d{2})\.nii\.gz$"
DEFAULT_AFFINE_ATOL = 1e-2  # matches the tolerance tumor_tracking.py uses to decide co-registration


# --------------------------------------------------------------------------
# Per-file inspection
# --------------------------------------------------------------------------


@dataclass
class FileReport:
    label: str
    date: Optional[pd.Timestamp]
    path: Path
    loadable: bool = True
    error: str = ""
    shape: Optional[tuple] = None
    spacing: Optional[tuple] = None
    affine: Optional[np.ndarray] = None
    dtype: str = ""
    unique_values: list = field(default_factory=list)
    is_binary: bool = False
    has_nan_or_inf: bool = False
    voxel_count: int = 0
    volume_mm3: float = 0.0
    n_components: int = 0
    component_volumes_mm3: list = field(default_factory=list)
    n_small_components: int = 0
    is_empty: bool = False
    issues: list = field(default_factory=list)


def discover_files(seg_dir: Path, filename_pattern: str) -> list[tuple[str, Optional[pd.Timestamp], Path]]:
    pattern = re.compile(filename_pattern)
    found = []
    for path in seg_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.search(path.name)
        if not match:
            continue
        label = match.group("date").replace("_", "-")
        try:
            date = pd.to_datetime(label)
        except (ValueError, TypeError):
            date = None
        found.append((label, date, path))
    return found


def connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndimage.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndimage.generate_binary_structure(3, 2)
    if connectivity == 26:
        return np.ones((3, 3, 3), dtype=np.uint8)
    raise ValueError(f"connectivity must be 6, 18, or 26. Got: {connectivity}")


def inspect_file(
    label: str, date: Optional[pd.Timestamp], path: Path, connectivity: int, min_volume_mm3: float
) -> FileReport:
    report = FileReport(label=label, date=date, path=path)

    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
    except Exception as exc:  # noqa: BLE001 -- any load failure is itself the finding
        report.loadable = False
        report.error = f"{type(exc).__name__}: {exc}"
        report.issues.append(f"FILE FAILED TO LOAD: {report.error}")
        return report

    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    report.shape = data.shape
    report.spacing = spacing
    report.affine = image.affine
    report.dtype = str(data.dtype)

    if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
        report.has_nan_or_inf = True
        report.issues.append("Contains NaN/Inf voxel values")

    unique_values = np.unique(data)
    report.unique_values = [float(v) for v in unique_values[:10]]
    report.is_binary = set(np.round(unique_values).astype(int).tolist()) <= {0, 1}
    if not report.is_binary:
        suffix = ", ..." if unique_values.size > 10 else ""
        report.issues.append(f"Not a binary mask -- unique values include {report.unique_values}{suffix}")

    mask = data > 0
    report.voxel_count = int(mask.sum())
    voxel_vol = float(np.prod(spacing))
    report.volume_mm3 = report.voxel_count * voxel_vol

    if report.voxel_count == 0:
        report.is_empty = True
        report.issues.append("Empty mask (no tumor voxels)")
    else:
        structure = connectivity_structure(connectivity)
        cc_map, n_components = ndimage.label(mask, structure=structure)
        sizes = ndimage.sum_labels(mask, cc_map, index=np.arange(1, n_components + 1)) * voxel_vol
        report.n_components = int(n_components)
        report.component_volumes_mm3 = [float(v) for v in sorted(sizes, reverse=True)]
        report.n_small_components = int(np.sum(sizes < min_volume_mm3))

        if report.n_small_components:
            report.issues.append(
                f"{report.n_small_components} of {n_components} component(s) are below the "
                f"{min_volume_mm3}mm3 noise threshold and would be dropped by tumor_tracking.py"
            )
        n_kept = n_components - report.n_small_components
        if n_kept > 1:
            report.issues.append(f"{n_kept} distinct tumor instance(s) above threshold at this timepoint")

    if any(v <= 0 for v in spacing):
        report.issues.append(f"Non-positive voxel spacing in header: {spacing}")

    return report


# --------------------------------------------------------------------------
# Consecutive-timepoint comparison
# --------------------------------------------------------------------------


@dataclass
class PairReport:
    label_a: str
    label_b: str
    days_between: Optional[float]
    same_shape: bool
    same_spacing: bool
    same_affine: bool
    max_affine_diff: float
    issues: list = field(default_factory=list)

    @property
    def is_coregistered(self) -> bool:
        return self.same_shape and self.same_affine


def compare_pair(a: FileReport, b: FileReport, affine_atol: float, max_gap_days: float) -> PairReport:
    days_between = None
    if a.date is not None and b.date is not None:
        days_between = (b.date - a.date).total_seconds() / 86400.0

    if not a.loadable or not b.loadable:
        return PairReport(
            label_a=a.label,
            label_b=b.label,
            days_between=days_between,
            same_shape=False,
            same_spacing=False,
            same_affine=False,
            max_affine_diff=float("nan"),
            issues=["Cannot compare -- one or both files failed to load"],
        )

    same_shape = a.shape == b.shape
    same_spacing = bool(np.allclose(a.spacing, b.spacing, atol=1e-3))
    max_affine_diff = float(np.max(np.abs(a.affine - b.affine)))
    same_affine = max_affine_diff <= affine_atol

    issues = []
    if not same_shape:
        issues.append(f"Shape mismatch: {a.shape} vs {b.shape}")
    if not same_spacing:
        issues.append(f"Voxel spacing changed: {a.spacing} vs {b.spacing}")
    if not same_affine:
        issues.append(f"Not co-registered -- affine differs by up to {max_affine_diff:.3f} (tolerance {affine_atol})")
    if days_between is not None and days_between <= 0:
        issues.append(f"Non-chronological or duplicate date ({days_between:.0f} days since previous)")
    elif days_between is not None and days_between > max_gap_days:
        issues.append(f"Large gap between visits: {days_between:.0f} days")

    return PairReport(
        label_a=a.label,
        label_b=b.label,
        days_between=days_between,
        same_shape=same_shape,
        same_spacing=same_spacing,
        same_affine=same_affine,
        max_affine_diff=max_affine_diff,
        issues=issues,
    )


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def build_csv_rows(file_reports: list[FileReport]) -> pd.DataFrame:
    rows = []
    for r in file_reports:
        rows.append(
            {
                "timepoint": r.label,
                "date": r.date,
                "path": str(r.path),
                "loadable": r.loadable,
                "shape": r.shape,
                "spacing_mm": r.spacing,
                "dtype": r.dtype,
                "is_binary": r.is_binary,
                "unique_values": r.unique_values,
                "voxel_count": r.voxel_count,
                "volume_mm3": round(r.volume_mm3, 2),
                "n_components": r.n_components,
                "n_small_components_below_threshold": r.n_small_components,
                "is_empty": r.is_empty,
                "has_nan_or_inf": r.has_nan_or_inf,
                "num_issues": len(r.issues),
                "issues": "; ".join(r.issues),
            }
        )
    return pd.DataFrame(rows)


def format_report_text(
    seg_dir: Path,
    file_reports: list[FileReport],
    pair_reports: list[PairReport],
    connectivity: int,
    min_volume_mm3: float,
    affine_atol: float,
    max_gap_days: float,
) -> str:
    lines = []
    lines.append("TUMOR SEGMENTATION DATA QUALITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Folder: {seg_dir}")
    lines.append(f"Files found: {len(file_reports)}")
    dated = [r for r in file_reports if r.date is not None]
    if dated:
        lines.append(f"Date range: {min(r.date for r in dated).date()} to {max(r.date for r in dated).date()}")
    lines.append(
        f"Settings: connectivity={connectivity}, min_volume_mm3={min_volume_mm3}, "
        f"affine_atol={affine_atol}, max_gap_days={max_gap_days}"
    )
    lines.append("")

    lines.append("PER-FILE DETAIL")
    lines.append("-" * 70)
    for r in file_reports:
        lines.append(f"[{r.label}] {r.path.name}")
        if not r.loadable:
            lines.append(f"    FAILED TO LOAD: {r.error}")
            lines.append("")
            continue

        lines.append(
            f"    shape={r.shape}  spacing_mm={tuple(round(s, 3) for s in r.spacing)}  dtype={r.dtype}"
        )
        lines.append(f"    voxels={r.voxel_count}  volume_mm3={r.volume_mm3:.1f}  components={r.n_components}")
        if r.component_volumes_mm3:
            shown = r.component_volumes_mm3[:5]
            top = ", ".join(f"{v:.1f}" for v in shown)
            more = " ..." if len(r.component_volumes_mm3) > len(shown) else ""
            lines.append(f"    component volumes (mm3, largest first): {top}{more}")

        if r.issues:
            for issue in r.issues:
                lines.append(f"    ISSUE: {issue}")
        else:
            lines.append("    No issues detected")
        lines.append("")

    lines.append("CONSECUTIVE-TIMEPOINT COMPARISON")
    lines.append("-" * 70)
    for p in pair_reports:
        gap = f"{p.days_between:.0f} days" if p.days_between is not None else "unknown"
        status = "OK -- co-registered" if p.is_coregistered else "NOT co-registered"
        lines.append(f"{p.label_a} -> {p.label_b}  (gap: {gap})  [{status}]")
        for issue in p.issues:
            lines.append(f"    ISSUE: {issue}")
        lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 70)
    n_files = len(file_reports)
    n_unloadable = sum(1 for r in file_reports if not r.loadable)
    n_empty = sum(1 for r in file_reports if r.is_empty)
    n_non_binary = sum(1 for r in file_reports if r.loadable and not r.is_binary)
    n_with_issues = sum(1 for r in file_reports if r.issues)
    n_pairs = len(pair_reports)
    n_pairs_coregistered = sum(1 for p in pair_reports if p.is_coregistered)
    n_pairs_not_coregistered = n_pairs - n_pairs_coregistered
    shapes_seen = sorted({r.shape for r in file_reports if r.loadable}, key=str)
    spacings_seen = sorted({r.spacing for r in file_reports if r.loadable}, key=str)

    lines.append(f"Files: {n_files} total, {n_unloadable} failed to load, {n_empty} empty, {n_non_binary} non-binary")
    lines.append(f"Files with at least one issue: {n_with_issues}/{n_files}")
    lines.append(f"Distinct shapes seen: {shapes_seen}")
    lines.append(f"Distinct voxel spacings seen (mm): {[tuple(round(s, 3) for s in sp) for sp in spacings_seen]}")
    lines.append(
        f"Consecutive timepoint pairs: {n_pairs} total, {n_pairs_coregistered} co-registered, "
        f"{n_pairs_not_coregistered} NOT co-registered"
    )
    lines.append("")

    if n_unloadable:
        lines.append(f"BLOCKING: {n_unloadable} file(s) failed to load and must be fixed/removed before running tumor_tracking.py.")

    if n_pairs == 0:
        lines.append("Fewer than 2 usable timepoints -- nothing to track.")
    elif n_pairs_not_coregistered == n_pairs:
        lines.append(
            "All consecutive timepoints are on different grids. tumor_tracking.py will use its "
            "volume-rank fallback for every pair -- fine if each timepoint has at most one tumor "
            "instance (see per-file component counts above), unreliable otherwise. Register scans "
            "to a common grid first for confident multi-lesion identity tracking."
        )
    elif n_pairs_not_coregistered > 0:
        lines.append(
            f"{n_pairs_not_coregistered}/{n_pairs} consecutive pairs are not co-registered and will "
            "use the volume-rank matching fallback in tumor_tracking.py; the rest will use full "
            "spatial matching."
        )
    else:
        lines.append("All consecutive timepoints share a common grid -- full spatial matching will be used throughout.")

    if n_non_binary:
        lines.append(
            f"{n_non_binary} file(s) contain non-binary label values -- confirm this is expected "
            "before treating them as tumor_tracking.py's binary mask input."
        )
    if n_empty:
        lines.append(f"{n_empty} file(s) have no tumor voxels at all -- confirm these are true negatives, not missing segmentations.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    seg_dir: Path,
    filename_pattern: str,
    connectivity: int,
    min_volume_mm3: float,
    affine_atol: float,
    max_gap_days: float,
    output_prefix: Path,
) -> tuple[Path, Path]:
    found = discover_files(seg_dir, filename_pattern)
    if not found:
        raise FileNotFoundError(f"No files matching pattern '{filename_pattern}' found in {seg_dir}")

    # Chronological order; files with an unparsable date sort last and get flagged.
    found_sorted = sorted(found, key=lambda item: (item[1] is None, item[1]))

    file_reports = []
    for label, date, path in found_sorted:
        report = inspect_file(label, date, path, connectivity, min_volume_mm3)
        if date is None:
            report.issues.append(f"Could not parse a date from filename (pattern: {filename_pattern})")
        file_reports.append(report)

    label_counts: dict[str, int] = {}
    for r in file_reports:
        label_counts[r.label] = label_counts.get(r.label, 0) + 1
    for r in file_reports:
        if label_counts[r.label] > 1:
            r.issues.append(f"Duplicate timepoint label '{r.label}' ({label_counts[r.label]} files share it)")

    pair_reports = [
        compare_pair(a, b, affine_atol, max_gap_days) for a, b in zip(file_reports[:-1], file_reports[1:])
    ]

    for r in file_reports:
        if r.issues:
            logger.warning("[%s] %s", r.label, "; ".join(r.issues))
    for p in pair_reports:
        if p.issues:
            logger.warning("[%s -> %s] %s", p.label_a, p.label_b, "; ".join(p.issues))

    csv_path = Path(str(output_prefix) + ".csv")
    txt_path = Path(str(output_prefix) + ".txt")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    build_csv_rows(file_reports).to_csv(csv_path, index=False)
    txt_path.write_text(
        format_report_text(seg_dir, file_reports, pair_reports, connectivity, min_volume_mm3, affine_atol, max_gap_days)
    )

    return csv_path, txt_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-flight data-quality report for a folder of dated tumor segmentation files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seg-dir", required=True, type=Path, help="Folder containing dated segmentation files, e.g. tumor_YYYY-MM.nii.gz or flair_YYYY_MM.nii.gz")
    parser.add_argument(
        "--filename-pattern",
        default=DEFAULT_FILENAME_PATTERN,
        help="Regex with a named group 'date' (YYYY-MM or YYYY_MM); should match tumor_tracking.py's setting",
    )
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26)
    parser.add_argument("--min-volume-mm3", type=float, default=8.0, help="Noise threshold; should match tumor_tracking.py's setting")
    parser.add_argument(
        "--affine-atol",
        type=float,
        default=DEFAULT_AFFINE_ATOL,
        help="Max per-element affine difference to still call two timepoints co-registered (matches tumor_tracking.py)",
    )
    parser.add_argument("--max-gap-days", type=float, default=400.0, help="Flag consecutive visits further apart than this many days")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Base path for outputs; writes <prefix>.csv and <prefix>.txt (default: <seg-dir>/data_quality_report)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    output_prefix = args.output_prefix or (args.seg_dir / "data_quality_report")

    csv_path, txt_path = run(
        seg_dir=args.seg_dir,
        filename_pattern=args.filename_pattern,
        connectivity=args.connectivity,
        min_volume_mm3=args.min_volume_mm3,
        affine_atol=args.affine_atol,
        max_gap_days=args.max_gap_days,
        output_prefix=output_prefix,
    )
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", txt_path)


if __name__ == "__main__":
    main()
