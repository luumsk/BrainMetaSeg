#!/usr/bin/env python3
"""Compare a folder of MRI scans against the BraTS/SRI24 input format.

A BraTS-trained nnU-Net checkpoint (no fine-tuning) expects 4 co-registered,
skull-stripped, 1mm-isotropic channels (T1, T1c, T2, FLAIR) on a shared
240x240x155 grid in SRI24 space. Real clinical/private folders routinely
differ on every one of those axes; this surfaces exactly which ones and by
how much, so you know what `register_to_sri24.py` (and skull-stripping) will
actually need to fix before inference, for ANY input folder -- just point
--input-dir at it.

Reference values (BRATS_REFERENCE_* below) come directly from this repo's
cached SRI24 template (see register_to_sri24.py), not from a spec doc:
* shape/spacing/orientation: read off templates/sri24/spgr.nii.gz
* skull-stripped threshold: spgr.nii.gz (skull-stripped) is ~16% nonzero
  voxels, spgr_unstrip.nii.gz (skull-on) is ~77% -- 40% is the cutoff
  between them. It's a heuristic (a small/necrotic brain or a very tight
  native FOV could confound it), not a guarantee -- read the per-file detail
  if a call looks wrong.

Modality (T1/T1c/T2/FLAIR) is guessed from the filename only (handles both
classic BraTS naming and the 2023+ t1n/t1c/t2w/t2f convention) -- there's no
way to verify contrast from the image data alone, so double check by eye if
"unknown" shows up.

Writes <prefix>.csv (one row per file) + <prefix>.txt (human-readable
report) next to --output-prefix.

Usage
-----
    python utils/check_brats_format.py --input-dir /path/to/new_scans
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd


logger = logging.getLogger("check_brats_format")

# Derived from templates/sri24/spgr.nii.gz -- see module docstring.
BRATS_REFERENCE_SHAPE = (240, 240, 155)
BRATS_REFERENCE_SPACING_MM = (1.0, 1.0, 1.0)
BRATS_REFERENCE_ORIENTATION = "LAS"
DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD = 0.40

REQUIRED_MODALITIES = ["T1", "T1c", "T2", "FLAIR"]
# Checked in this order -- "t1c"/"t2f" are substrings of naive "t1"/"t2"
# checks, so the more specific aliases must be tried first.
MODALITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("FLAIR", ["flair", "t2f"]),
    ("T1c", ["t1c", "t1ce", "t1gd"]),
    ("T2", ["t2w", "t2"]),
    ("T1", ["t1n", "t1w", "t1"]),
]


def discover_scans(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    glob_fn = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(p for p in glob_fn(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir}")
    return paths


def guess_modality(filename: str) -> str:
    name = filename.lower()
    for modality, keywords in MODALITY_KEYWORDS:
        if any(kw in name for kw in keywords):
            return modality
    return "unknown"


# --------------------------------------------------------------------------
# Per-file inspection
# --------------------------------------------------------------------------


@dataclass
class FileReport:
    path: Path
    modality: str
    loadable: bool = True
    error: str = ""
    shape: Optional[tuple] = None
    spacing: Optional[tuple] = None
    orientation: str = ""
    dtype: str = ""
    nonzero_frac: float = 0.0
    likely_skull_stripped: bool = False
    is_isotropic: bool = False
    matches_brats_shape: bool = False
    matches_brats_spacing: bool = False
    matches_brats_orientation: bool = False
    issues: list = field(default_factory=list)

    @property
    def matches_brats_format(self) -> bool:
        return (
            self.loadable
            and self.matches_brats_shape
            and self.matches_brats_spacing
            and self.likely_skull_stripped
        )


def inspect_file(path: Path, skull_stripped_threshold: float) -> FileReport:
    report = FileReport(path=path, modality=guess_modality(path.name))

    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj)
    except Exception as exc:  # noqa: BLE001 -- any load failure is itself the finding
        report.loadable = False
        report.error = f"{type(exc).__name__}: {exc}"
        report.issues.append(f"FILE FAILED TO LOAD: {report.error}")
        return report

    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    shape = tuple(int(v) for v in data.shape[:3])
    report.shape = shape
    report.spacing = spacing
    report.orientation = "".join(nib.aff2axcodes(image.affine))
    report.dtype = str(data.dtype)
    report.nonzero_frac = float((data > 0).mean())
    report.likely_skull_stripped = report.nonzero_frac < skull_stripped_threshold

    spacing_ratio = (max(spacing) / min(spacing)) if min(spacing) > 0 else float("inf")
    report.is_isotropic = spacing_ratio <= 1.05

    report.matches_brats_shape = shape == BRATS_REFERENCE_SHAPE
    report.matches_brats_spacing = all(abs(s - r) <= 0.05 for s, r in zip(spacing, BRATS_REFERENCE_SPACING_MM))
    report.matches_brats_orientation = report.orientation == BRATS_REFERENCE_ORIENTATION

    if report.modality == "unknown":
        report.issues.append("Could not guess modality (T1/T1c/T2/FLAIR) from filename")
    if any(v <= 0 for v in spacing):
        report.issues.append(f"Non-positive voxel spacing in header: {spacing}")
    if not report.is_isotropic:
        report.issues.append(f"Anisotropic voxels: {tuple(round(s, 3) for s in spacing)}mm (ratio {spacing_ratio:.1f}x)")
    if not report.matches_brats_shape:
        report.issues.append(f"Shape {shape} != BraTS/SRI24 reference {BRATS_REFERENCE_SHAPE}")
    if not report.matches_brats_spacing:
        report.issues.append(
            f"Spacing {tuple(round(s, 3) for s in spacing)}mm != BraTS/SRI24 reference {BRATS_REFERENCE_SPACING_MM}mm"
        )
    if not report.likely_skull_stripped:
        report.issues.append(
            f"Likely NOT skull-stripped (heuristic: {report.nonzero_frac:.1%} nonzero voxels, "
            f"threshold {skull_stripped_threshold:.0%})"
        )
    if not report.matches_brats_orientation:
        report.issues.append(f"Orientation {report.orientation} != BraTS/SRI24 reference {BRATS_REFERENCE_ORIENTATION}")

    return report


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def build_csv_rows(file_reports: list[FileReport]) -> pd.DataFrame:
    rows = []
    for r in file_reports:
        rows.append(
            {
                "file": r.path.name,
                "path": str(r.path),
                "modality_guess": r.modality,
                "loadable": r.loadable,
                "shape": r.shape,
                "spacing_mm": r.spacing,
                "orientation": r.orientation,
                "dtype": r.dtype,
                "nonzero_frac": round(r.nonzero_frac, 4),
                "likely_skull_stripped": r.likely_skull_stripped,
                "is_isotropic": r.is_isotropic,
                "matches_brats_shape": r.matches_brats_shape,
                "matches_brats_spacing": r.matches_brats_spacing,
                "matches_brats_orientation": r.matches_brats_orientation,
                "matches_brats_format": r.matches_brats_format,
                "num_issues": len(r.issues),
                "issues": "; ".join(r.issues),
            }
        )
    return pd.DataFrame(rows)


def format_report_text(
    input_dir: Path,
    file_reports: list[FileReport],
    skull_stripped_threshold: float,
) -> str:
    lines = []
    lines.append("BraTS/SRI24 FORMAT COMPLIANCE REPORT")
    lines.append("=" * 70)
    lines.append(f"Folder: {input_dir}")
    lines.append(f"Files found: {len(file_reports)}")
    lines.append(
        f"BraTS/SRI24 reference: shape={BRATS_REFERENCE_SHAPE}, spacing={BRATS_REFERENCE_SPACING_MM}mm, "
        f"orientation={BRATS_REFERENCE_ORIENTATION}, skull-stripped-nonzero-threshold={skull_stripped_threshold:.0%} "
        "(all read off templates/sri24/spgr.nii.gz)"
    )
    lines.append("")

    lines.append("PER-FILE DETAIL")
    lines.append("-" * 70)
    for r in file_reports:
        lines.append(f"[{r.modality}] {r.path.name}")
        if not r.loadable:
            lines.append(f"    FAILED TO LOAD: {r.error}")
            lines.append("")
            continue

        lines.append(
            f"    shape={r.shape}  spacing_mm={tuple(round(s, 3) for s in r.spacing)}  "
            f"orientation={r.orientation}  dtype={r.dtype}"
        )
        lines.append(
            f"    nonzero={r.nonzero_frac:.1%}  skull_stripped={r.likely_skull_stripped}  "
            f"isotropic={r.is_isotropic}  matches_brats_format={r.matches_brats_format}"
        )
        if r.issues:
            for issue in r.issues:
                lines.append(f"    ISSUE: {issue}")
        else:
            lines.append("    Matches BraTS/SRI24 format")
        lines.append("")

    lines.append("MODALITY COVERAGE")
    lines.append("-" * 70)
    groups: dict[str, list[FileReport]] = {}
    for r in file_reports:
        groups.setdefault(r.modality, []).append(r)
    for modality in REQUIRED_MODALITIES:
        count = len(groups.get(modality, []))
        status = "OK" if count else "MISSING"
        lines.append(f"{modality:<8} {count:>3} file(s)  [{status}]")
    unknown_count = len(groups.get("unknown", []))
    if unknown_count:
        lines.append(f"{'unknown':<8} {unknown_count:>3} file(s) -- could not guess modality from filename")
    missing = [m for m in REQUIRED_MODALITIES if not groups.get(m)]
    lines.append("")
    if missing:
        lines.append(
            f"MISSING {len(missing)}/4 required modalities: {missing}. A BraTS-pretrained (non-fine-tuned) "
            "checkpoint needs all 4 channels (T1, T1c, T2, FLAIR) per case -- this blocks correct inference "
            "regardless of registration/skull-stripping."
        )
    else:
        lines.append("All 4 required modalities (T1, T1c, T2, FLAIR) are present.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 70)
    n_files = len(file_reports)
    n_unloadable = sum(1 for r in file_reports if not r.loadable)
    loadable = [r for r in file_reports if r.loadable]
    n_matching = sum(1 for r in loadable if r.matches_brats_format)
    shapes_seen = sorted({r.shape for r in loadable}, key=str)
    spacings_seen = sorted({r.spacing for r in loadable}, key=str)
    orientations_seen = sorted({r.orientation for r in loadable})

    lines.append(f"Files: {n_files} total, {n_unloadable} failed to load")
    lines.append(f"Already match BraTS/SRI24 format (shape+spacing+skull-strip): {n_matching}/{len(loadable)}")
    lines.append(f"Distinct shapes seen: {shapes_seen}")
    lines.append(f"Distinct voxel spacings seen (mm): {[tuple(round(s, 3) for s in sp) for sp in spacings_seen]}")
    lines.append(f"Distinct orientations seen: {orientations_seen}")
    lines.append("")

    if n_unloadable:
        lines.append(f"BLOCKING: {n_unloadable} file(s) failed to load and must be fixed/removed first.")
    if missing:
        lines.append("Fix modality coverage first -- registration/skull-stripping can't compensate for missing channels.")
    if n_matching < len(loadable):
        lines.append(
            f"{len(loadable) - n_matching}/{len(loadable)} file(s) need registration to SRI24 and/or "
            "skull-stripping before they match the reference format -- see utils/register_to_sri24.py."
        )
    if n_matching == len(loadable) and not missing and not n_unloadable:
        lines.append("Folder already matches the BraTS/SRI24 reference format.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    input_dir: Path,
    pattern: str,
    recursive: bool,
    skull_stripped_threshold: float,
    output_prefix: Path,
) -> tuple[Path, Path]:
    paths = discover_scans(input_dir, pattern, recursive)
    logger.info("Found %d file(s) in %s", len(paths), input_dir)

    file_reports = [inspect_file(path, skull_stripped_threshold) for path in paths]

    for r in file_reports:
        if r.issues:
            logger.warning("[%s] %s", r.path.name, "; ".join(r.issues))

    csv_path = Path(str(output_prefix) + ".csv")
    txt_path = Path(str(output_prefix) + ".txt")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    build_csv_rows(file_reports).to_csv(csv_path, index=False)
    txt_path.write_text(format_report_text(input_dir, file_reports, skull_stripped_threshold))

    return csv_path, txt_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a folder of MRI scans against the BraTS/SRI24 input format (shape, spacing, skull-strip, modality coverage).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing MRI scans to check")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --input-dir) for files to check")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search --input-dir recursively",
    )
    parser.add_argument(
        "--skull-stripped-nonzero-threshold",
        type=float,
        default=DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD,
        help="Nonzero-voxel-fraction cutoff below which a scan is called skull-stripped (see module docstring for how this was derived)",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Base path for outputs; writes <prefix>.csv and <prefix>.txt (default: <input-dir>/brats_format_report)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    output_prefix = args.output_prefix or (args.input_dir / "brats_format_report")

    csv_path, txt_path = run(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        skull_stripped_threshold=args.skull_stripped_nonzero_threshold,
        output_prefix=output_prefix,
    )
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", txt_path)


if __name__ == "__main__":
    main()
