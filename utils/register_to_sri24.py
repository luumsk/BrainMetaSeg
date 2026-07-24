#!/usr/bin/env python3
"""Co-register a folder of NIfTI scans onto the SRI24 atlas so they share one grid.

SRI24 (Rohlfing et al., 2010) is a normal-adult-brain MRI atlas distributed
by https://github.com/muschellij2/sri24 (an R data package; this script
talks to the same GitHub repo directly, so no R install is needed). Putting
every scan into that shared space is exactly the "register scans to a common
grid" prerequisite `tumor_tracking.py` calls out for reliable spatial
matching across timepoints/patients.

Available template channels (`--template-channel`), all published by that
repo under `inst/extdata/`:
    spgr          T1-weighted (SPGR), skull-stripped   [default]
    spgr_unstrip  T1-weighted (SPGR), with skull        -- use this if your
                  input scans still have skull on; registering skull-on
                  scans to the skull-stripped `spgr` channel biases the
                  affine fit.
    erly          Early-echo T2 (PD-weighted), skull-stripped
    late          Late-echo T2, skull-stripped
Channels are auto-downloaded from the GitHub repo and cached locally; pass
`--template-path` to use a local file instead (e.g. a different atlas, or
one already installed via the R package).

Registration defaults to `Affine` (rigid + affine, 12 DOF), not deformable
`SyN` -- full nonlinear warping can distort a lesion's shape/volume, which
matters here since this repo's tumor-volume tracking depends on volumes
being trustworthy after registration. Pass `--transform-type SyN` (or
`SyNRA`) if you specifically want deformable alignment.

Requires ANTsPy (`pip install antspyx`, already in requirements.txt).

Expected input
--------------
Any folder of NIfTI scans, optionally nested in per-subject/per-timepoint
subfolders -- the output mirrors whatever structure --input-dir has.

Output
------
    output_dir/...                     registered scans (same relative
                                        layout as --input-dir)
    <output_dir>/../sri24_transforms/  per-scan forward transforms, if
                                        --save-transforms is set (on by
                                        default)
    <output_dir>/../sri24_registration_manifest.csv
                                        one row per scan: status, timing,
                                        shapes/spacings before and after

Usage
-----
    python utils/register_to_sri24.py \\
        --input-dir /data/patient1/raw \\
        --output-dir /data/patient1/sri24

Registering ground-truth labels alongside their scans
-------------------------------------------------------
Pass --labels-dir to also carry each scan's matching label/segmentation file
through the exact same transform computed for that scan -- using a
label-preserving interpolator ("genericLabel") regardless of --interpolator,
so mask values stay crisp instead of blurring like an intensity image would.
A scan with no matching label is still registered normally; only the label
step is skipped for it. --labels-naming-scheme controls how a label
filename is derived from its scan's filename ("identical", or
"braintracking" for this repo's flair_YYYY_MM.nii.gz <-> tumor_YYYY-MM.nii.gz
convention -- see compute_seg_metrics.py's naming schemes, same idea,
opposite direction).

    python utils/register_to_sri24.py \\
        --input-dir nifti_native --output-dir registered \\
        --labels-dir tumor_volume --labels-naming-scheme braintracking
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import ants
import pandas as pd


logger = logging.getLogger("register_to_sri24")

SRI24_REPO_RAW_BASE = "https://raw.githubusercontent.com/muschellij2/sri24/master/inst/extdata"
SRI24_CHANNELS = {
    "spgr": "spgr.nii.gz",
    "spgr_unstrip": "spgr_unstrip.nii.gz",
    "erly": "erly.nii.gz",
    "late": "late.nii.gz",
}
SRI24_CITATION = (
    "T. Rohlfing, N. M. Zahr, E. V. Sullivan, and A. Pfefferbaum, "
    "\"The SRI24 multichannel atlas of normal adult human brain structure,\" "
    "Human Brain Mapping, vol. 31, no. 5, pp. 798-819, 2010. "
    "Distributed (CC-BY-SA) via https://github.com/muschellij2/sri24 -- cite if you publish."
)

DEFAULT_TEMPLATE_CACHE_DIR = Path(__file__).resolve().parent.parent / "templates" / "sri24"


@dataclass
class RegistrationSettings:
    transform_type: str = "Affine"
    n4_correct: bool = True
    interpolator: str = "linear"


@dataclass
class RegistrationResult:
    case: str
    input_path: Path
    output_path: Optional[Path] = None
    status: str = "ok"
    error: str = ""
    elapsed_sec: float = 0.0
    template_channel: str = ""
    transform_type: str = ""
    n4_correct: bool = True
    interpolator: str = ""
    input_shape: tuple = ()
    input_spacing: tuple = ()
    output_shape: tuple = ()
    output_spacing: tuple = ()
    transforms_dir: str = ""
    label_input_path: Optional[Path] = None
    label_output_path: Optional[Path] = None
    label_status: str = ""  # "", "ok", "not_found", "failed"
    label_error: str = ""


# --------------------------------------------------------------------------
# SRI24 template
# --------------------------------------------------------------------------


def ensure_template(channel: str, template_path: Optional[Path], cache_dir: Path) -> Path:
    if template_path is not None:
        if not template_path.exists():
            raise FileNotFoundError(f"--template-path {template_path} does not exist")
        return template_path

    filename = SRI24_CHANNELS[channel]
    cached = cache_dir / filename
    if cached.exists():
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"{SRI24_REPO_RAW_BASE}/{filename}"
    logger.info("Downloading SRI24 '%s' channel from %s", channel, url)
    logger.info(SRI24_CITATION)

    import ssl
    import urllib.request

    try:
        # macOS python.org framework builds ship without a populated cert
        # bundle until "Install Certificates.command" is run; fall back to
        # certifi's bundle (usually already present transitively) instead of
        # failing the download outright.
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_context = ssl.create_default_context()

    tmp_path = cached.with_suffix(cached.suffix + ".part")
    with urllib.request.urlopen(url, context=ssl_context) as response, open(tmp_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.rename(cached)
    return cached


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover_scans(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    glob_fn = input_dir.rglob if recursive else input_dir.glob
    paths = sorted(p for p in glob_fn(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir}")
    return paths


def strip_nifti_suffix(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


# --------------------------------------------------------------------------
# Scan -> label filename mapping (inverse of compute_seg_metrics.py's
# gt_to_seg naming schemes, since here the scan drives discovery)
# --------------------------------------------------------------------------


def identical_scan_to_label_filename(scan_filename: str) -> str:
    return scan_filename


def braintracking_scan_to_label_filename(scan_filename: str) -> str:
    """Map scan "flair_2016_11.nii.gz" to its matching label "tumor_2016-11.nii.gz"."""
    stem = strip_nifti_suffix(scan_filename)
    suffix = scan_filename[len(stem):]

    scan_prefix = "flair_"
    if not stem.startswith(scan_prefix):
        raise ValueError(f"Expected scan filename to start with '{scan_prefix}', got {scan_filename!r}")

    date_part = stem[len(scan_prefix):].replace("_", "-")
    return f"tumor_{date_part}{suffix}"


LABEL_NAMING_SCHEMES = {
    "identical": identical_scan_to_label_filename,
    "braintracking": braintracking_scan_to_label_filename,
}


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def save_transforms(fwd_transforms: list[str], dest_dir: Path) -> Path:
    """Copy ANTs' temp-dir transform files into a persistent, ordered folder.

    Order matters for reuse (see module docstring's Extension point), so the
    original --fwdtransforms order is preserved via fwdtransforms.json rather
    than relying on filename sort.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, src in enumerate(fwd_transforms):
        src_path = Path(src)
        dest_path = dest_dir / f"{i:02d}_{src_path.name}"
        shutil.copyfile(src_path, dest_path)
        saved.append(dest_path.name)
    (dest_dir / "fwdtransforms.json").write_text(json.dumps(saved, indent=2))
    return dest_dir


# --------------------------------------------------------------------------
# Per-scan registration
# --------------------------------------------------------------------------


def register_one(
    scan_path: Path,
    case: str,
    fixed_img: "ants.ANTsImage",
    template_channel: str,
    output_path: Path,
    transforms_dir: Optional[Path],
    settings: RegistrationSettings,
    label_path: Optional[Path] = None,
    label_output_path: Optional[Path] = None,
) -> RegistrationResult:
    result = RegistrationResult(
        case=case,
        input_path=scan_path,
        template_channel=template_channel,
        transform_type=settings.transform_type,
        n4_correct=settings.n4_correct,
        interpolator=settings.interpolator,
        label_input_path=label_path,
        label_status="not_found" if label_path is None else "",
    )
    t0 = time.time()
    try:
        moving_img = ants.image_read(str(scan_path))
        result.input_shape = tuple(moving_img.shape)
        result.input_spacing = tuple(moving_img.spacing)

        if settings.n4_correct:
            moving_img = ants.n4_bias_field_correction(moving_img)

        reg = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=settings.transform_type)
        warped = ants.apply_transforms(
            fixed=fixed_img,
            moving=moving_img,
            transformlist=reg["fwdtransforms"],
            interpolator=settings.interpolator,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(warped, str(output_path))
        result.output_path = output_path
        result.output_shape = tuple(warped.shape)
        result.output_spacing = tuple(warped.spacing)

        if transforms_dir is not None:
            saved_dir = save_transforms(reg["fwdtransforms"], transforms_dir / case)
            result.transforms_dir = str(saved_dir)

        if label_path is not None:
            try:
                label_img = ants.image_read(str(label_path))
                # Always label-preserving here, independent of --interpolator
                # (which governs the intensity image above): "genericLabel"
                # is a majority-vote resample that keeps discrete label
                # values crisp instead of blurring them like linear/bSpline
                # would.
                warped_label = ants.apply_transforms(
                    fixed=fixed_img,
                    moving=label_img,
                    transformlist=reg["fwdtransforms"],
                    interpolator="genericLabel",
                )
                label_output_path.parent.mkdir(parents=True, exist_ok=True)
                ants.image_write(warped_label, str(label_output_path))
                result.label_output_path = label_output_path
                result.label_status = "ok"
            except Exception as label_exc:
                result.label_status = "failed"
                result.label_error = str(label_exc)
                logger.error("%s: label registration failed -- %s", case, label_exc)

    except Exception as exc:  # one bad scan shouldn't kill the whole batch
        result.status = "failed"
        result.error = str(exc)
        logger.error("%s: registration failed -- %s", case, exc)
    finally:
        result.elapsed_sec = time.time() - t0

    return result


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    recursive: bool,
    template_channel: str,
    template_path: Optional[Path],
    template_cache_dir: Path,
    settings: RegistrationSettings,
    output_suffix: str,
    transforms_dir: Optional[Path],
    manifest_csv: Path,
    overwrite: bool,
    labels_dir: Optional[Path] = None,
    labels_output_dir: Optional[Path] = None,
    labels_naming_scheme: str = "identical",
) -> pd.DataFrame:
    fixed_path = ensure_template(template_channel, template_path, template_cache_dir)
    logger.info("Using SRI24 '%s' template: %s", template_channel, fixed_path)
    fixed_img = ants.image_read(str(fixed_path))

    scans = discover_scans(input_dir, pattern, recursive)
    logger.info("Found %d scan(s) in %s", len(scans), input_dir)
    if labels_dir is not None:
        logger.info("Also registering matching labels from %s (naming scheme: %s)", labels_dir, labels_naming_scheme)
        map_scan_to_label = LABEL_NAMING_SCHEMES[labels_naming_scheme]

    results = []
    for i, scan_path in enumerate(scans, start=1):
        rel_path = scan_path.relative_to(input_dir)
        case = str((rel_path.parent / strip_nifti_suffix(scan_path.name)).as_posix())
        out_name = f"{strip_nifti_suffix(scan_path.name)}{output_suffix}.nii.gz"
        output_path = output_dir / rel_path.parent / out_name

        label_path = None
        label_output_path = None
        if labels_dir is not None:
            try:
                label_filename = map_scan_to_label(scan_path.name)
                candidate = labels_dir / label_filename
                if candidate.is_file():
                    label_path = candidate
                    label_output_path = labels_output_dir / label_filename
                else:
                    logger.info("%s: no matching label at %s -- registering scan only", case, candidate)
            except ValueError as exc:
                logger.warning("%s: could not derive label filename -- %s", case, exc)

        if output_path.exists() and not overwrite:
            logger.info("[%d/%d] %s: output already exists, skipping (--overwrite to redo)", i, len(scans), case)
            results.append(
                RegistrationResult(
                    case=case,
                    input_path=scan_path,
                    output_path=output_path,
                    status="skipped",
                    template_channel=template_channel,
                    transform_type=settings.transform_type,
                    n4_correct=settings.n4_correct,
                    interpolator=settings.interpolator,
                    label_input_path=label_path,
                    label_status="skipped" if label_path is not None else "",
                )
            )
            continue

        logger.info("[%d/%d] %s: registering", i, len(scans), case)
        result = register_one(
            scan_path, case, fixed_img, template_channel, output_path, transforms_dir, settings,
            label_path=label_path, label_output_path=label_output_path,
        )
        results.append(result)

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(manifest_csv, index=False)

    n_ok = int((df["status"] == "ok").sum())
    n_failed = int((df["status"] == "failed").sum())
    n_skipped = int((df["status"] == "skipped").sum())
    msg = f"Done: {n_ok} ok, {n_failed} failed, {n_skipped} skipped."
    if labels_dir is not None:
        n_label_ok = int((df["label_status"] == "ok").sum())
        n_label_failed = int((df["label_status"] == "failed").sum())
        n_label_missing = int((df["label_status"] == "not_found").sum())
        msg += f" Labels: {n_label_ok} ok, {n_label_failed} failed, {n_label_missing} had no match."
    logger.info("%s Manifest: %s", msg, manifest_csv)

    return df


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Co-register a folder of NIfTI scans onto the SRI24 atlas (https://github.com/muschellij2/sri24) so they share one voxel grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None, help="Folder containing scans to register (required unless --download-only)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where registered scans are written, mirrors --input-dir's subfolder structure (required unless --download-only)",
    )
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --input-dir) for scans to register")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search --input-dir recursively (matches subfolders per subject/timepoint)",
    )
    parser.add_argument("--output-suffix", default="", help="Suffix inserted before .nii.gz in output filenames, e.g. '_sri24'")

    parser.add_argument(
        "--template-channel",
        choices=sorted(SRI24_CHANNELS),
        default="spgr",
        help="SRI24 channel to register to: spgr=T1 skull-stripped, spgr_unstrip=T1 with skull, erly/late=T2 skull-stripped",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=None,
        help="Use this NIfTI file as the fixed/reference image instead of auto-downloading --template-channel",
    )
    parser.add_argument(
        "--template-cache-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_CACHE_DIR,
        help="Where downloaded SRI24 template channels are cached",
    )

    parser.add_argument(
        "--transform-type",
        choices=["Rigid", "Affine", "SyN", "SyNRA"],
        default="Affine",
        help="ANTs transform type. Affine (rigid+affine, 12 DOF) aligns to atlas space without deforming lesion shape/volume, unlike deformable SyN",
    )
    parser.add_argument(
        "--n4-correct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply N4 bias field correction to each scan before registration",
    )
    parser.add_argument(
        "--interpolator",
        choices=["linear", "bSpline", "nearestNeighbor"],
        default="linear",
        help="Interpolation used when resampling the registered scan into template space",
    )

    parser.add_argument(
        "--save-transforms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist each case's forward transforms for later reuse (e.g. warping a companion segmentation mask)",
    )
    parser.add_argument(
        "--transforms-dir",
        type=Path,
        default=None,
        help="Where to save transforms if --save-transforms is set (defaults to a 'sri24_transforms' folder next to --output-dir)",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Path to write the per-scan registration manifest CSV (defaults next to --output-dir)",
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Folder of ground-truth label/segmentation files to carry through the same per-scan transform "
        "(always uses a label-preserving interpolator, independent of --interpolator)",
    )
    parser.add_argument(
        "--labels-output-dir",
        type=Path,
        default=None,
        help="Where registered labels are written if --labels-dir is set (default: 'registered_labels' next to --output-dir)",
    )
    parser.add_argument(
        "--labels-naming-scheme",
        choices=sorted(LABEL_NAMING_SCHEMES),
        default="identical",
        help="How to derive a label filename from its scan's filename: 'identical' (same name), or "
        "'braintracking' (scan 'flair_2016_11.nii.gz' -> label 'tumor_2016-11.nii.gz')",
    )

    parser.add_argument("--overwrite", action="store_true", help="Re-register and overwrite scans that already have an output file")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Just fetch/cache --template-channel (or verify --template-path) and exit -- no registration, --input-dir/--output-dir not required",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.download_only:
        fixed_path = ensure_template(args.template_channel, args.template_path, args.template_cache_dir)
        logger.info("SRI24 '%s' template ready at: %s", args.template_channel, fixed_path)
        return

    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required unless --download-only is set")

    settings = RegistrationSettings(
        transform_type=args.transform_type,
        n4_correct=args.n4_correct,
        interpolator=args.interpolator,
    )

    transforms_dir = args.transforms_dir or (args.output_dir.parent / "sri24_transforms")
    manifest_csv = args.manifest_csv or (args.output_dir.parent / "sri24_registration_manifest.csv")
    labels_output_dir = args.labels_output_dir or (args.output_dir.parent / "registered_labels")

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        template_channel=args.template_channel,
        template_path=args.template_path,
        template_cache_dir=args.template_cache_dir,
        settings=settings,
        output_suffix=args.output_suffix,
        transforms_dir=transforms_dir if args.save_transforms else None,
        manifest_csv=manifest_csv,
        overwrite=args.overwrite,
        labels_dir=args.labels_dir,
        labels_output_dir=labels_output_dir if args.labels_dir else None,
        labels_naming_scheme=args.labels_naming_scheme,
    )


if __name__ == "__main__":
    main()
