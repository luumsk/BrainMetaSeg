#!/usr/bin/env python3
"""BraTS-matched multi-modality preprocessing: one input folder in, one
nnU-Net-ready output folder out.

Each timepoint's 4 modalities (T1, T1c, T2, FLAIR) are processed as one
group, not as independent files:
    1. N4 bias field correction (--n4-correct), per modality.
    2. Intra-subject co-registration: every modality is rigid-registered to
       ONE reference modality per timepoint, chosen by BraTS priority
       T1 > T2 > T1c > FLAIR (REFERENCE_PRIORITY below).
    3. Atlas registration (--register): the reference (carrying the other
       3 modalities via a single composed transform each -- one resample,
       not two) is rigid-registered to SRI24. This is what makes all 4
       modalities of a timepoint land on the exact same grid.
    4. Skull-stripping (HD-BET) runs ONCE per timepoint, on the atlas-space
       reference image only. The resulting mask is reused (not
       recomputed) for the other 3 modalities.
    5. Z-score intensity normalization (--normalize), per modality, within
       the shared brain mask. OFF by default -- see PreprocessSettings.do_normalize.
Every final output (image, label, mask) is reoriented to BRATS_REFERENCE_ORIENTATION
right before it's saved, regardless of which SRI24 channel it was registered
to -- see the comment on that constant for why this matters.
A single ground-truth label (--labels-dir) rides along through the SAME
composed transform as its host modality (FLAIR, via the "braintracking"
naming scheme) and gets masked with the same shared brain mask.

Step 4 needs its own venv (nnunetv2 version conflict with this repo's own
fork -- see setup_hdbet_venv.sh) so this script shells out to it as a
subprocess for that one step; everything else runs in-process here.

Output is written directly in nnU-Net's raw-dataset convention:
    output_dir/
        imagesTr/<case_id>_<CCCC>.nii.gz   CCCC = NNUNET_CHANNEL_INDEX below
        labelsTr/<case_id>.nii.gz          if a label was found for that case
        dataset.json                       nnU-Net dataset descriptor, written
                                            if --labels-dir was given
        masks/<case_id>_mask.nii.gz        shared brain mask (QC/audit only)
        transforms/<case_id>/...           saved transforms, if
                                            --register and --save-transforms
        preprocess_report.csv/.txt         one row per output file
        preprocess_summary.txt             aggregate-only summary
        verification_report.csv/.txt       independent post-hoc check of the
                                            saved outputs, if --verify

Usage
-----
    python utils/preprocess_brats.py \\
        --input-dir nifti_native_4modalities --output-dir brats_preprocessed \\
        --labels-dir tumor_volume --labels-naming-scheme braintracking
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import ants
import nibabel as nib
import numpy as np
import pandas as pd

from apply_brain_mask import apply_mask_to_label, mask_image
from check_brats_format import (
    BRATS_REFERENCE_SHAPE,
    BRATS_REFERENCE_SPACING_MM,
    DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD,
    guess_modality,
)
from register_to_sri24 import (
    DEFAULT_TEMPLATE_CACHE_DIR,
    LABEL_NAMING_SCHEMES,
    SRI24_CHANNELS,
    discover_scans,
    ensure_template,
    save_transforms,
    strip_nifti_suffix,
)


logger = logging.getLogger("preprocess_brats")

INTERP_CODES = {"linear": 0, "nearestNeighbor": 1, "bSpline": 4}  # ants.resample_image's interp_type
MASK_SUFFIX = "_bet.nii.gz"  # HD-BET's own convention for --save_bet_mask output

# Official BraTS/BraTS-MET orientation (confirmed against a real BraTS-MET
# case's header) -- distinct from either locally-cached SRI24 atlas
# channel's own header (spgr=LAS, spgr_unstrip=RAS), despite all three
# sharing the same physical atlas space. nnU-Net's default SimpleITKIO
# reader doesn't reorient to a canonical direction, so without this
# explicit reorient step, final output silently inherits whatever
# convention the registration target's header happened to use.
BRATS_REFERENCE_ORIENTATION = ("L", "P", "S")


def reorient_to_target(img: "nib.Nifti1Image", target_axcodes: tuple = BRATS_REFERENCE_ORIENTATION) -> "nib.Nifti1Image":
    current_ornt = nib.io_orientation(img.affine)
    target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    return img.as_reoriented(transform)


def reorient_file_to_target(path: Path, target_axcodes: tuple = BRATS_REFERENCE_ORIENTATION) -> None:
    """Reorient an already-saved NIfTI file in place (used for the mask/label
    outputs, which -- unlike the main image -- aren't already passing through
    stage3_finalize_image's reorient_to_target call)."""
    img = nib.load(str(path))
    reoriented = reorient_to_target(img, target_axcodes)
    nib.save(reoriented, str(path))

# BraTS intra-subject co-registration reference priority (T1w > T2w > T1Gd >
# FLAIR) -- the first of these present in a timepoint is the modality every
# other modality of that timepoint is registered to.
REFERENCE_PRIORITY = ["T1", "T2", "T1c", "FLAIR"]
# nnU-Net BraTS channel convention.
NNUNET_CHANNEL_INDEX = {"T1c": "0000", "T1": "0001", "FLAIR": "0002", "T2": "0003"}
# Official BraTS naming for dataset.json's channel_names values (distinct from
# our internal canonical modality names above).
NNUNET_CHANNEL_NAMES = {"T1c": "T1C", "T1": "T1N", "FLAIR": "T2F", "T2": "T2W"}
# This repo's ground-truth labels are a single binary tumor mask (confirmed
# via tumor_volume/*.nii.gz: unique values are always [0, 1]) -- not BraTS's
# multi-class WT/TC/ET region scheme, which needs 3 distinct label values
# this dataset doesn't have.
NNUNET_LABELS = {"background": 0, "tumor": 1}
# Label naming (braintracking scheme) is hardcoded to the FLAIR filename --
# always look up a timepoint's label via its FLAIR sibling, regardless of
# which modality is the registration reference.
LABEL_HOST_MODALITY = "FLAIR"


def case_id_for_timepoint(timepoint_key: str) -> str:
    return f"case_{timepoint_key}"


@dataclass
class PreprocessSettings:
    n4_correct: bool = True
    do_register: bool = True
    transform_type: str = "Rigid"
    do_resample: bool = True  # only takes effect when do_register is False
    interpolator: str = "linear"
    do_skull_strip: bool = True
    hdbet_device: str = ""  # "" -> auto-detect (cuda if available, else cpu)
    hdbet_disable_tta: bool = False
    # Default False: imagesTr is nnU-Net's *raw* dataset format, which must
    # carry raw intensities -- nnU-Net computes and applies its own z-score
    # normalization from the plans it derives at training time, and reuses
    # those exact parameters at inference. Pre-normalizing here and letting
    # nnU-Net normalize again on top double-applies the transform against a
    # distribution shape (already ~mean 0/std 1) that no longer resembles
    # the raw intensities nnU-Net's normalization was fit against.
    do_normalize: bool = False


@dataclass
class PreprocessResult:
    case_id: str
    modality: str  # canonical modality name, or "label"
    input_path: Path
    output_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    status: str = "ok"
    error: str = ""
    elapsed_sec: float = 0.0
    steps_applied: str = ""
    input_shape: tuple = ()
    input_spacing: tuple = ()
    output_shape: tuple = ()
    output_spacing: tuple = ()
    nonzero_frac_before: float = 0.0
    nonzero_frac_after: float = 0.0
    brain_mean_raw: Optional[float] = None
    brain_std_raw: Optional[float] = None
    label_input_path: Optional[Path] = None
    label_output_path: Optional[Path] = None
    label_status: str = ""  # "", "ok", "not_found", "failed"
    label_error: str = ""

    @property
    def case(self) -> str:
        return f"{self.case_id}/{self.modality}"


# --------------------------------------------------------------------------
# Timepoint grouping: discover scans, classify modality, group by timepoint
# --------------------------------------------------------------------------


@dataclass
class TimepointGroup:
    key: str
    modalities: dict[str, Path] = field(default_factory=dict)  # canonical modality -> path
    label_path: Optional[Path] = None


def discover_timepoint_groups(
    input_dir: Path,
    pattern: str,
    recursive: bool,
    labels_dir: Optional[Path],
    labels_naming_scheme: str,
) -> list[TimepointGroup]:
    scans = discover_scans(input_dir, pattern, recursive)
    map_scan_to_label = LABEL_NAMING_SCHEMES[labels_naming_scheme] if labels_dir is not None else None

    groups: dict[str, dict[str, Path]] = {}
    for scan_path in scans:
        stem = strip_nifti_suffix(scan_path.name)
        if "_" not in stem:
            logger.warning("%s: filename has no '<modality>_<timepoint>' structure -- skipping", scan_path.name)
            continue
        prefix_token, timepoint_key = stem.split("_", 1)
        modality = guess_modality(prefix_token)
        if modality == "unknown":
            logger.warning("%s: could not guess modality from prefix '%s' -- skipping", scan_path.name, prefix_token)
            continue
        groups.setdefault(timepoint_key, {})[modality] = scan_path

    timepoint_groups = []
    for timepoint_key in sorted(groups):
        modalities = groups[timepoint_key]
        if not any(m in modalities for m in REFERENCE_PRIORITY):
            logger.warning(
                "%s: none of the reference-eligible modalities %s present -- skipping timepoint",
                timepoint_key, REFERENCE_PRIORITY,
            )
            continue

        label_path = None
        if map_scan_to_label is not None and LABEL_HOST_MODALITY in modalities:
            try:
                candidate = labels_dir / map_scan_to_label(modalities[LABEL_HOST_MODALITY].name)
                if candidate.is_file():
                    label_path = candidate
                else:
                    logger.info("%s: no matching label at %s -- processing scans only", timepoint_key, candidate)
            except ValueError as exc:
                logger.warning("%s: could not derive label filename -- %s", timepoint_key, exc)

        timepoint_groups.append(TimepointGroup(key=timepoint_key, modalities=modalities, label_path=label_path))

    return timepoint_groups


def pick_reference_modality(modalities: dict[str, Path]) -> str:
    for m in REFERENCE_PRIORITY:
        if m in modalities:
            return m
    raise ValueError(f"no reference-eligible modality present among {sorted(modalities)}")


# --------------------------------------------------------------------------
# Stage 1 (per timepoint, main venv/ants): N4 + intra-subject coregister +
# atlas register -- reference modality first, then each dependent modality
# via ONE composed transform (dependent->reference->atlas), avoiding a
# double resample.
# --------------------------------------------------------------------------


def resample_isotropic(img: "ants.ANTsImage", interpolator: str) -> "ants.ANTsImage":
    return ants.resample_image(img, (1.0, 1.0, 1.0), use_voxels=False, interp_type=INTERP_CODES[interpolator])


def load_and_n4(path: Path, do_n4: bool) -> "ants.ANTsImage":
    img = ants.image_read(str(path))
    if do_n4:
        img = ants.n4_bias_field_correction(img)
    return img


def stage1_register_reference(
    reference_path: Path, fixed_img: Optional["ants.ANTsImage"], settings: PreprocessSettings
) -> dict:
    """Register the timepoint's reference modality to the atlas (or resample/
    passthrough if --no-register). Returns the warped image, the forward
    transform list to reuse for dependents (empty if none), the reference's
    own N4'd native-space image (the fixed target for dependents' intra-
    subject registration), and the "group fixed image" every modality in
    this timepoint gets resampled onto."""
    n4_img = load_and_n4(reference_path, settings.n4_correct)
    info = {
        "input_shape": tuple(ants.image_read(str(reference_path)).shape),
        "steps": ["n4"] if settings.n4_correct else [],
    }

    if settings.do_register:
        reg = ants.registration(fixed=fixed_img, moving=n4_img, type_of_transform=settings.transform_type)
        warped = ants.apply_transforms(
            fixed=fixed_img, moving=n4_img, transformlist=reg["fwdtransforms"], interpolator=settings.interpolator
        )
        fwd_transforms = reg["fwdtransforms"]
        group_fixed_img = fixed_img
        info["steps"].append("register_atlas")
    elif settings.do_resample:
        warped = resample_isotropic(n4_img, settings.interpolator)
        fwd_transforms = []
        group_fixed_img = warped
        info["steps"].append("resample")
    else:
        warped = n4_img
        fwd_transforms = []
        group_fixed_img = warped

    info["warped"] = warped
    info["fwd_transforms"] = fwd_transforms
    info["reference_n4_img"] = n4_img
    info["group_fixed_img"] = group_fixed_img
    return info


def stage1_register_dependent(
    dependent_path: Path,
    reference_n4_img: "ants.ANTsImage",
    group_fixed_img: "ants.ANTsImage",
    reference_fwd_transforms: list,
    settings: PreprocessSettings,
) -> dict:
    """Co-register a dependent modality to the (native-space) reference, then
    warp it directly into the group's shared grid in ONE resample by
    composing [reference_fwd_transforms, dependent->reference transforms].
    ANTs/ITK's apply_transforms applies the LAST-listed transform first, so
    the reference's own transform (dependent-space -> ... -> atlas/group
    space) must be listed first and the intra-subject transform last --
    empirically verified against a two-step sequential resample."""
    n4_img = load_and_n4(dependent_path, settings.n4_correct)
    info = {
        "input_shape": tuple(ants.image_read(str(dependent_path)).shape),
        "steps": (["n4"] if settings.n4_correct else []) + ["coregister_to_reference"],
    }

    reg_dep = ants.registration(fixed=reference_n4_img, moving=n4_img, type_of_transform=settings.transform_type)
    transformlist = list(reference_fwd_transforms) + list(reg_dep["fwdtransforms"])
    warped = ants.apply_transforms(
        fixed=group_fixed_img, moving=n4_img, transformlist=transformlist, interpolator=settings.interpolator
    )
    if reference_fwd_transforms:
        info["steps"].append("register_atlas")

    info["warped"] = warped
    info["transformlist"] = transformlist
    return info


# --------------------------------------------------------------------------
# Stage 2 (batch subprocess, isolated hdbet_venv): skull-strip the
# reference-modality atlas-space image only, once per timepoint.
# --------------------------------------------------------------------------


def resolve_hdbet_device(device: str, hdbet_venv_dir: Path) -> str:
    if device:
        return device
    python_bin = hdbet_venv_dir / "bin" / "python3"
    result = subprocess.run(
        [str(python_bin), "-c", "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = result.stdout.strip()
    logger.info("Auto-detected HD-BET device: %s", resolved)
    return resolved


def run_hdbet(input_dir: Path, output_dir: Path, hdbet_bin: Path, device: str, disable_tta: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(hdbet_bin), "-i", str(input_dir), "-o", str(output_dir), "-device", device, "--save_bet_mask", "--verbose"]
    if disable_tta:
        cmd.append("--disable_tta")
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------
# Stage 3 (per output file, main venv/numpy): normalize image, mask label
# --------------------------------------------------------------------------


def zscore_normalize(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    brain_voxels = data[mask > 0]
    mean = float(brain_voxels.mean()) if brain_voxels.size else 0.0
    std = float(brain_voxels.std()) if brain_voxels.size else 1.0
    if std == 0:
        std = 1.0
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask > 0] = (data[mask > 0].astype(np.float32) - mean) / std
    return normalized, mean, std


def stage3_finalize_image(
    stage1_image_path: Path,
    skull_stripped_path: Optional[Path],
    mask_path: Optional[Path],
    out_image_path: Path,
    settings: PreprocessSettings,
) -> dict:
    info = {"mask_path": mask_path}

    if skull_stripped_path is not None and skull_stripped_path.is_file():
        img = nib.load(str(skull_stripped_path))
        info["steps"] = ["skull_strip"]
    else:
        img = nib.load(str(stage1_image_path))
        info["steps"] = []

    data = np.asarray(img.dataobj)
    info["nonzero_frac_after"] = float((data > 0).mean())

    if settings.do_normalize:
        if mask_path is not None and mask_path.is_file():
            mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0
        else:
            logger.warning(
                "%s: normalizing without a skull-strip brain mask -- falling back to a nonzero-voxel "
                "approximation (less accurate; run with --skull-strip for a real brain mask)",
                out_image_path.name,
            )
            mask = data > 0
        normalized, mean, std = zscore_normalize(data, mask)
        out_img = nib.Nifti1Image(normalized, img.affine, img.header)
        # mean/std are the PRE-normalization brain-intensity stats -- i.e.
        # the parameters normalization used ((x - mean) / std), not the
        # (trivially ~0/~1) post-normalization result. Useful as a QA signal
        # for spotting scans with unusual raw intensity scale across a
        # cohort; verified separately during testing that the actual output
        # voxels land at mean~0/std~1 within the brain mask as expected.
        info["brain_mean_raw"], info["brain_std_raw"] = mean, std
        info["steps"].append("normalize")
    else:
        out_img = img
        info["brain_mean_raw"], info["brain_std_raw"] = None, None

    out_img = reorient_to_target(out_img)

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_image_path))
    return info


# --------------------------------------------------------------------------
# Verification (post-hoc, re-reads the SAVED files independently rather than
# re-trusting values already computed during processing -- catches bugs in
# the processing/reporting code itself, not just failures during
# processing. This is exactly how the mislabeled brain_mean/brain_std report
# bug got caught: the pipeline's own report claimed success, but re-checking
# the actual saved file's statistics showed the field was mislabeled).
# --------------------------------------------------------------------------


@dataclass
class VerificationResult:
    case: str
    image_path: Optional[Path] = None
    status: str = "ok"
    error: str = ""
    shape_ok: Optional[bool] = None
    spacing_ok: Optional[bool] = None
    orientation_ok: Optional[bool] = None
    skull_strip_ok: Optional[bool] = None
    normalize_ok: Optional[bool] = None
    label_grid_match_ok: Optional[bool] = None
    label_discrete_ok: Optional[bool] = None
    issues: str = ""

    @property
    def all_ok(self) -> bool:
        checks = [
            self.shape_ok, self.spacing_ok, self.orientation_ok, self.skull_strip_ok,
            self.normalize_ok, self.label_grid_match_ok, self.label_discrete_ok,
        ]
        return self.status == "ok" and all(c is not False for c in checks)


def verify_case(
    result: PreprocessResult,
    settings: PreprocessSettings,
    reference_orientation: Optional[str],
    normalize_tol: float = 0.05,
) -> VerificationResult:
    v = VerificationResult(case=result.case, image_path=result.output_path)
    issues = []
    try:
        img = nib.load(str(result.output_path))
        data = np.asarray(img.dataobj)
        shape = tuple(int(s) for s in data.shape[:3])
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

        if settings.do_register:
            v.shape_ok = shape == BRATS_REFERENCE_SHAPE
            v.spacing_ok = all(abs(s - r) <= 0.05 for s, r in zip(spacing, BRATS_REFERENCE_SPACING_MM))
            orientation = "".join(nib.aff2axcodes(img.affine))
            v.orientation_ok = reference_orientation is None or orientation == reference_orientation
            if not v.shape_ok:
                issues.append(f"shape {shape} != {BRATS_REFERENCE_SHAPE}")
            if not v.spacing_ok:
                issues.append(f"spacing {spacing} != {BRATS_REFERENCE_SPACING_MM}")
            if not v.orientation_ok:
                issues.append(f"orientation {orientation} != {reference_orientation}")

        if settings.do_skull_strip:
            nonzero_frac = float((data != 0).mean())
            v.skull_strip_ok = nonzero_frac < DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD
            if not v.skull_strip_ok:
                issues.append(f"nonzero fraction {nonzero_frac:.1%} >= skull-stripped threshold {DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD:.0%}")

        if settings.do_normalize:
            brain = data[data != 0]
            if brain.size:
                mean, std = float(brain.mean()), float(brain.std())
                v.normalize_ok = abs(mean) <= normalize_tol and abs(std - 1.0) <= normalize_tol
                if not v.normalize_ok:
                    issues.append(f"brain mean/std {mean:.4f}/{std:.4f} not within {normalize_tol} of 0/1")
            else:
                v.normalize_ok = False
                issues.append("no nonzero (brain) voxels to check normalization against")

        if result.label_output_path is not None and result.label_output_path.is_file():
            lbl_img = nib.load(str(result.label_output_path))
            lbl_data = np.asarray(lbl_img.dataobj)
            v.label_grid_match_ok = lbl_data.shape == data.shape and np.allclose(lbl_img.affine, img.affine, atol=1e-2)
            if not v.label_grid_match_ok:
                issues.append("label does not share the image's grid (shape/affine mismatch)")

            uniques = np.unique(lbl_data)
            v.label_discrete_ok = uniques.size <= 20 and np.allclose(uniques, np.round(uniques))
            if not v.label_discrete_ok:
                issues.append(f"label has {uniques.size} unique value(s), not discrete -- possible interpolation corruption")

    except Exception as exc:
        v.status = "failed"
        v.error = str(exc)
        issues.append(f"verification crashed: {exc}")

    v.issues = "; ".join(issues)
    return v


def run_verification(
    results: list[PreprocessResult], settings: PreprocessSettings, reference_orientation: Optional[str]
) -> list[VerificationResult]:
    verifications = []
    for r in results:
        if r.status != "ok" or r.output_path is None:
            continue
        v = verify_case(r, settings, reference_orientation)
        if not v.all_ok:
            logger.warning("[VERIFY] %s: %s", v.case, v.issues)
        verifications.append(v)
    return verifications


def write_dataset_json(output_dir: Path, labels_out_dir: Path) -> Path:
    """Write nnU-Net v2's dataset.json descriptor. numTraining is derived by
    counting labelsTr/*.nii.gz on disk (not in-memory results) so it stays
    correct across reruns where some/all cases were skipped as already done."""
    # dataset.json's channel_names keys are plain integers ("0", "1", ...),
    # distinct from the zero-padded "0000" suffix used in filenames.
    channel_names = {str(int(index)): NNUNET_CHANNEL_NAMES[modality] for modality, index in NNUNET_CHANNEL_INDEX.items()}
    num_training = len(list(labels_out_dir.glob("*.nii.gz"))) if labels_out_dir.is_dir() else 0
    dataset_json = {
        "channel_names": channel_names,
        "labels": NNUNET_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    dataset_json_path = output_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    return dataset_json_path


def format_verification_text(verifications: list[VerificationResult]) -> str:
    lines = ["BraTS PREPROCESSING VERIFICATION", "=" * 70, ""]
    n_pass = sum(v.all_ok for v in verifications)
    lines.append(f"{n_pass}/{len(verifications)} case(s) passed all applicable checks")
    lines.append("(re-derived independently from the saved output files, not from the processing report)")
    lines.append("")
    for v in verifications:
        status = "PASS" if v.all_ok else "FAIL"
        lines.append(f"[{status}] {v.case}")
        if v.issues:
            lines.append(f"    {v.issues}")
    return "\n".join(lines)


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
    settings: PreprocessSettings,
    hdbet_venv_dir: Path,
    labels_dir: Optional[Path],
    labels_naming_scheme: str,
    save_transforms_flag: bool,
    overwrite: bool,
    verify: bool = True,
) -> pd.DataFrame:
    timepoint_groups = discover_timepoint_groups(input_dir, pattern, recursive, labels_dir, labels_naming_scheme)
    logger.info("Found %d timepoint(s) in %s", len(timepoint_groups), input_dir)

    fixed_img = None
    if settings.do_register:
        fixed_path = ensure_template(template_channel, template_path, template_cache_dir)
        logger.info("Using SRI24 '%s' template: %s", template_channel, fixed_path)
        fixed_img = ants.image_read(str(fixed_path))

    images_dir = output_dir / "imagesTr"
    labels_out_dir = output_dir / "labelsTr"
    masks_dir = output_dir / "masks"
    transforms_dir = output_dir / "transforms" if (save_transforms_flag and settings.do_register) else None
    tmp_dir = output_dir / "_intermediate"
    stage1_reference_dir = tmp_dir / "stage1_reference"  # fed to HD-BET, one file per case
    stage1_dependent_dir = tmp_dir / "stage1_dependent"
    stage1_labels_dir = tmp_dir / "stage1_labels"
    stage2_images_dir = tmp_dir / "stage2_skullstripped"
    stage2_dependent_masked_dir = tmp_dir / "stage2_dependent_masked"

    # ---- Stage 1: per timepoint, N4 + intra-subject coregister + register -
    cases: list[dict] = []
    for i, group in enumerate(timepoint_groups, start=1):
        case_id = case_id_for_timepoint(group.key)
        expected_paths = [images_dir / f"{case_id}_{NNUNET_CHANNEL_INDEX[m]}.nii.gz" for m in group.modalities]
        if group.label_path is not None:
            expected_paths.append(labels_out_dir / f"{case_id}.nii.gz")

        if not overwrite and all(p.exists() for p in expected_paths):
            logger.info("[%d/%d] %s: all outputs already exist, skipping (--overwrite to redo)", i, len(timepoint_groups), case_id)
            cases.append({"case_id": case_id, "group": group, "status": "skipped"})
            continue

        logger.info("[%d/%d] %s: stage 1 (register)", i, len(timepoint_groups), case_id)
        t0 = time.time()
        try:
            reference_modality = pick_reference_modality(group.modalities)
            reference_path = group.modalities[reference_modality]
            ref_info = stage1_register_reference(reference_path, fixed_img, settings)

            modality_results: dict[str, dict] = {reference_modality: ref_info}
            transformlists: dict[str, list] = {reference_modality: ref_info["fwd_transforms"]}
            for modality, dep_path in group.modalities.items():
                if modality == reference_modality:
                    continue
                dep_info = stage1_register_dependent(
                    dep_path, ref_info["reference_n4_img"], ref_info["group_fixed_img"], ref_info["fwd_transforms"], settings
                )
                modality_results[modality] = dep_info
                transformlists[modality] = dep_info["transformlist"]

            stage1_paths: dict[str, Path] = {}
            for modality, info in modality_results.items():
                out_path = (
                    (stage1_reference_dir / f"{case_id}.nii.gz")
                    if modality == reference_modality
                    else (stage1_dependent_dir / f"{case_id}_{modality}.nii.gz")
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ants.image_write(info["warped"], str(out_path))
                stage1_paths[modality] = out_path

            stage1_label_path = None
            label_status = ""
            if group.label_path is not None:
                if LABEL_HOST_MODALITY in transformlists:
                    label_img = ants.image_read(str(group.label_path))
                    tlist = transformlists[LABEL_HOST_MODALITY]
                    warped_label = (
                        ants.apply_transforms(
                            fixed=ref_info["group_fixed_img"], moving=label_img, transformlist=tlist, interpolator="genericLabel"
                        )
                        if tlist
                        else label_img
                    )
                    stage1_label_path = stage1_labels_dir / f"{case_id}.nii.gz"
                    stage1_label_path.parent.mkdir(parents=True, exist_ok=True)
                    ants.image_write(warped_label, str(stage1_label_path))
                    label_status = "ok"
                else:
                    label_status = "not_found"

            cases.append(
                {
                    "case_id": case_id,
                    "group": group,
                    "reference_modality": reference_modality,
                    "modality_results": modality_results,
                    "stage1_paths": stage1_paths,
                    "stage1_label_path": stage1_label_path,
                    "label_status": label_status,
                    "elapsed_stage1": time.time() - t0,
                    "status": "prepared",
                }
            )
        except Exception as exc:  # one bad timepoint shouldn't kill the whole batch
            logger.error("%s: stage 1 failed -- %s", case_id, exc)
            cases.append({"case_id": case_id, "group": group, "status": "failed", "error": str(exc), "elapsed_stage1": time.time() - t0})

    # ---- Stage 2: batch skull-strip (separate venv), reference images only
    if settings.do_skull_strip and any(c["status"] == "prepared" for c in cases):
        hdbet_bin = hdbet_venv_dir / "bin" / "hd-bet"
        if not hdbet_bin.is_file():
            raise FileNotFoundError(f"hd-bet not found at {hdbet_bin} -- run scripts/setup_hdbet_venv.sh first")
        device = resolve_hdbet_device(settings.hdbet_device, hdbet_venv_dir)
        logger.info("Stage 2 (skull-strip, batch, device=%s)", device)
        run_hdbet(stage1_reference_dir, stage2_images_dir, hdbet_bin, device, settings.hdbet_disable_tta)

    # ---- Stage 3: per timepoint, normalize + mask label with shared mask --
    results: list[PreprocessResult] = []
    for c in cases:
        case_id = c["case_id"]
        group: TimepointGroup = c["group"]

        if c["status"] == "skipped":
            for modality in group.modalities:
                results.append(PreprocessResult(case_id=case_id, modality=modality, input_path=group.modalities[modality], status="skipped"))
            if group.label_path is not None:
                results.append(PreprocessResult(case_id=case_id, modality="label", input_path=group.label_path, status="skipped"))
            continue
        if c["status"] == "failed":
            for modality in group.modalities:
                results.append(PreprocessResult(case_id=case_id, modality=modality, input_path=group.modalities[modality], status="failed", error=c["error"]))
            continue

        reference_modality = c["reference_modality"]
        stage1_paths = c["stage1_paths"]
        modality_results = c["modality_results"]

        mask_path = None
        if settings.do_skull_strip:
            candidate_mask = stage2_images_dir / f"{case_id}{MASK_SUFFIX}"
            if candidate_mask.is_file():
                mask_path = candidate_mask
        final_mask_path = None
        if mask_path is not None:
            final_mask_path = masks_dir / f"{case_id}_mask.nii.gz"
            final_mask_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mask_path, final_mask_path)
            # Keep the QC/audit mask on the same grid+orientation as the
            # final images it was derived from (stage3_finalize_image
            # reorients those; this copy predates that step).
            reorient_file_to_target(final_mask_path)

        for modality, stage1_path in stage1_paths.items():
            t0 = time.time()
            result = PreprocessResult(
                case_id=case_id, modality=modality, input_path=group.modalities[modality],
                input_shape=modality_results[modality]["input_shape"],
            )
            final_image_path = images_dir / f"{case_id}_{NNUNET_CHANNEL_INDEX[modality]}.nii.gz"
            try:
                if modality == reference_modality:
                    skull_stripped_path = stage2_images_dir / stage1_path.name if mask_path is not None else None
                else:
                    skull_stripped_path = None
                    if mask_path is not None:
                        skull_stripped_path = stage2_dependent_masked_dir / stage1_path.name
                        mask_image(stage1_path, mask_path, skull_stripped_path)

                info = stage3_finalize_image(stage1_path, skull_stripped_path, mask_path, final_image_path, settings)
                result.output_path = final_image_path
                result.output_shape = tuple(nib.load(str(final_image_path)).shape[:3])
                result.output_spacing = tuple(float(s) for s in nib.load(str(final_image_path)).header.get_zooms()[:3])
                result.nonzero_frac_after = info["nonzero_frac_after"]
                result.brain_mean_raw = info.get("brain_mean_raw")
                result.brain_std_raw = info.get("brain_std_raw")
                result.mask_path = final_mask_path
                result.steps_applied = ",".join(modality_results[modality]["steps"] + info["steps"])
                result.status = "ok"
            except Exception as exc:
                result.status = "failed"
                result.error = str(exc)
                logger.error("%s/%s: stage 3 failed -- %s", case_id, modality, exc)
            result.elapsed_sec = c["elapsed_stage1"] / len(stage1_paths) + (time.time() - t0)
            results.append(result)

        if group.label_path is not None:
            label_result = PreprocessResult(case_id=case_id, modality="label", input_path=group.label_path)
            label_result.label_input_path = group.label_path
            final_label_path = labels_out_dir / f"{case_id}.nii.gz"
            if c["stage1_label_path"] is not None:
                try:
                    if mask_path is not None:
                        mask_result = apply_mask_to_label(c["stage1_label_path"], mask_path, final_label_path)
                        label_result.label_status = mask_result.status
                        label_result.label_error = mask_result.error
                    else:
                        final_label_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(c["stage1_label_path"], final_label_path)
                        label_result.label_status = "ok"
                    if label_result.label_status == "ok":
                        # Keep the label on the same grid+orientation as its
                        # image (stage3_finalize_image reorients that).
                        reorient_file_to_target(final_label_path)
                    label_result.label_output_path = final_label_path
                    label_result.output_path = final_label_path
                    label_result.status = "ok" if label_result.label_status == "ok" else "failed"
                except Exception as exc:
                    label_result.status = "failed"
                    label_result.error = str(exc)
                    logger.error("%s: label finalize failed -- %s", case_id, exc)
            else:
                label_result.status = "failed"
                label_result.label_status = c.get("label_status", "not_found")
            results.append(label_result)

    if not (output_dir / "_keep_intermediate").exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Report -----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) | {"case": r.case} for r in results])
    df.to_csv(output_dir / "preprocess_report.csv", index=False)

    (output_dir / "preprocess_report.txt").write_text(format_report_text(results))
    (output_dir / "preprocess_summary.txt").write_text(format_summary_text(input_dir, output_dir, settings, results))

    n_ok = int((df["status"] == "ok").sum())
    n_failed = int((df["status"] == "failed").sum())
    n_skipped = int((df["status"] == "skipped").sum())
    logger.info(
        "Done: %d ok, %d failed, %d skipped. Reports written to %s", n_ok, n_failed, n_skipped, output_dir
    )

    if labels_dir is not None:
        dataset_json_path = write_dataset_json(output_dir, labels_out_dir)
        logger.info("Wrote %s", dataset_json_path)
    else:
        logger.info("--labels-dir not set -- skipping dataset.json (nnU-Net needs a label per training case)")

    # ---- Verification (independent re-check of the saved output files) --
    if verify:
        # Every final output is explicitly reoriented to BRATS_REFERENCE_ORIENTATION
        # in stage3_finalize_image/reorient_file_to_target, regardless of which
        # atlas channel registration used -- so that's what to verify against,
        # not whatever resolve_reference_orientation would read off the atlas.
        reference_orientation = "".join(BRATS_REFERENCE_ORIENTATION)
        image_results = [r for r in results if r.modality != "label"]
        verifications = run_verification(image_results, settings, reference_orientation)
        pd.DataFrame([asdict(v) for v in verifications]).to_csv(output_dir / "verification_report.csv", index=False)
        (output_dir / "verification_report.txt").write_text(format_verification_text(verifications))

        n_pass = sum(v.all_ok for v in verifications)
        if n_pass < len(verifications):
            logger.warning(
                "Verification: %d/%d case(s) FAILED at least one check -- see %s",
                len(verifications) - n_pass, len(verifications), output_dir / "verification_report.txt",
            )
        else:
            logger.info("Verification: %d/%d case(s) passed all checks", n_pass, len(verifications))

    return df


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_report_text(results: list[PreprocessResult]) -> str:
    lines = ["BraTS PREPROCESSING REPORT", "=" * 70, ""]
    for r in results:
        lines.append(f"[{r.case}] {r.input_path.name}")
        if r.status == "skipped":
            lines.append("    SKIPPED (output already exists, use --overwrite to redo)")
            lines.append("")
            continue
        if r.status == "failed":
            lines.append(f"    FAILED: {r.error}")
            lines.append("")
            continue
        if r.modality == "label":
            lines.append(f"    label: {r.label_status}" + (f" -- {r.label_error}" if r.label_error else ""))
            lines.append("")
            continue
        lines.append(f"    steps applied: {r.steps_applied}")
        lines.append(f"    shape: {r.input_shape} -> {r.output_shape}")
        lines.append(f"    spacing_mm: -> {r.output_spacing}")
        lines.append(f"    nonzero fraction after: {r.nonzero_frac_after:.1%}")
        if r.brain_mean_raw is not None:
            lines.append(
                f"    pre-normalization brain intensity: mean={r.brain_mean_raw:.3f} std={r.brain_std_raw:.3f} "
                "(the z-score parameters -- output voxels within the brain mask are scaled to mean~0/std~1)"
            )
        lines.append(f"    elapsed: {r.elapsed_sec:.1f}s")
        lines.append("")
    return "\n".join(lines)


def format_summary_text(
    input_dir: Path, output_dir: Path, settings: PreprocessSettings, results: list[PreprocessResult]
) -> str:
    lines = ["BraTS PREPROCESSING SUMMARY", "=" * 70]
    lines.append(f"Input:  {input_dir}")
    lines.append(f"Output: {output_dir}")
    lines.append("")
    lines.append("Steps enabled:")
    lines.append(f"  N4 bias correction:   {settings.n4_correct}")
    lines.append(f"  Co-register to SRI24: {settings.do_register} (transform={settings.transform_type})")
    lines.append(f"  Standalone resample:  {settings.do_resample} (only applies if register is off)")
    lines.append(f"  Skull-strip (HD-BET), shared mask per case: {settings.do_skull_strip} (device={settings.hdbet_device or 'auto'})")
    lines.append(f"  Z-score normalize:    {settings.do_normalize}")
    lines.append("")

    image_results = [r for r in results if r.modality != "label"]
    label_results = [r for r in results if r.modality == "label"]
    n_total = len(image_results)
    ok = [r for r in image_results if r.status == "ok"]
    n_ok, n_failed, n_skipped = len(ok), sum(r.status == "failed" for r in image_results), sum(r.status == "skipped" for r in image_results)
    n_cases = len({r.case_id for r in results})
    n_label_ok = sum(1 for r in label_results if r.status == "ok")

    lines.append(f"Timepoints (cases): {n_cases}")
    lines.append(f"Image files: {n_total} total -- {n_ok} ok, {n_failed} failed, {n_skipped} skipped")
    if label_results:
        lines.append(f"Labels: {len(label_results)} case(s) had a matching label -- {n_label_ok} preprocessed ok")

    if ok:
        shapes = sorted({r.output_shape for r in ok}, key=str)
        spacings = sorted({r.output_spacing for r in ok}, key=str)
        lines.append(f"Output shapes seen: {shapes}")
        lines.append(f"Output spacings seen (mm): {spacings}")
        total_sec = sum(r.elapsed_sec for r in ok)
        lines.append(f"Total per-file elapsed: {total_sec:.1f}s")
        if settings.do_normalize:
            means = [r.brain_mean_raw for r in ok if r.brain_mean_raw is not None]
            stds = [r.brain_std_raw for r in ok if r.brain_std_raw is not None]
            if means:
                lines.append(
                    f"Pre-normalization brain intensity across cases: mean={np.mean(means):.2f} "
                    f"(range {min(means):.2f}-{max(means):.2f}), std={np.mean(stds):.2f} "
                    f"(range {min(stds):.2f}-{max(stds):.2f}) -- wide spread here can flag scans with "
                    "unusual raw contrast/protocol before normalization masks it"
                )

    if n_failed:
        lines.append("")
        lines.append(f"BLOCKING: {n_failed} image file(s) failed -- see preprocess_report.txt for details.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BraTS-matched multi-modality preprocessing: N4, intra-subject coregister to one reference "
        "modality per timepoint, co-register to SRI24, shared skull-strip mask, z-score normalize -- nnU-Net-ready output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing scans to preprocess")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where all outputs are written (imagesTr/, labelsTr/, masks/, reports)")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --input-dir) for scans")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Search --input-dir recursively")

    parser.add_argument("--labels-dir", type=Path, default=None, help="Folder of ground-truth labels to preprocess alongside their matching timepoint (looked up via the FLAIR scan)")
    parser.add_argument(
        "--labels-naming-scheme",
        choices=sorted(LABEL_NAMING_SCHEMES),
        default="identical",
        help="'identical' (same filename as the FLAIR scan), or 'braintracking' (scan 'flair_2016_11.nii.gz' -> label 'tumor_2016-11.nii.gz')",
    )

    parser.add_argument("--n4-correct", action=argparse.BooleanOptionalAction, default=True, help="Step 1: N4 bias field correction")

    parser.add_argument("--register", dest="do_register", action=argparse.BooleanOptionalAction, default=True, help="Step 3: rigid co-registration of the reference modality to SRI24 (also yields isotropic spacing)")
    parser.add_argument("--template-channel", choices=sorted(SRI24_CHANNELS), default="spgr_unstrip", help="SRI24 channel to register to (spgr_unstrip=skull-on, spgr=skull-stripped)")
    parser.add_argument("--template-path", type=Path, default=None, help="Use this NIfTI file instead of auto-downloading --template-channel")
    parser.add_argument("--template-cache-dir", type=Path, default=DEFAULT_TEMPLATE_CACHE_DIR, help="Where downloaded SRI24 template channels are cached")
    parser.add_argument("--transform-type", choices=["Rigid", "Affine", "SyN", "SyNRA"], default="Rigid", help="ANTs transform type, used for both intra-subject coregistration and atlas registration (BraTS uses Rigid to preserve true volume)")
    parser.add_argument("--interpolator", choices=["linear", "bSpline", "nearestNeighbor"], default="linear", help="Interpolation for resampling images (not labels, which always use a label-preserving interpolator)")

    parser.add_argument("--resample", dest="do_resample", action=argparse.BooleanOptionalAction, default=True, help="Standalone isotropic resample of the reference modality -- only takes effect when --no-register is set")

    parser.add_argument("--skull-strip", dest="do_skull_strip", action=argparse.BooleanOptionalAction, default=True, help="Step 4: skull-strip the reference modality via HD-BET and reuse that ONE mask for all modalities of the timepoint (needs scripts/setup_hdbet_venv.sh run once first)")
    parser.add_argument("--hdbet-venv-dir", type=Path, default=None, help="Where HD-BET's isolated venv lives (default: '<repo>/hdbet_venv')")
    parser.add_argument("--hdbet-device", default="", help="cpu, cuda, or mps -- empty auto-detects (cuda if available, else cpu)")
    parser.add_argument("--hdbet-disable-tta", action="store_true", help="Disable HD-BET test-time augmentation (faster, slightly lower quality -- consider it on cpu)")

    parser.add_argument("--normalize", dest="do_normalize", action=argparse.BooleanOptionalAction, default=False, help="Step 5: z-score intensity normalization within the shared brain mask -- OFF by default: nnU-Net's raw dataset format expects raw intensities and normalizes itself at train/inference time, using parameters fit against a raw distribution, so pre-normalizing here causes double normalization")

    parser.add_argument("--save-transforms", action=argparse.BooleanOptionalAction, default=True, help="Persist each case's registration transforms under output_dir/transforms/ (only if --register)")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After processing, independently re-check the saved output files (shape/spacing/orientation, "
        "skull-strip status, normalization stats, image/label grid match, label discreteness -- whichever "
        "apply given the steps that ran) and write verification_report.csv/.txt",
    )
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep output_dir/_intermediate/ (per-stage files) instead of deleting it at the end")
    parser.add_argument("--overwrite", action="store_true", help="Re-process timepoints whose final outputs already exist")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    settings = PreprocessSettings(
        n4_correct=args.n4_correct,
        do_register=args.do_register,
        transform_type=args.transform_type,
        do_resample=args.do_resample,
        interpolator=args.interpolator,
        do_skull_strip=args.do_skull_strip,
        hdbet_device=args.hdbet_device,
        hdbet_disable_tta=args.hdbet_disable_tta,
        do_normalize=args.do_normalize,
    )

    hdbet_venv_dir = args.hdbet_venv_dir or (Path(__file__).resolve().parent.parent / "hdbet_venv")

    if args.keep_intermediate:
        (args.output_dir / "_keep_intermediate").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "_keep_intermediate").touch()

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        template_channel=args.template_channel,
        template_path=args.template_path,
        template_cache_dir=args.template_cache_dir,
        settings=settings,
        hdbet_venv_dir=hdbet_venv_dir,
        labels_dir=args.labels_dir,
        labels_naming_scheme=args.labels_naming_scheme,
        save_transforms_flag=args.save_transforms,
        overwrite=args.overwrite,
        verify=args.verify,
    )

    if not args.keep_intermediate:
        marker = args.output_dir / "_keep_intermediate"
        if marker.exists():
            marker.unlink()


if __name__ == "__main__":
    main()
