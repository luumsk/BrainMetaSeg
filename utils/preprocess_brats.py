#!/usr/bin/env python3
"""Single-entry-point BraTS-like preprocessing pipeline: one input folder in,
one output folder out.

Runs the standard BraTS preprocessing steps, each independently toggleable:
    1. N4 bias field correction        (--n4-correct)
    2. Rigid co-registration to SRI24  (--register)      -- also produces
       isotropic 1mm spacing as a side effect, since the SRI24 template
       itself is the 1mm grid being registered onto (see register_to_sri24.py
       for why this is one step, not two, in the standard pipeline).
    3. Standalone isotropic resample   (--resample)       -- only takes
       effect if --no-register is set; otherwise step 2 already produced
       isotropic spacing and this is a no-op (added deliberately so you
       still get one flag per BraTS step, without double-resampling).
    4. Skull-stripping (HD-BET)        (--skull-strip)
    5. Z-score intensity normalization (--normalize)      -- computed from
       brain-tissue voxels only (the mask from step 4, or a nonzero-voxel
       approximation if skull-stripping was skipped), background forced to
       exactly 0.
Ground-truth labels (--labels-dir) ride along through steps 2/3 (a
label-preserving interpolator, never blurred like the image) and get the
same brain mask applied in step 4/5 -- HD-BET itself only ever sees images,
never labels, since it's a network trained on MRI contrast.

Step 4 needs its own venv (nnunetv2 version conflict with this repo's own
fork -- see setup_hdbet_venv.sh) so this script shells out to it as a
subprocess for that one step; everything else runs in-process here.

After processing, a verification pass (--verify, on by default) re-loads
every saved output file from disk and independently re-derives shape,
spacing, orientation, skull-strip status, and normalization statistics --
rather than re-trusting the values already computed during processing. This
is deliberate: it's exactly how a real bug got caught during development
(the report briefly mislabeled pre-normalization stats as post-normalization
ones -- the processing itself was correct, but nothing during processing
would have caught the report being wrong). Also checks that each label
shares its image's exact grid and stayed discrete (no interpolation
corruption).

Modality completeness (T1/T1c/T2/FLAIR) is intentionally NOT checked or
enforced here -- see check_brats_format.py if you need that; this pipeline
processes whatever's in --input-dir regardless of which/how many
modalities are present.

Output layout
-------------
    output_dir/
        images/...                  final preprocessed scans (mirrors
                                     --input-dir's subfolder structure)
        labels/...                  final preprocessed labels, if
                                     --labels-dir was given
        masks/...                   brain masks, if --skull-strip ran
        transforms/<case>/...       saved registration transforms, if
                                     --register and --save-transforms
        preprocess_report.csv       one row per case: status, timing,
                                     shapes/spacings, brain intensity stats
        preprocess_report.txt       same, human-readable, per case
        preprocess_summary.txt      aggregate-only summary (counts, which
                                     steps ran, distinct shapes/spacings)
        verification_report.csv/.txt  independent post-hoc check of the
                                     saved outputs (see above), if --verify

Usage
-----
    python utils/preprocess_brats.py \\
        --input-dir nifti_native --output-dir brats_preprocessed \\
        --labels-dir tumor_volume --labels-naming-scheme braintracking
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import ants
import nibabel as nib
import numpy as np
import pandas as pd

from apply_brain_mask import apply_mask_to_label
from check_brats_format import (
    BRATS_REFERENCE_SHAPE,
    BRATS_REFERENCE_SPACING_MM,
    DEFAULT_SKULL_STRIPPED_NONZERO_THRESHOLD,
    resolve_reference_orientation,
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
    do_normalize: bool = True


@dataclass
class PreprocessResult:
    case: str
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


# --------------------------------------------------------------------------
# Stage 1 (per case, main venv/ants): N4 + register-or-resample-or-passthrough
# --------------------------------------------------------------------------


def resample_isotropic(img: "ants.ANTsImage", interpolator: str) -> "ants.ANTsImage":
    return ants.resample_image(img, (1.0, 1.0, 1.0), use_voxels=False, interp_type=INTERP_CODES[interpolator])


def stage1_prepare(
    scan_path: Path,
    label_path: Optional[Path],
    fixed_img: Optional["ants.ANTsImage"],
    out_image_path: Path,
    out_label_path: Optional[Path],
    transforms_dir: Optional[Path],
    case: str,
    settings: PreprocessSettings,
) -> dict:
    moving = ants.image_read(str(scan_path))
    info = {
        "input_shape": tuple(moving.shape),
        "input_spacing": tuple(moving.spacing),
        "nonzero_frac_before": float((moving.numpy() > 0).mean()),
        "steps": [],
    }

    if settings.n4_correct:
        moving = ants.n4_bias_field_correction(moving)
        info["steps"].append("n4")

    label_img = ants.image_read(str(label_path)) if label_path is not None else None
    warped_label = None

    if settings.do_register:
        reg = ants.registration(fixed=fixed_img, moving=moving, type_of_transform=settings.transform_type)
        warped = ants.apply_transforms(
            fixed=fixed_img, moving=moving, transformlist=reg["fwdtransforms"], interpolator=settings.interpolator
        )
        info["steps"].append("register")
        if transforms_dir is not None:
            save_transforms(reg["fwdtransforms"], transforms_dir / case)
        if label_img is not None:
            warped_label = ants.apply_transforms(
                fixed=fixed_img, moving=label_img, transformlist=reg["fwdtransforms"], interpolator="genericLabel"
            )
    elif settings.do_resample:
        warped = resample_isotropic(moving, settings.interpolator)
        info["steps"].append("resample")
        if label_img is not None:
            warped_label = resample_isotropic(label_img, "nearestNeighbor")
    else:
        warped = moving
        if label_img is not None:
            warped_label = label_img

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(out_image_path))
    info["output_shape"] = tuple(warped.shape)
    info["output_spacing"] = tuple(warped.spacing)

    if warped_label is not None:
        out_label_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(warped_label, str(out_label_path))

    return info


# --------------------------------------------------------------------------
# Stage 2 (batch subprocess, isolated hdbet_venv): skull-strip images only
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
# Stage 3 (per case, main venv/numpy): normalize image, mask label
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
    scans = discover_scans(input_dir, pattern, recursive)
    logger.info("Found %d scan(s) in %s", len(scans), input_dir)

    fixed_img = None
    if settings.do_register:
        fixed_path = ensure_template(template_channel, template_path, template_cache_dir)
        logger.info("Using SRI24 '%s' template: %s", template_channel, fixed_path)
        fixed_img = ants.image_read(str(fixed_path))

    map_scan_to_label = LABEL_NAMING_SCHEMES[labels_naming_scheme] if labels_dir is not None else None

    images_dir = output_dir / "images"
    labels_out_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    transforms_dir = output_dir / "transforms" if (save_transforms_flag and settings.do_register) else None
    tmp_dir = output_dir / "_intermediate"
    stage1_images_dir = tmp_dir / "stage1_images"
    stage1_labels_dir = tmp_dir / "stage1_labels"
    stage2_images_dir = tmp_dir / "stage2_skullstripped"

    # ---- Stage 1: per case, N4 + register/resample/passthrough ----------
    cases: list[dict] = []
    for i, scan_path in enumerate(scans, start=1):
        rel_path = scan_path.relative_to(input_dir)
        case = str((rel_path.parent / strip_nifti_suffix(scan_path.name)).as_posix())
        final_image_path = images_dir / rel_path.parent / f"{strip_nifti_suffix(scan_path.name)}.nii.gz"

        if final_image_path.exists() and not overwrite:
            logger.info("[%d/%d] %s: output already exists, skipping (--overwrite to redo)", i, len(scans), case)
            cases.append({"case": case, "scan_path": scan_path, "status": "skipped"})
            continue

        label_path = None
        if map_scan_to_label is not None:
            try:
                candidate = labels_dir / map_scan_to_label(scan_path.name)
                if candidate.is_file():
                    label_path = candidate
                else:
                    logger.info("%s: no matching label at %s -- processing scan only", case, candidate)
            except ValueError as exc:
                logger.warning("%s: could not derive label filename -- %s", case, exc)

        stage1_out = stage1_images_dir / rel_path.parent / f"{strip_nifti_suffix(scan_path.name)}.nii.gz"
        stage1_label_out = (
            stage1_labels_dir / map_scan_to_label(scan_path.name) if label_path is not None else None
        )

        result = PreprocessResult(case=case, input_path=scan_path, label_input_path=label_path)
        t0 = time.time()
        try:
            logger.info("[%d/%d] %s: stage 1 (prepare)", i, len(scans), case)
            info = stage1_prepare(
                scan_path, label_path, fixed_img, stage1_out, stage1_label_out, transforms_dir, case, settings
            )
            result.input_shape = info["input_shape"]
            result.input_spacing = info["input_spacing"]
            result.nonzero_frac_before = info["nonzero_frac_before"]
            result.output_shape = info["output_shape"]
            result.output_spacing = info["output_spacing"]
            result.steps_applied = ",".join(info["steps"])
            if label_path is not None:
                result.label_status = "ok" if stage1_label_out.is_file() else "failed"
        except Exception as exc:  # one bad scan shouldn't kill the whole batch
            result.status = "failed"
            result.error = str(exc)
            result.elapsed_sec = time.time() - t0
            logger.error("%s: stage 1 failed -- %s", case, exc)
            cases.append({"case": case, "scan_path": scan_path, "result": result, "status": "failed"})
            continue
        result.elapsed_sec = time.time() - t0

        cases.append(
            {
                "case": case,
                "scan_path": scan_path,
                "rel_dir": rel_path.parent,
                "stage1_image": stage1_out,
                "stage1_label": stage1_label_out,
                "label_filename": map_scan_to_label(scan_path.name) if label_path is not None else None,
                "final_image_path": final_image_path,
                "result": result,
                "status": "prepared",
            }
        )

    # ---- Stage 2: batch skull-strip (separate venv, images only) --------
    if settings.do_skull_strip and any(c["status"] == "prepared" for c in cases):
        hdbet_bin = hdbet_venv_dir / "bin" / "hd-bet"
        if not hdbet_bin.is_file():
            raise FileNotFoundError(f"hd-bet not found at {hdbet_bin} -- run scripts/setup_hdbet_venv.sh first")
        device = resolve_hdbet_device(settings.hdbet_device, hdbet_venv_dir)
        logger.info("Stage 2 (skull-strip, batch, device=%s)", device)
        run_hdbet(stage1_images_dir, stage2_images_dir, hdbet_bin, device, settings.hdbet_disable_tta)

    # ---- Stage 3: per case, normalize + mask label -----------------------
    for c in cases:
        if c["status"] != "prepared":
            continue
        case = c["case"]
        result: PreprocessResult = c["result"]
        t0 = time.time()
        try:
            skull_stripped_path = None
            mask_path = None
            if settings.do_skull_strip:
                candidate_img = stage2_images_dir / c["stage1_image"].name
                candidate_mask = stage2_images_dir / f"{strip_nifti_suffix(c['stage1_image'].name)}{MASK_SUFFIX}"
                if candidate_img.is_file():
                    skull_stripped_path = candidate_img
                if candidate_mask.is_file():
                    mask_path = candidate_mask

            logger.info("[%s] stage 3 (finalize)", case)
            info = stage3_finalize_image(
                c["stage1_image"], skull_stripped_path, mask_path, c["final_image_path"], settings
            )
            result.output_path = c["final_image_path"]
            result.nonzero_frac_after = info["nonzero_frac_after"]
            result.brain_mean_raw = info.get("brain_mean_raw")
            result.brain_std_raw = info.get("brain_std_raw")
            if info["steps"]:
                result.steps_applied = ",".join([s for s in result.steps_applied.split(",") if s] + info["steps"])
            if mask_path is not None:
                final_mask_path = masks_dir / c["rel_dir"] / mask_path.name
                final_mask_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mask_path, final_mask_path)
                result.mask_path = final_mask_path

            if c["stage1_label"] is not None and c["stage1_label"].is_file():
                final_label_path = labels_out_dir / c["rel_dir"] / c["label_filename"]
                if mask_path is not None:
                    mask_result = apply_mask_to_label(c["stage1_label"], mask_path, final_label_path)
                    result.label_status = mask_result.status
                    result.label_error = mask_result.error
                else:
                    final_label_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(c["stage1_label"], final_label_path)
                    result.label_status = "ok"
                result.label_output_path = final_label_path

            result.status = "ok"
        except Exception as exc:
            result.status = "failed"
            result.error = str(exc)
            logger.error("%s: stage 3 failed -- %s", case, exc)
        result.elapsed_sec += time.time() - t0

    if not (output_dir / "_keep_intermediate").exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- Report -----------------------------------------------------------
    results = []
    for c in cases:
        if "result" in c:
            results.append(c["result"])
        else:
            results.append(PreprocessResult(case=c["case"], input_path=c["scan_path"], status=c["status"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "preprocess_report.csv", index=False)

    (output_dir / "preprocess_report.txt").write_text(format_report_text(results))
    (output_dir / "preprocess_summary.txt").write_text(format_summary_text(input_dir, output_dir, settings, results))

    n_ok = int((df["status"] == "ok").sum())
    n_failed = int((df["status"] == "failed").sum())
    n_skipped = int((df["status"] == "skipped").sum())
    logger.info(
        "Done: %d ok, %d failed, %d skipped. Reports written to %s", n_ok, n_failed, n_skipped, output_dir
    )

    # ---- Verification (independent re-check of the saved output files) --
    if verify:
        reference_orientation = None
        if settings.do_register:
            reference_orientation = resolve_reference_orientation(template_channel, template_cache_dir)
        verifications = run_verification(results, settings, reference_orientation)
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
        lines.append(f"    steps applied: {r.steps_applied}")
        lines.append(f"    shape: {r.input_shape} -> {r.output_shape}")
        lines.append(f"    spacing_mm: {r.input_spacing} -> {r.output_spacing}")
        lines.append(f"    nonzero fraction: {r.nonzero_frac_before:.1%} -> {r.nonzero_frac_after:.1%}")
        if r.brain_mean_raw is not None:
            lines.append(
                f"    pre-normalization brain intensity: mean={r.brain_mean_raw:.3f} std={r.brain_std_raw:.3f} "
                "(the z-score parameters -- output voxels within the brain mask are scaled to mean~0/std~1)"
            )
        lines.append(f"    elapsed: {r.elapsed_sec:.1f}s")
        if r.label_input_path is not None:
            lines.append(f"    label: {r.label_status}" + (f" -- {r.label_error}" if r.label_error else ""))
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
    lines.append(f"  Skull-strip (HD-BET): {settings.do_skull_strip} (device={settings.hdbet_device or 'auto'})")
    lines.append(f"  Z-score normalize:    {settings.do_normalize}")
    lines.append("")

    n_total = len(results)
    ok = [r for r in results if r.status == "ok"]
    n_ok, n_failed, n_skipped = len(ok), sum(r.status == "failed" for r in results), sum(r.status == "skipped" for r in results)
    n_with_label = sum(1 for r in ok if r.label_input_path is not None)
    n_label_ok = sum(1 for r in ok if r.label_status == "ok")

    lines.append(f"Cases: {n_total} total -- {n_ok} ok, {n_failed} failed, {n_skipped} skipped")
    if n_with_label:
        lines.append(f"Labels: {n_with_label} case(s) had a matching label -- {n_label_ok} preprocessed ok")

    if ok:
        shapes = sorted({r.output_shape for r in ok}, key=str)
        spacings = sorted({r.output_spacing for r in ok}, key=str)
        lines.append(f"Output shapes seen: {shapes}")
        lines.append(f"Output spacings seen (mm): {spacings}")
        total_sec = sum(r.elapsed_sec for r in ok)
        lines.append(f"Total per-case elapsed (excludes batch skull-strip time): {total_sec:.1f}s")
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
        lines.append(f"BLOCKING: {n_failed} case(s) failed -- see preprocess_report.txt for details.")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-entry-point BraTS-like preprocessing: N4, co-register, skull-strip, z-score normalize -- one input folder, one output folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing scans to preprocess")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where all outputs are written (images/, labels/, masks/, reports)")
    parser.add_argument("--pattern", default="*.nii.gz", help="Glob pattern (relative to --input-dir) for scans")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Search --input-dir recursively")

    parser.add_argument("--labels-dir", type=Path, default=None, help="Folder of ground-truth labels to preprocess alongside their matching scan")
    parser.add_argument(
        "--labels-naming-scheme",
        choices=sorted(LABEL_NAMING_SCHEMES),
        default="identical",
        help="'identical' (same filename), or 'braintracking' (scan 'flair_2016_11.nii.gz' -> label 'tumor_2016-11.nii.gz')",
    )

    parser.add_argument("--n4-correct", action=argparse.BooleanOptionalAction, default=True, help="Step 1: N4 bias field correction")

    parser.add_argument("--register", dest="do_register", action=argparse.BooleanOptionalAction, default=True, help="Step 2: rigid co-registration to SRI24 (also yields isotropic spacing)")
    parser.add_argument("--template-channel", choices=sorted(SRI24_CHANNELS), default="spgr_unstrip", help="SRI24 channel to register to (spgr_unstrip=skull-on, spgr=skull-stripped)")
    parser.add_argument("--template-path", type=Path, default=None, help="Use this NIfTI file instead of auto-downloading --template-channel")
    parser.add_argument("--template-cache-dir", type=Path, default=DEFAULT_TEMPLATE_CACHE_DIR, help="Where downloaded SRI24 template channels are cached")
    parser.add_argument("--transform-type", choices=["Rigid", "Affine", "SyN", "SyNRA"], default="Rigid", help="ANTs transform type for --register (BraTS uses Rigid to preserve true volume)")
    parser.add_argument("--interpolator", choices=["linear", "bSpline", "nearestNeighbor"], default="linear", help="Interpolation for resampling the image (not the label, which always uses a label-preserving interpolator)")

    parser.add_argument("--resample", dest="do_resample", action=argparse.BooleanOptionalAction, default=True, help="Step 3: standalone isotropic resample -- only takes effect when --no-register is set")

    parser.add_argument("--skull-strip", dest="do_skull_strip", action=argparse.BooleanOptionalAction, default=True, help="Step 4: skull-strip via HD-BET (needs scripts/setup_hdbet_venv.sh run once first)")
    parser.add_argument("--hdbet-venv-dir", type=Path, default=None, help="Where HD-BET's isolated venv lives (default: '<repo>/hdbet_venv')")
    parser.add_argument("--hdbet-device", default="", help="cpu, cuda, or mps -- empty auto-detects (cuda if available, else cpu)")
    parser.add_argument("--hdbet-disable-tta", action="store_true", help="Disable HD-BET test-time augmentation (faster, slightly lower quality -- consider it on cpu)")

    parser.add_argument("--normalize", dest="do_normalize", action=argparse.BooleanOptionalAction, default=True, help="Step 5: z-score intensity normalization within the brain mask")

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
    parser.add_argument("--overwrite", action="store_true", help="Re-process cases whose final output already exists")
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
