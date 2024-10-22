from argparse import ArgumentParser, Namespace
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import segmentationmetrics as sm
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog='meta24_compute_metrics.py',
        description=(
            'Compute segmentation metrics for META24 predictions '
            + 'and store as csv'
        ),
    )
    parser.add_argument(
        '--pr',
        type=str,
        required=True,
        dest='pr',
        help=(
            'Directory with predicted masks. '
            + 'For each there must be a corresponding ground truth mask '
            + 'with the same file name in the GT directory'
        ),
    )
    parser.add_argument(
        '--gt',
        type=str,
        required=True,
        dest='gt',
        help='Directory with ground truth masks.',
    )
    parser.add_argument(
        '-o', '--out',
        type=str,
        required=False,
        default='table.csv',
        dest='output_file',
        help='Output CSV file',
    )
    return parser.parse_args()


def et(mask: np.ndarray) -> np.ndarray:
    return mask == 3


def tc(mask: np.ndarray) -> np.ndarray:
    return (mask == 1) | (mask == 3)


def wt(mask: np.ndarray) -> np.ndarray:
    return (mask != 0)


def compute_sample_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    zoom: tuple[float, float],
) -> dict[str, float]:
    et_metrics = sm.SegmentationMetrics(
        prediction=et(pred_mask),
        truth=et(true_mask),
        zoom=zoom,
    )
    tc_metrics = sm.SegmentationMetrics(
        prediction=tc(pred_mask),
        truth=tc(true_mask),
        zoom=zoom,
    )
    wt_metrics = sm.SegmentationMetrics(
        prediction=wt(pred_mask),
        truth=wt(true_mask),
        zoom=zoom,
    )
    return {
        'DICE_et': et_metrics.dice,
        'DICE_tc': tc_metrics.dice,
        'DICE_wt': wt_metrics.dice,
        'Hausdorff_et': et_metrics.hausdorff_distance,
        'Hausdorff_tc': tc_metrics.hausdorff_distance,
        'Hausdorff_wt': wt_metrics.hausdorff_distance,
        'Sensitivity_et': et_metrics.sensitivity,
        'Sensitivity_tc': tc_metrics.sensitivity,
        'Sensitivity_wt': wt_metrics.sensitivity,
        'Specificity_et': et_metrics.specificity,
        'Specificity_tc': tc_metrics.specificity,
        'Specificity_wt': wt_metrics.specificity,
        'True_volume_et': et_metrics.true_volume,
        'True_volume_tc': tc_metrics.true_volume,
        'True_volume_wt': wt_metrics.true_volume,
        'Predicted_volume_et': et_metrics.predicted_volume,
        'Predicted_volume_tc': tc_metrics.predicted_volume,
        'Predicted_volume_wt': wt_metrics.predicted_volume,
    }


def evaluate_model(
    gt_folder: str,
    pr_folder: str,
) -> pd.DataFrame:
    entries = []
    for pred_mask_path in tqdm(glob.glob(f'{pr_folder}/*.nii.gz')):
        file_name = os.path.basename(pred_mask_path)
        true_mask_path = f'{gt_folder}/{file_name}'
        pred_mask_file = nib.load(pred_mask_path)
        true_mask_file = nib.load(true_mask_path)

        pred_mask = np.round(pred_mask_file.get_fdata()).astype(np.int64)
        true_mask = np.round(true_mask_file.get_fdata()).astype(np.int64)
        sample_metrics = compute_sample_metrics(
            pred_mask=pred_mask,
            true_mask=true_mask,
            zoom=true_mask_file.header.get_zooms(),
        )
        entries.append({
            'sample': file_name,
            **sample_metrics,
        })

    return pd.DataFrame(entries)


def main():
    args = parse_args()
    stats = evaluate_model(
        gt_folder=args.gt,
        pr_folder=args.pr,
    )
    stats.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
