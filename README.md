# BrainMetaSeg: Transfer Learning Approaches for Brain Metastases Screenings

[![Paper](https://img.shields.io/badge/Biomedicines-10.3390%2Fbiomedicines12112561-blue.svg)](https://doi.org/10.3390/biomedicines12112561)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for our study on transfer learning for brain metastasis segmentation in MRI.
Segmentation models are pretrained on a large public multi-institutional dataset
(BraTS Metastasis 2024) and fine-tuned on a small clinical dataset (Siberian Brain Tumor, SBT).
They are compared with the same models trained from scratch on the clinical data.

> Minh Sao Khue Luu, Bair N. Tuchinov, Victor Suvorov, Roman M. Kenzhin, Evgeniya V. Amelina, Andrey Yu. Letyagin.
> *Transfer Learning Approaches for Brain Metastases Screenings.*
> Biomedicines, 12(11):2561, 2024.
> [[paper]](https://www.mdpi.com/2227-9059/12/11/2561)

## Overview

Three configurations are trained within the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
framework (`3d_fullres`) on the three nested BraTS regions: enhancing tumor (ET),
tumor core (TC) and whole tumor (WT).

| Configuration | Trainer | Architecture | Loss | Optimizer |
| --- | --- | --- | --- | --- |
| Default | `nnUNetTrainer_DefaultLoss` | nnU-Net U-Net, deep supervision | Dice + BCE | SGD (Nesterov, momentum 0.99), lr 1e-2, poly decay |
| TverskyBCE | `nnUNetTrainer_TverskyBCE` | nnU-Net U-Net, deep supervision | Tversky (α = 0.3, β = 0.7) + BCE weighted ×11 on tumor voxels | SGD, lr 1e-2, poly decay |
| SegResNet | `nnUNetTrainerSegResNet` | MONAI SegResNet (32 filters, blocks (1, 2, 2, 4)), no deep supervision | Dice + BCE | Adam, lr 1e-4, weight decay 1e-5, grad clip 12 |

**Training protocol**

1. Pretrain each configuration on 652 BraTS Metastasis 2024 training cases with 5-fold cross-validation (200 epochs).
2. Fine-tune from the pretrained weights on 26 SBT training cases (100 epochs).
3. Train the same configuration from random initialization on the same 26 SBT cases (100 epochs).
4. Evaluate the fold ensemble of each model on 10 held-out SBT test cases.

The `nnUNetTrainerSegResNet_2xFeat*` trainers are wider and deeper SegResNet variants.
They were not used in the paper.

## Results

The 10 SBT test cases, averaged over the three configurations for each case (mean ± std).
P-values are from a one-sided paired Wilcoxon signed-rank test with Holm correction over the nine comparisons.

| Metric | Region | Fine-tuned | Scratch | FT better | p (Holm) |
| --- | --- | --- | --- | --- | --- |
| DSC ↑ | ET | 0.905 ± 0.060 | 0.899 ± 0.061 | 8/10 | 0.059 |
| | TC | 0.938 ± 0.068 | 0.930 ± 0.066 | 8/10 | 0.059 |
| | WT | 0.921 ± 0.042 | 0.917 ± 0.043 | 7/10 | 0.290 |
| HD95 (mm) ↓ | ET | 2.66 ± 2.10 | 3.76 ± 3.38 | 9/10 | 0.059 |
| | TC | 1.89 ± 1.91 | 4.56 ± 3.77 | 9/10 | **0.018** |
| | WT | 3.02 ± 2.24 | 2.94 ± 2.08 | 5/10 | 0.500 |
| Sensitivity ↑ | ET | 0.907 ± 0.090 | 0.894 ± 0.096 | 8/10 | **0.048** |
| | TC | 0.919 ± 0.106 | 0.903 ± 0.111 | 9/10 | **0.039** |
| | WT | 0.881 ± 0.071 | 0.874 ± 0.077 | 7/10 | 0.290 |

- The gains from fine-tuning are small but consistent in direction. Fine-tuning gave no clear benefit for the WT.
- The benefit was largest for TverskyBCE: fine-tuning increased TC DSC in all ten test cases (0.942 → 0.955).
- Most of the gain comes from detection. Over the three configurations, fine-tuned models detected 88 of 99 TC lesions, against 78 for scratch-trained models (lesions ≥ 20 mm³, 26-connectivity).
- In an expert review, both fine-tuned models still missed small, scattered metastases, even in cases with high scores.

These numbers come from the re-evaluation in [`scores/*_2026-09-29.csv`](scores/) and
[`utils/statistical_tests_2026-09-29.py`](utils/statistical_tests_2026-09-29.py). They are the numbers reported in the
author's dissertation. The paper reports the original evaluation, in which the improvement from fine-tuning was not statistically significant.

## Installation

Tested with Python 3.10, PyTorch 2.0.1 and MONAI 1.3.0.

```bash
# 1. Clone this repository and install its dependencies
git clone https://github.com/luumsk/BrainMetaSeg.git
cd BrainMetaSeg
pip install -r requirements.txt

# 2. Install nnU-Net v2 from source, next to this repository
git clone https://github.com/MIC-DKFZ/nnUNet.git ../nnUNet
pip install -e ../nnUNet

# 3. Copy the custom trainers into nnU-Net
cp trainers/*.py ../nnUNet/nnunetv2/training/nnUNetTrainer/

# 4. Set the nnU-Net data paths (edit the file first)
source scripts/setvars.sh
```

For a CUDA build of PyTorch, install it before step 1, for example:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Data

- **BraTS Metastasis 2024**: available from the [ASNR-MICCAI BraTS 2024 challenge](https://www.synapse.org/brats2024). We use dataset ID `111`.
- **SBT**: a private clinical dataset from the Federal Neurosurgical Center, Novosibirsk. It contains patient data and is not publicly available. We use dataset ID `222`.

Both datasets need the four sequences (T1W, T1C, T2W, FLAIR) in the
[nnU-Net raw format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).
They use labels 1 = necrotic / non-enhancing core, 2 = edema and 3 = enhancing tumor.
Define the ET, TC and WT regions in `dataset.json`
([region-based training](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/region_based_training.md)).

## Usage

### Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

The trainers set `num_epochs = 100`. For BraTS pretraining we used 200 epochs, so change `self.num_epochs` in the trainer before you pretrain.

```bash
# From scratch (or pretraining on BraTS)
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TRAINER_NAME

# Fine-tuning from pretrained weights
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr TRAINER_NAME -pretrained_weights /path/to/checkpoint_final.pth
```

nnU-Net requires the source and target datasets to share the same plans before fine-tuning. See
[pretraining and fine-tuning](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md).

### Pretrained weights

Pretrained weights: TBA.

### Inference

Put the trained models in `$nnUNet_results` and predict with the fold ensemble:

```bash
nnUNetv2_predict \
    -i /path/to/input_images \
    -o /path/to/predictions \
    -d DATASET_ID \
    -c 3d_fullres \
    -f 0 1 2 3 4 \
    -tr TRAINER_NAME
```

### Evaluation

This script computes DSC, HD95, sensitivity and specificity for ET, TC and WT. A prediction and its ground truth must have the same filename.

```bash
python meta24_compute_metrics.py --pr /path/to/predictions --gt /path/to/ground_truth --out scores.csv
```

To run the paired statistical tests on the per-case scores in `scores/`:

```bash
python utils/statistical_tests_2026-09-29.py --scores-dir scores --suffix _2026-09-29
```

### Additional utilities

These tools are for longitudinal analysis of binary tumor masks. They are not part of the paper.

| Script | Wrapper | Purpose |
| --- | --- | --- |
| [`utils/check_tumor_data.py`](utils/check_tumor_data.py) | `scripts/run_check_tumor_data.sh` | Data-quality report for dated masks (shapes, registration, labels, gaps between visits) |
| [`utils/tumor_tracking.py`](utils/tumor_tracking.py) | `scripts/run_tumor_tracking.sh` | Matches tumor instances across timepoints and tracks their volume |
| [`utils/plot_tumor_volume.py`](utils/plot_tumor_volume.py) | — | Volume-over-time plots with nadir and progression threshold |
| [`utils/compute_seg_metrics.py`](utils/compute_seg_metrics.py) | `scripts/run_compute_seg_metrics.sh` | Binary Dice, HD95, volume and instance counts |

## Repository structure

```
trainers/                  custom nnU-Net trainers (copy into nnunetv2/training/nnUNetTrainer/)
meta24_compute_metrics.py  ET/TC/WT segmentation metrics
scores/                    per-case scores on the SBT test set and statistical test results
notebooks/                 result analysis and statistical tests
utils/                     statistical tests and longitudinal tracking tools
scripts/                   environment setup and wrapper scripts
img/                       figures
```

## Citation

If you use this code, please cite:

```bibtex
@article{luu_transfer_2024,
  title   = {Transfer Learning Approaches for Brain Metastases Screenings},
  author  = {Luu, Minh Sao Khue and Tuchinov, Bair N. and Suvorov, Victor and Kenzhin, Roman M. and Amelina, Evgeniya V. and Letyagin, Andrey Yu.},
  journal = {Biomedicines},
  volume  = {12},
  number  = {11},
  pages   = {2561},
  year    = {2024},
  doi     = {10.3390/biomedicines12112561},
  url     = {https://www.mdpi.com/2227-9059/12/11/2561}
}
```

This work builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) (Isensee et al., *Nature Methods*, 2021)
and [MONAI](https://monai.io/). Please cite them as well.

## License

The code in this repository is released under the [MIT License](LICENSE).
nnU-Net is licensed separately under Apache 2.0. The BraTS data are subject to the challenge's terms of use.