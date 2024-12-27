# BrainMetaSeg: Transfer Learning Approaches for Brain Metastases Screenings

This repository contains the code for our research on **Transfer Learning Approaches for Brain Metastases Screenings**. This work explores different models and transfer learning strategies for brain metastasis segmentation using MRI data. We utilize the BraTS Metastases 2024 dataset and a private dataset for model training and evaluation.

**Research Paper**: [Transfer Learning Approaches for Brain Metastases Screenings](https://www.mdpi.com/2227-9059/12/11/2561)

## Installation

1. Clone only this branch of the repository:
   ```bash
   git clone --branch metastases2024 --single-branch git@bigdata.nsu.ru:medical/BraTS23.git
   ```

2. Change directory to the repo
   ```bash
   cd BraTS23
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install `nnUNetv2` from source:
   ```bash
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
   ```

5. Copy the custom trainer files fron `trainers` folder to the `nnunetv2/training/` folder:
   ```bash
   cp -r trainers/* nnunetv2/training/nnUNetTrainer/
   ```

## Usage

To train, run inference, or evaluate the model on MRI data, follow the steps below:

### Preprocessing

In our study, we select `111` as DATASET_ID for BraTS 2024 dataset, and `222` as DATASET_ID for our private SBT dataset.

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

```bash
nnUNetv2_train DATASET_ID CONGIFURATION FOLD -tr TRAINER_NAME -pretrained_weights <path_to_pretrained_weights>
```

If you have multiple CUDA devices, you can specify which device is being used for this training process with CUDA_VISIBLE_DEVICES=<cuda_id>.

In this example, we use CUDA device id `0`, DATASET_ID `111`, configuration `3d_fullres`, fold `1`, trainer `nnUNetTrainer_TverskyBCE`, path to pretrained weights is `/media/storage/luu/nnUNet_results/Dataset111_Meta/nnUNetTrainer_TverskyBCE__nnUNetPlans_aligned__3d_fullres/fold_all/checkpoint_final.pth`

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train \
    111 3d_fullres 1 \
    -tr nnUNetTrainer_TverskyBCE \
    -pretrained_weights /media/storage/luu/nnUNet_results/Dataset111_Meta/nnUNetTrainer_TverskyBCE__nnUNetPlans_aligned__3d_fullres/fold_all/checkpoint_final.pth
```

### Inference

1. Download checkpoints and saved in foler `nnUNet_results`
2. Run the command:

   ```bash
      nnUNetv2_predict \
         -i <directory_of_input_images> \
         -o <directory_of_output_predictions> \
         -d DATASET_ID \
         -c CONFIGURATION \
         -f FOLD \
         -tr TRAINER_NAME \
         --disable_progress_bar \
         --save_probabilities
   ```

   In this example, we train DATASET_ID `111`, configuration `3d_fullres`, fold `1`, trainer `nnUNetTrainerSegResNet`

   ```bash
   nnUNetv2_predict \
      -i /media/storage/luu/nnUNet_raw/Dataset222_SBT/imagesTs \
      -o /media/storage/luu/nnUNet_predictions/Dataset222_SBT/segresnet_finetune_preds/fold_1 \
      -d 111 \
      -c 3d_fullres \
      -f 1 \
      -tr nnUNetTrainerSegResNet \
      --disable_progress_bar \
      --save_probabilities
   ```

### Evaluation

To get evaluation scores, run the file `meta24_compute_metrics.py`

```bash
python -m meta24_compute_metrics.py --pr <directory_of_predictions> --gt <directory_of_ground_truths> --out <path_to_output_csv_file>
```

## Contact

For questions or collaborations, contact [khue.luu@g.nsu.ru](mailto:khue.luu@g.nsu.ru).
