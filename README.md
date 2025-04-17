# BrainMetaSeg: Transfer Learning Approaches for Brain Metastases Screenings

This repository contains the code for our research on **Transfer Learning Approaches for Brain Metastases Screenings**. This work explores different models and transfer learning strategies for brain metastasis segmentation using MRI data. We utilize the BraTS Metastases 2024 dataset and a private dataset for model training and evaluation.

**Research Paper**: [Transfer Learning Approaches for Brain Metastases Screenings](https://www.mdpi.com/2227-9059/12/11/2561)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luumsk/BrainMetaSeg.git
   ```

2. Change directory to the repo
   ```bash
   cd BrainMetaSeg
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

6. Set up paths to nnUNet folders in the `scripts/setvars.sh` and execute this script
   ```bash
   source scripts/setvars.sh
   ```

## Usage

To train, run inference, or evaluate the model on MRI data, follow the steps below:

### Preprocessing

In our study, we select `111` as DATASET_ID for BraTS 2024 dataset, and `222` as DATASET_ID for our private SBT dataset.

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

**Train from scratch**

```bash
nnUNetv2_train DATASET_ID CONGIFURATION FOLD -tr TRAINER_NAME
```

**Fine-tuning**

nnUNet requires aligned datasets fingerprints before fine-tuning. See intructions [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md).
```bash
nnUNetv2_train DATASET_ID CONGIFURATION FOLD -tr TRAINER_NAME -pretrained_weights <path_to_pretrained_weights>
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

### Evaluation

To get evaluation scores, run the file `meta24_compute_metrics.py`

```bash
python -m meta24_compute_metrics.py --pr <directory_of_predictions> --gt <directory_of_ground_truths> --out <path_to_output_csv_file>
```

## Contact

For questions or collaborations, contact [khue.luu@g.nsu.ru](mailto:khue.luu@g.nsu.ru).
