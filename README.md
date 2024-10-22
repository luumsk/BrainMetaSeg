# BrainMetaSeg: Transfer Learning Approaches for Brain Metastases Screenings

This repository contains the code for our research on **Transfer Learning Approaches for Brain Metastases Screenings**. This work explores different models and transfer learning strategies for brain metastasis segmentation using MRI data. We utilize the BraTS Metastases 2024 dataset and a private dataset for model training and evaluation.

**Research Paper**: [Transfer Learning Approaches for Brain Metastases Screenings](#) _(Link TBA)_

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luumsk/BrainMetaSeg.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `nnUNetv2` from source:
   ```bash
   git clone https://github.com/MIC-DKFZ/nnUNet.git
   cd nnUNet
   pip install -e .
   ```

4. Copy the custom trainer files to the `nnunetv2/training/` directory:
   ```bash
   cp -r BrainMetaSeg/trainers/* nnunetv2/training/nnUNetTrainer/
   ```

## Usage

To train, run inference, or evaluate the model on MRI data, follow the steps below:

### Training

TBA

### Inference

TBA

### Evaluation

TBA

## Contact

For questions or collaborations, contact [khue.luu@g.nsu.ru](mailto:khue.luu@g.nsu.ru).
