# BrainMetaSeg

This repository contains the code and resources for the research project on **Brain Metastasis Segmentation** using deep learning techniques. The focus of the project is on improving the segmentation of brain metastases using a variety of medical imaging datasets, particularly leveraging models fine-tuned with limited annotated data.

## Project Overview

Brain metastasis is a critical condition that requires precise detection and segmentation from medical images such as MRI. Accurate segmentation is crucial for effective treatment planning. This project aims to address the challenges of segmenting brain metastases by using advanced 3D convolutional neural networks (CNNs) and techniques that minimize the need for extensive labeled datasets.


## Features
- **Brain Metastasis Segmentation**: Automated segmentation of brain metastases from MRI volumes.
- **3D UNet**: A robust architecture for handling 3D medical imaging data.
- **Performance Metrics**: High accuracy measured with the Dice similarity coefficient.
- **Minimal Labeled Data**: Use of transfer learning to reduce the dependency on large annotated datasets.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luumsk/BrainMetaSeg.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model or run inference on new MRI data, follow the steps below:

### Training
To train the model on the dataset:
```bash
python train.py --dataset <path-to-dataset>
```

### Inference
To run the model for inference on new MRI scans:
```bash
python inference.py --input <path-to-input-image>
```

### Evaluation
To evaluate the model on the test dataset:
```bash
python evaluate.py --model <path-to-model> --testdata <path-to-test-data>
```

## Contact

For questions or collaborations, contact [khue.luu@g.nsu.ru](mailto:khue.luu@g.nsu.ru).
```
