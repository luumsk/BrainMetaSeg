# DEVELOPER GUIDE

To run models of this research on Tunka (a remote server) for training and predicting, to the following steps:

- To install virtual environment with `Python 3.10` on Tunka with no sudo, run:

```bash
/home/luu/.pyenv/versions/3.10.13/bin/python3.10 -m venv <dir_to_venv>
```

For example:

```bash
/home/luu/.pyenv/versions/3.10.13/bin/python3.10 -m venv brainmetaseg_venv
```

- To activate virtual environment and export nnUNet paths, run:

```bash
source /path/to/venv/bin/activate
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
export PATH=$PATH:/usr/bin
```

- To install `pytorch2.0.1` with cuda, run:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

- To install `mamba` (only needed for training/inference with the UMamba trainers), run:

```bash
pip install "causal-conv1d>=1.2.0" --no-build-isolation
pip install mamba-ssm --no-cache-dir
```

To install `nnUNetv2` locally, run this from the repo root (NOT from inside `nnunetv2/` - `setup.py` lives at the repo root since `nnunetv2/` itself is the package directory):

For example:

```bash
pip install -e .
```

- To run models using custom trainer, copy the trainer files in `trainers` folder to `nnunetv2/training/nnUNetTrainer`. For example, to use `nnUNetTrainer_TverskyBCE` trainer, run:

```bash
cp trainers/nnUNetTrainer_TverskyBCE.py nnunetv2/training/nnUNetTrainer/
```

## To run inference

- Require the same set of modalities, for example: T1, T1W, T2, FLAIR.
- No need to run preprocessing script.
- 
