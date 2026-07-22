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
source scriprs/setvars.sh
```

Or

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

- To install `mamba`, run:

```bash
pip install causal-conv1d>=1.2.0 and pip install mamba-ssm --no-cache-dir
```

- To install `nnUNetv2` locally, run:

```bash
cd nnunetv2
pip install -e
```

