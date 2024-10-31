FROM python:3.10

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 torchvision==0.15.2 monai==1.3.0 nibabel==5.2.1 numpy==1.26.4 python-multipart fastapi uvicorn
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /nnUNet
RUN pip install -e .

COPY trainers/nnUNetTrainer_TverskyBCE.py /nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_TverskyBCE.py
COPY app.py /app.py
COPY data/ /data/

ENV nnUNet_raw="data/nnUNet_raw"
ENV nnUNet_preprocessed="data/nnUNet_preprocessed"
ENV nnUNet_results="data/nnUNet_results"
ENV nnUNet_output="data/nnUNet_output"

WORKDIR /
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
