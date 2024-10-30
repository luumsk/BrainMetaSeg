
import os
import tempfile

import numpy as np
import nibabel as nib
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data  # nnUNetv2 prediction utility
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results, nnUNet_raw


app = FastAPI()

model_name = "3d_fullres"
trainer_class_name = "nnUNetTrainer_TverskyBCE"


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Welcome to the Brain Metastases MRI Segmentation API."
        }
    )


@app.post("/segment/")
async def segment(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_input_file:
        temp_input_file.write(await file.read())
        temp_input_path = temp_input_file.name

    # Prepare output directory for prediction results
    output_folder = tempfile.mkdtemp()

    # Run the prediction using nnUNetv2
    predict_from_raw_data(
        [(temp_input_path,)],  # List of input files
        output_folder,
        model_name=model_name,
        trainer_class_name=trainer_class_name,
        plans_identifier="nnUNetPlans", 
        checkpoint_name="checkpoint_final.pth",
        configuration="3d_fullres",
        use_folds=(0,),
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    # Locate the output segmentation file
    output_file_path = os.path.join(output_folder, os.listdir(output_folder)[0])

    # Return the segmented NIfTI file and clean up files after response
    response = FileResponse(output_file_path, media_type="application/gzip", filename="segmentation_output.nii.gz")
    response.call_on_close(lambda: os.remove(output_file_path) or os.rmdir(output_folder) or os.remove(temp_input_path))
    return response

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)