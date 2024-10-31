import os
import glob
import torch
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


app = FastAPI()

model_folder = "data/nnUNet_results/Dataset111_Meta/nnUNetTrainer_TverskyBCE__nnUNetPlans__3d_fullres"
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True if torch.cuda.is_available() else False,
        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name='checkpoint_final.pth')

@app.get("/", status_code=200)
async def root():
    return JSONResponse(
        content={
            "message": "Welcome to the Brain Metastases MRI Segmentation API.",
            "status": "Server is running",
            "instructions": {
                "upload_endpoint": "/segment/",
                "method": "POST",
                "description": "Submit four .nii.gz files of 4 modalities as inputs for segmentation. Files should be placed in this order (T1C, T1N, T2F, T2W)",
            }
        }
    )

def cleanup_temp_files(dir):
    for fn in os.listdir(dir):
        path = os.path.join(dir, fn)
        if os.path.exists(path):
            os.remove(path)

@app.post("/segment/")
async def segment(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if len(files) != 4:
        return JSONResponse(
            content={"error": "Exactly four files are required, in the order: T1C, T1N, T2F, T2W."},
            status_code=400
        )

    # Define temporary directories
    temp_input_dir = os.environ.get("nnUNet_raw")
    temp_output_dir = os.environ.get("nnUNet_output")

    # Save uploaded .nii.gz files directly to the temporary input directory
    for file in files:
        file_path = os.path.join(temp_input_dir, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(await file.read())

    # Run prediction using the predictor instance
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=temp_input_dir,
            output_folder_or_list_of_truncated_output_files=temp_output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
    except Exception as e:
        return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)

    # Find the output .nii.gz file
    output_file_path = next(glob.iglob(os.path.join(temp_output_dir, "*.nii.gz")), None)
    if not output_file_path or not os.path.isfile(output_file_path):
        return JSONResponse(content={"error": "Prediction completed, but no output file was generated."}, status_code=500)

    # Return the segmented NIfTI file as the response
    response = FileResponse(output_file_path, media_type="application/nii.gz", filename="segmentation_output.nii.gz")

    # Schedule cleanup for temporary files
    background_tasks.add_task(cleanup_temp_files, temp_input_dir)
    background_tasks.add_task(cleanup_temp_files, temp_output_dir)

    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
