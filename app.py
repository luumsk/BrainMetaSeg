import os
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# Set nnUNet environment variables
os.environ["nnUNet_raw"] = "data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "data/nnUNet_results"
os.environ["nnUNet_output"] = "data/nnUNet_output"

app = FastAPI()

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

def cleanup_temp_files(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)

@app.post("/segment/")
async def segment(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if len(files) != 4:
        return JSONResponse(
            content={"error": "Exactly four files are required, in the order: T1C, T1N, T2F, T2W."},
            status_code=400
        )

    temp_input_dir = os.environ.get("nnUNet_raw")
    temp_input_paths = []
    for file in files:
            file_path = os.path.join(temp_input_dir, file.filename)
            with open(file_path, "wb") as temp_file:
                temp_file.write(await file.read())
            temp_input_paths.append(file_path)

    temp_output_dir = os.environ.get("nnUNet_output")
    command = [
        "nnUNetv2_predict",
        "-i", temp_input_dir,
        "-o", temp_output_dir,
        "-d", "111",  # dataset id
        "-f", "0",  # fold 0
        "-c", "3d_fullres",  # configuration
        "-tr", "nnUNetTrainer_TverskyBCE",  # trainer
        "-chk", "checkpoint_final.pth",
        "--disable_progress_bar"
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)

    # Locate the output segmentation file in the output directory
    temp_output_paths = [os.path.join(temp_output_dir, fn) for fn in os.listdir(temp_output_dir)]
    output_file_path = [pth for pth in temp_output_paths if pth.endswith('.nii.gz')][0]

    # Return the segmented NIfTI file as the response
    response = FileResponse(output_file_path, media_type="application/gzip", filename="segmentation_output.nii.gz")

    # Schedule cleanup for only the temporary input files
    background_tasks.add_task(cleanup_temp_files, temp_input_paths)
    background_tasks.add_task(cleanup_temp_files, temp_output_paths)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
