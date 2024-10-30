import os
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

# Set nnUNet environment variables
os.environ["nnUNet_raw"] = "data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "data/nnUNet_results"

print("nnUNet_raw:", os.environ.get("nnUNet_raw"))
print("nnUNet_preprocessed:", os.environ.get("nnUNet_preprocessed"))
print("nnUNet_results:", os.environ.get("nnUNet_results"))

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
        os.remove(path)

@app.post("/segment/")
async def segment(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    # Check that exactly four files are uploaded
    if len(files) != 4:
        return JSONResponse(
            content={"error": "Exactly four files are required, in the order: T1C, T1N, T2F, T2W."},
            status_code=400
        )

    # Create a temporary directory for input files
    with tempfile.TemporaryDirectory() as temp_input_dir:
        temp_input_paths = []

        # Save each uploaded file in the temporary directory
        for file in files:
            file_path = os.path.join(temp_input_dir, file.filename)
            with open(file_path, "wb") as temp_file:
                temp_file.write(await file.read())
            temp_input_paths.append(file_path)

        output_dir = os.path.join(os.environ.get("nnUNet_results"), "Dataset111_Meta", "nnUNetTrainer_TverskyBCE__nnUNetPlans__3d_fullres")
        command = [
            "nnUNetv2_predict",
            "-i", temp_input_dir,
            "-o", output_dir,
            "-d", "111",  # dataset id
            "-f", "0",  # fold 0
            "-c", "3d_fullres",  # configuration
            "-tr", "nnUNetTrainer_TverskyBCE",  # trainer
            "-chk", "checkpoint_final.pth"
        ]
        print(command)

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)

        # Locate the output segmentation file in the output directory
        output_file_path = os.path.join(output_dir, os.listdir(output_dir)[0])
        print('output file path:', output_file_path)

        # Return the segmented NIfTI file as the response
        response = FileResponse(output_file_path, media_type="application/gzip", filename="segmentation_output.nii.gz")

        # # Add cleanup task to delete temporary input files after response
        # background_tasks.add_task(cleanup_temp_files, temp_input_paths)
        # background_tasks.add_task(cleanup_temp_files, [output_file_path])

        return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
