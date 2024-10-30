import os
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

os.environ["nnUNet_raw"] = "data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "data/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "data/nnUNet_results"

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

@app.post("/segment/")
async def segment(files: list[UploadFile] = File(...)):
    # Check that exactly four files are uploaded
    if len(files) != 4:
        return JSONResponse(
            content={"error": "Exactly four files are required, in the order: T1C, T1N, T2F, T2W."},
            status_code=400
        )

    print('Saving input files...')
    # Save each uploaded file temporarily
    temp_input_paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_input_file:
            temp_input_file.write(await file.read())
            temp_input_paths.append(temp_input_file.name)

    print('Creating temp output folder...')
    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as temp_output_dir:
        command = [
            "nnUNetv2_predict",
            "-i", ",".join(temp_input_paths),
            "-o", temp_output_dir,
            "-d", "111", # dataset id
            "-f", "0", # fold 0
            "-c", "3d_fullres", # configuration,
            "-tr", "nnUNetTrainer_TverskyBCE", # trainer
            "-chk", "checkpoint_final.pth"
        ]
        print(command)

        # Execute the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)

        # Locate the output segmentation file in the temporary output directory
        output_file_path = os.path.join(temp_output_dir, os.listdir(temp_output_dir)[0])

        # Return the segmented NIfTI file as the response
        response = FileResponse(output_file_path, media_type="application/gzip", filename="segmentation_output.nii.gz")

        # Clean up input files after response is sent
        response.call_on_close(lambda: [os.remove(path) for path in temp_input_paths])
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
