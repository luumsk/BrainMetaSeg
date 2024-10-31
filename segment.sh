curl -X POST "http://localhost:8000/segment/" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/home/luu/BrainMetaSeg/BraTS-MET-00920-000_0000.nii.gz;type=application/nii.gz" \
  -F "files=@/home/luu/BrainMetaSeg/BraTS-MET-00920-000_0001.nii.gz;type=application/nii.gz" \
  -F "files=@/home/luu/BrainMetaSeg/BraTS-MET-00920-000_0002.nii.gz;type=application/nii.gz" \
  -F "files=@/home/luu/BrainMetaSeg/BraTS-MET-00920-000_0003.nii.gz;type=application/nii.gz" \
  -o segmentation_output.nii.gz
