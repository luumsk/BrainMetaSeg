#!/bin/bash

INPUT_DIR=/media/storage/luu/BrainTumorTracking/nifti_native
PREPROCESSED_DIR=/media/storage/luu/BrainTumorTracking/prepared_for_nnunet

# Create absolute paths for safety
INPUT_DIR=$(realpath "$INPUT_DIR")
PREPROCESSED_DIR=$(realpath "$PREPROCESSED_DIR")

echo "=== Step 1: Preparing Target Directory ==="
mkdir -p "$PREPROCESSED_DIR"
echo "Target preprocessed directory verified at: $PREPROCESSED_DIR"

echo "=== Step 2: Duplicating Channels ==="
COUNTER=0

# Loop through files looking for flair_YYYY_MM patterns
for filepath in "$INPUT_DIR"/flair_[0-9][0-9][0-9][0-9]_[0-9][0-9].nii.gz; do
    # Check if any matching files exist
    [ -e "$filepath" ] || continue
    
    filename=$(basename "$filepath")
    # Extract id (e.g., flair_2016_11)
    case_id="${filename%.nii.gz}"
    
    echo "Processing case ID: $case_id"
    
    # Duplicate FLAIR directly into your permanent preprocessed storage folder
    cp "$filepath" "$PREPROCESSED_DIR/${case_id}_0000.nii.gz" # T1C
    cp "$filepath" "$PREPROCESSED_DIR/${case_id}_0001.nii.gz" # T1N
    cp "$filepath" "$PREPROCESSED_DIR/${case_id}_0002.nii.gz" # T2F
    cp "$filepath" "$PREPROCESSED_DIR/${case_id}_0003.nii.gz" # T2W
    
    ((COUNTER++))
done

# Ensure at least one file was processed
if [ "$COUNTER" -eq 0 ]; then
    echo "Error: No matching 'flair_YYYY_MM.nii.gz' files found in $INPUT_DIR"
    exit 1
fi

echo "=== Success ==="
echo "Successfully prepared $COUNTER cases into: $PREPROCESSED_DIR"
