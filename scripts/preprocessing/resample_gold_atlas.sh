#!/bin/bash

# Set base directories
input_dir=""  # define
output_dir=""  # define

# Loop through all subject folders (e.g., 1_01_P, 1_02_P, etc.)
for subject in ${input_dir}/*_P; do
    # Extract subject ID from path
    subject_id=$(basename "$subject")

    # Create output directory for the subject if it doesn't exist
    mkdir -p "${output_dir}/${subject_id}"

    # link ct file
    ln -sf "${subject}/ct.nii.gz" "${output_dir}/${subject_id}/ct.nii.gz"

    # T1 processing
    t1_file=$(find "$subject" -type f -name '*_t1_*.nii.gz' | head -n 1)
    if [ -n "$t1_file" ]; then
        echo "Processing T1 for ${subject_id}..."
        mri_vol2vol --mov "$t1_file" \
            --targ "${subject}/ct.nii.gz" \
            --o "${output_dir}/${subject_id}/t1.nii.gz" \
            --regheader --no-save-reg
    else
        echo "No T1 file found for ${subject_id}"
    fi

    # T2 processing
    t2_file=$(find "$subject" -type f -name '*_t2_*.nii.gz' | head -n 1)
    if [ -n "$t2_file" ]; then
        echo "Processing T2 for ${subject_id}..."
        mri_vol2vol --mov "$t2_file" \
            --targ "${subject}/ct.nii.gz" \
            --o "${output_dir}/${subject_id}/t2.nii.gz" \
            --regheader --no-save-reg
    else
        echo "No T2 file found for ${subject_id}"
    fi

done
