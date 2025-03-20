#!/bin/bash

# Set the path to the conformed folder and target folder from command line arguments
CONFORMED_FOLDER=$1
TARGET_FOLDER=$2
mkdir -p $TARGET_FOLDER

# Optional named argument (--reference_file) for reference file, default FLAIR.nii.gz
REFERENCE_FILE=${3:-"FLAIR.nii.gz"}

# Set the number of subjects to process. If not provided, process all subjects.
NUM_SUBJECTS=${4:-$(ls -1q $CONFORMED_FOLDER | wc -l)}

# Get the number of GPUs
NUM_GPUS=8
GPU_IDX=0

echo "Processing $NUM_SUBJECTS subjects using $NUM_GPUS GPUs"
echo "Reference file: $REFERENCE_FILE"

# Initialize a counter
COUNTER=0

for entry in $CONFORMED_FOLDER/*; do
    # If the counter reaches the limit, break the loop
    if [ $COUNTER -eq $NUM_SUBJECTS ]; then
        break
    fi

    # Extract the last part of the path as the subject
    SUBJECT=$(basename "$entry")
    echo "Processing $SUBJECT on GPU $GPU_IDX"

    # Find T1
    T1=$(find "$entry"/T1* | sort | head -n 1)
    # T1="T1_2.nii.gz"  # in case multiple T1s are present!
    T1_ARG="--t1 /data/$SUBJECT/$(basename "$T1")"

    echo $GPU_IDX
    # Check if output exists
    if [ -d "$TARGET_FOLDER/$SUBJECT/stats" ]; then
        echo "Output already exists for $SUBJECT"
    else
        # Run the Docker command for each subject
        SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU_IDX singularity exec --nv --no-home \
            -B $CONFORMED_FOLDER:/data \
            -B $TARGET_FOLDER:/output \
            $HOME/singularity/fastsurfer-gpu.sif /fastsurfer/run_fastsurfer.sh \
            $T1_ARG --sid "$SUBJECT" --sd /output \
            --parallel --threads 16 --seg_only --no_cereb &
    fi

    # Conform the mask after segmentation
    python scripts/preprocessing/conform_brain_mask.py $TARGET_FOLDER \
        $CONFORMED_FOLDER $SUBJECT --reference $REFERENCE_FILE &

    echo "Processing started for $SUBJECT on GPU $GPU_IDX"

    # Increment the counter
    ((COUNTER++))

    # Cycle through GPUs in round-robin fashion
    GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

    # Control parallel jobs (adjust if needed)
    if (( COUNTER % NUM_GPUS == 0 )); then
        wait  # Wait for all current jobs to finish before launching new ones
    fi
done

wait  # Ensure all jobs complete before the script exits
echo "All subjects processed."