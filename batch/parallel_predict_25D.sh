#!/bin/bash

# script to run predict.py in parallel on multiple GPUs
# running this script from the flairsyn directory will run predict.py on all volumes
# note that all args are forward through this script

# Define endpoint script
ENDPOINT_SCRIPT="predict/25d_yoda_predict.py"

# Default values for start and end indices
START_INDEX=0
END_INDEX=42

echo "Running predict.py on volumes $START_INDEX to $END_INDEX"

# Query the number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Forward the args
ADDITIONAL_ARGS="$@"

# Calculate the number of volumes per GPU
NUM_VOLUMES=$((END_INDEX - START_INDEX))
VOLUMES_PER_GPU=$((NUM_VOLUMES / NUM_GPUS))

# Launch the scripts
for (( GPU=0; GPU<NUM_GPUS; GPU++ ))
do
    GPU_START=$((START_INDEX + GPU * VOLUMES_PER_GPU))
    GPU_END=$((GPU_START + VOLUMES_PER_GPU))

    # Adjust the last GPU end index to cover all volumes
    if [ $GPU -eq $((NUM_GPUS - 1)) ]; then
        GPU_END=$END_INDEX
    fi

    echo "Launching on GPU $GPU: Volumes $GPU_START to $GPU_END"
    CUDA_VISIBLE_DEVICES=$GPU python $ENDPOINT_SCRIPT $ADDITIONAL_ARGS \
           --start=$GPU_START --end=$GPU_END &

done

wait
echo "All processes completed."
