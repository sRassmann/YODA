#!/bin/bash

# Parse input arguments
dir_a=${1:-"a"}       # Default to "a" if not provided
dir_b=${2:-"b"}       # Default to "b" if not provided
file_a=${3:-"flair.nii.gz"}  # Default to "flair.nii.gz" if not provided
file_b=${4:-"$file_a"}  # Default to $fila_a if not provided

# Loop through the subdirectories in dir_a
for subdir in "$dir_a"/*; do
  # Extract the subdirectory name
  sub_name=$(basename "$subdir")

  # Check if the sequence file exists in subdirectory of dir_a
  seq_file="$subdir/$file_a"

  # Create corresponding subdirectory in dir_b if it doesn't exist
  mkdir -p "$dir_b/$sub_name"

  # Create symlink for the file in the corresponding subdirectory of dir_b
  ln -s "$(realpath "$seq_file")" "$dir_b/$sub_name/$file_b"

  echo "Symlink created for $seq_file in $dir_b/$sub_name/"

done
