#!/bin/bash

# Define paths
SOURCE_PATH="/home/s124z/Code/YOLO/heico_dataset/test"
DEST_PATH="s124z@odcf-worker02:/dkfz/cluster/gpu/data/OE0176/s124z/HEICO_dataset"

# Check if source directory exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: Source directory does not exist: $SOURCE_PATH"
    exit 1
fi

# Sync data to cluster with progress and human-readable sizes
rsync -avh --inplace \
    --progress \
    "$SOURCE_PATH" \
    "$DEST_PATH"

# Check rsync exit status
if [ $? -eq 0 ]; then
    echo "Data sync completed successfully"
else
    echo "Error: Data sync failed"
    exit 1
