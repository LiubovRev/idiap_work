#!/usr/bin/env bash

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

VID_FOLDER="/idiap/temp/lrevutska/project/videos/head_detection_test/"
OUT_FOLDER="/idiap/temp/lrevutska/project/videos/output/head_detection_sam3/"
MODEL_FILE="/idiap/temp/mvillamizar/local/projects/humantracker/models/head_detection/nano.pt"

MAX_FRAMES=100000
CONF=0.7
IOU=0.2
DEVICE="cpu"

python scripts/demo_head_detection.py \
    --video_folder "$VID_FOLDER" \
    --output_folder "$OUT_FOLDER" \
    --model_file "$MODEL_FILE" \
    --confidence "$CONF" \
    --iou "$IOU" \
    --max_frames "$MAX_FRAMES" \
    --device "$DEVICE"
