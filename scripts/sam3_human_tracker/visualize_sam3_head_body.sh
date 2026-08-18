#!/bin/bash

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

python scripts/visualize_sam3_head_body.py \
    --video /idiap/temp/lrevutska/project/videos/head_detection_test/video_processed_a_test.mp4 \
    --csv /idiap/temp/lrevutska/project/videos/sam3_head_body_detections.csv \
    --output /idiap/temp/lrevutska/project/videos/head_detection_test/sam3_head_body_visualization_test.mp4
