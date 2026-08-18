#!/bin/bash

VIDEO="/idiap/temp/lrevutska/project/videos/video_processed_a.mp4"
CSV="/idiap/temp/lrevutska/project/videos/sam3_body_detections.csv"
OUTPUT="/idiap/temp/lrevutska/project/videos/video_processed_a_sam3_boxes.mp4"

python scripts/visualize_sam3_boxes.py \
    --video "$VIDEO" \
    --csv "$CSV" \
    --output "$OUTPUT"
