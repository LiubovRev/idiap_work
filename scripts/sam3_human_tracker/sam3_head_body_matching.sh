#!/bin/bash

BODY_CSV="/idiap/temp/lrevutska/project/videos/sam3_body_detections.csv"

HEAD_CSV="/idiap/temp/lrevutska/project/videos/output/head_detection_sam3/video_processed_a_test.csv"

OUTPUT="/idiap/temp/lrevutska/project/videos/sam3_head_body_detections.csv"

python scripts/sam3_head_body_matching.py \
    --body_csv "$BODY_CSV" \
    --head_csv "$HEAD_CSV" \
    --output "$OUTPUT"
