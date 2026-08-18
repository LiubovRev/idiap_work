#!/bin/bash

MASK_DIR="/idiap/temp/lrevutska/project/videos/MaskDir"
OUTPUT="/idiap/temp/lrevutska/project/videos/sam3_body_detections.csv"

python scripts/sam3_masks_to_body_csv.py \
    --mask_dir "$MASK_DIR" \
    --output "$OUTPUT"
