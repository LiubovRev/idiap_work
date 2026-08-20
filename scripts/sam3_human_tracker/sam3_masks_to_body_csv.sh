#!/bin/bash

MASK_DIR="/home/liubov/Documents/psifx_sam3/videos/7_INDIVIDUAL_14/full_video/MaskDir_manual_merged"
OUTPUT="/home/liubov/Documents/psifx_sam3/videos/7_INDIVIDUAL_14/full_video/sam3_body_detections.csv"

python scripts/sam3_masks_to_body_csv.py \
    --mask_dir "$MASK_DIR" \
    --output "$OUTPUT" \
    --min_area 1000 \
    --border_margin 5
