#!/bin/bash

# Base paths
VIDEO_FOLDER="${VIDEO_FOLDER:-/idiap/temp/mvillamizar/local/projects/humantracker/data/chuv/dataset/videos/}"
PROC_FOLDER="${PROC_FOLDER:-/idiap/temp/mvillamizar/local/projects/humantracker/data/chuv/dataset/processed_validated/}"

# Configuration
INDIVIDUAL_ID="${INDIVIDUAL_ID:-14}"
SESSION_ID="${SESSION_ID:-7}"
VIDEO_SUBDIR="${VIDEO_SUBDIR:-1000_frames}"
VIDEO_FILE="${VIDEO_FILE:-output_crf24_a_processed_1000frames.mp4}"
ELAN_FILE="${ELAN_FILE:-11-1-2024_#7_INDIVIDUAL_(14)_WAKEE_16.10.25_BL.txt}"

# Paths
BASE="$VIDEO_FOLDER/${INDIVIDUAL_ID}_INDIVIDUAL_${INDIVIDUAL_ID}"
ELAN="$BASE/$ELAN_FILE"
VIDEO="$BASE/$VIDEO_SUBDIR/$VIDEO_FILE"
OUT="$BASE/Annotation/annotations_csv"
META="$BASE/processing_log.json"

# Create output directory
mkdir -p "$OUT"

# Run conversion
python elan_to_csv_converter.py \
  --elan_txt "$ELAN" \
  --video "$VIDEO" \
  --sam3_metadata "$META" \
  --individual_id "$INDIVIDUAL_ID" \
  --session_id "$SESSION_ID" \
  --output "$OUT/ind_${INDIVIDUAL_ID}_sess_${SESSION_ID}.csv" && echo "Done"
