#!/bin/bash

BASE="/home/lrevutska/Documents/chuv_machine_downloads/videos/7_INDIVIDUAL_14"

elan="$BASE/11-1-2024_#7_INDIVIDUAL_(14)_WAKEE_16.10.25_BL.txt"
video="$BASE/1000_frames/output_crf24_a_processed_1000frames.mp4"
out="$BASE/Annotation/annotations_csv"

mkdir -p "$out"

python elan_to_csv_converter.py \
  --elan_txt "$elan" \
  --video "$video" \
  --individual_id 14 --session_id 7 \
  --output "$out/ind_14_sess_7.csv" && echo "✓ Done"
