#!/usr/bin/env python3

# TODO: change the code to work with multiple people

TRACK_FILE="./data/output_crf24_a_processed/extracted_data.csv"
# TRACK_FILE="./data/output_crf24_a_processed/extracted_data_edited.csv"
VIDEO_FILE="./data/output_crf24_a_processed/output_crf24_a_processed.mp4"

python track_editor.py $VIDEO_FILE $TRACK_FILE -o tracks_fixed.csv --log tracks_fixed_log.csv
