# CHUV-ADHDArtTherapy Data Architecture & Processing Pipeline

## 1. Directory Structure

The repository/server storage is divided into raw immutable data (Naomi Archive) and derived features/outputs.

```text
ROOT_Directory_Raw/                           # [READ-ONLY] Naomi archive folder structure
└── SESSIONS/
    └── <day>-<month>-<year>_#<session_number>_<TYPE>_[<child_ids>]/  # e.g. 10-1-2024_#6_INDIVIDUAL_[15]
        ├── raw_video.mkv (or .mp4)           # Video recording
        ├── audio_track.wav                   # Separate audio (iPhone)
        ├── depth_data.bin                    # Azure Kinect depth stream
        └── <session_id>_BL.txt               # Exported ELAN manual annotation file

ROOT_Directory_Processed/                     # Derived outputs and extracted cues
├── GENERAL_FILES/
│   ├── CHUV_data_tables.xlsx                 # Primary Excel metadata tables (Sessions, Children, Tracking)
│   ├── sessions_inventory.json               # Authoritative machine-generated archive listing
│   ├── children_table.csv                    # Parsed participant metadata (child_id, age 7-12)
│   ├── sessions_list.csv                     # Parsed session registry with audio_offset_ms
│   └── tracking_validation.json              # QA status tracking per session (valid / needs_correction)
└── SESSIONS/
    └── <session_id>/                         # Normalized session ID, e.g., 15_6
        ├── tracking/
        │   ├── masks/                        # Binary/RLE masks from SAM3
        │   └── tracks.json                   # Bounding boxes and track_id mapped to role (child vs clinician)
        ├── features/
        │   ├── heads/                        # OpenFace 2.0 / Pierre Gaze model outputs
        │   └── skeleton/                     # Sapiens / MediaPipe pose keypoints
        ├── annotations/
        │   ├── <session_id>_annotations.json # Parsed ELAN annotations (via convert_annotations.py)
        │   └── <session_id>_annotations.csv  # Frame-by-frame tab-delimited alignment
        └── validation/
            └── validation_rendered.mp4       # Video with overlaid masks, IDs & ELAN tiers
