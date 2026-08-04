# CHUV-ADHDArtTherapy Data Structure & Processing Pipeline

## 1. Directory Structure

The project data is split into raw (read-only) and processed datasets.

```text
ROOT_Directory_Raw/                           # [READ-ONLY] Original raw session recordings
├── SESSIONS/
│   └── SESSION_ID/
│       ├── raw_video.mp4
│       ├── depth_data.bin
│       └── audio_track.wav

ROOT_Directory_Processed/                     # Processed outputs and derived features
├── GENERAL_FILES/
│   ├── children_table.csv                    # Metadata for participants (child_id, age, etc.)
│   ├── sessions_list.json                    # Session registry with flags and offsets
│   └── tracking_validation.json              # Global status registry for session tracking
└── SESSIONS/
    └── SESSION_ID/
        ├── tracking/
        │   ├── masks/                        # Binary/RLE masks (e.g. via SAM3)
        │   └── tracks.json                   # Track IDs mapped to role (child vs clinician)
        ├── features/
        │   ├── heads/                        # Head detection & association
        │   └── skeleton/                     # Pose estimations
        ├── annotations/
        │   └── manual_annotations.json       # Clinician / manual ground-truth labels
        └── validation/
            └── validation_rendered.mp4       # Video render for QA check
