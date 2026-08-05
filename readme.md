# CHUV-ADHDArtTherapy Data Architecture & Processing Pipeline

## 1. Directory Structure

The repository/server storage is divided into raw immutable data (Naomi Archive) and derived features/outputs.

```text
ROOT_Directory_Raw/
└── SESSIONS/
    └── <day>-<month>-<year>_#<session_number>_<TYPE>_[<child_ids>]/
        ├── raw_video.mkv                  # Main session recording (e.g., 10-1-2024_#6_INDIVIDUAL_[15].mkv)
        ├── audio_track.wav                # Synchronized audio (iPhone capture + manual offset sync)
        ├── depth_data.bin                 # Azure Kinect depth stream (optional)
        └── <session_id>_BL.txt            # Exported ELAN manual annotation (tab-delimited tiers)
---
ROOT_Directory_Processed/
│
├── GENERAL_FILES/                         # Centralized metadata & tables
│   ├── CHUV_data_tables.xlsx               # Primary Excel workbook
│   │   ├── Overview                        # Summary: 306 sessions, 25 children, coding progress
│   │   ├── Sessions (all)                  # One row per session; file availability flags
│   │   ├── Sessions (annotated)            # Subset with manual ELAN coding (36 rows, deprecated)
│   │   ├── Children                        # One row per child; participation metadata
│   │   └── Tracking + Features             # Person-level tracking (child vs. therapist per session)
│   │
│   ├── sessions_inventory.json             # Authoritative machine-generated file listing
│   │   └── Fields: session_id, session_date, child_id, session_type, all_files, mkv_files, audio_files, eaf_files
│   │
│   ├── children_table.csv                  # Quick reference: child_id, age, gender, clinical_diagnosis
│   ├── sessions_list.csv                   # Session registry with audio_offset_ms for alignment
│   └── tracking_validation.json            # QA status per session (valid | needs_correction | not_processed)
│
└── SESSIONS/
    └── <session_id>/                       # Normalized ID, e.g., 15_6 (child 15, session #6)
        │
        ├── tracking/                       # Computer vision outputs: masks, bounding boxes, tracking IDs
        │   ├── masks/                      # Binary or RLE-encoded masks from SAM3 (per-person segmentation)
        │   │   ├── 15_6_c_mask_frames.npz  # Child mask across all frames
        │   │   └── 15_6_t_mask_frames.npz  # Therapist mask across all frames
        │   │
        │   ├── bboxes/                     # Bounding boxes derived from masks
        │   │   └── 15_6_bboxes.json        # {track_id: [x_min, y_min, x_max, y_max, frame_no]}
        │   │
        │   └── tracks.json                 # Tracking metadata
        │       ├── track_id: "15_6_c" | "15_6_t"
        │       ├── role: "child" | "therapist"
        │       ├── mask_source: "SAM3"
        │       ├── pose_source: "MediaPipe" | "Sapiens"
        │       └── gaze_face_source: "OpenFace2.0" | "Pierre_GazeModel"
        │
        ├── features/                       # Extracted body pose & gaze direction
        │   ├── heads/                      # Face/head outputs from OpenFace 2.0 & gaze models
        │   │   ├── 15_6_c_head_pose.json   # Head rotation (yaw, pitch, roll) per frame
        │   │   ├── 15_6_c_gaze_3d.json     # Gaze direction (3D normalized vector) per frame
        │   │   ├── 15_6_t_head_pose.json   # Therapist head pose
        │   │   └── 15_6_t_gaze_3d.json     # Therapist gaze direction
        │   │
        │   └── skeleton/                   # Keypoint pose data
        │       ├── 15_6_c_skeleton.json    # Child skeleton keypoints (e.g., MediaPipe 33-point model)
        │       └── 15_6_t_skeleton.json    # Therapist skeleton keypoints
        │
        ├── annotations/                    # Manual ELAN coding (parsed into standard formats)
        │   ├── 15_6_annotations.json       # Unified JSON: {tier_name: [{start_sec, end_sec, code, duration}]}
        │   ├── 15_6_annotations.csv        # Frame-by-frame tab-delimited (for alignment with video)
        │   └── 15_6_annotations_schema.txt # Human-readable tier legend (for quick reference)
        │
        └── validation/                     # Quality control & visualization
            ├── validation_rendered.mp4     # Video with overlaid masks, track IDs & ELAN tiers
            ├── validation_report.json      # QA metrics (mask completeness, track stability, annotation overlap)
            └── validation_status.txt       # Status: valid | needs_correction | in_progress
