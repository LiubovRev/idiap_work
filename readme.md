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
---
# Data Dictionary & Schema Reference

Complete reference for all data structures used in the CHUV project pipeline.

---

## Section A: Metadata Tables

### CHUV_data_tables.xlsx

#### Sheet 1: Overview

High-level summary of the full dataset.

| Column | Type | Example | Definition |
|--------|------|---------|-----------|
| `metric` | string | "Total Sessions" | Metric name |
| `value` | integer or float | 306 | Metric value |
| `description` | string | "All recorded therapy sessions" | What this metric counts |

**Example rows**:
```
metric,value,description
Total Sessions,306,All recorded therapy sessions
Individual Sessions,212,One child per session
Group Sessions,94,Multiple children per session
Unique Children,25,Participants (ages 7–12)
Sessions with Audio,292,Has synchronized audio track
Sessions Manually Coded,30,ELAN annotations completed (9.8% progress)
Sessions with Bounding Boxes,5,SAM3 masks extracted
Sessions with Skeletons,5,Pose keypoints available
```

---

#### Sheet 2: Sessions (all)

**One row per recorded therapy session** (306 total rows).

| Column | Type | Example | Definition | Notes |
|--------|------|---------|-----------|-------|
| `session_id` | string | "15_6" | Unique session identifier | Format: `{child_id}_{session_number}` (not date-based for stability) |
| `session_date` | date | 2024-01-10 | Date of recording | ISO 8601 format |
| `child_id` | string/int | 15 | Individual child identifier | For group sessions, comma-separated: "1,3" |
| `session_type` | string | "INDIVIDUAL" or "GROUP" | Session classification | Coding schema differs by type |
| `session_number` | integer | 6 | Sequential session number (per child) | Not always unique per child; use session_id for unique key |
| `naomi_folder_name` | string | "10-1-2024_#6_INDIVIDUAL_[15]" | Raw folder name in Naomi archive | Parsed into above columns |
| `audio_available` | boolean | True | Audio track exists | iPhone-captured audio + time sync |
| `time_offset_ms` | integer | 1250 | Audio→Video time offset | Video starts this many ms before audio; for alignment |
| `coded_bei_xuan` | boolean | True | Manually annotated by Bei-Xuan | Primary annotator |
| `coded_emily` | boolean | False | Manually annotated by Emily | Secondary/validation annotator |
| `eaf_available` | boolean | True | ELAN file exists | `.eaf` or `.txt` export available |
| `mkv_files_count` | integer | 1 | Number of video files | Usually 1; multiple if recording split |
| `depth_available` | boolean | False | Azure Kinect depth stream | Optional; not all sessions have depth |
| `psifx_processed` | boolean | False | Run through pipeline | Mask/skeleton/gaze extraction complete |
| `bounding_boxes_folder_available` | boolean | True | SAM3 bounding box outputs exist | Derived from masks |
| `skeletons_folder_available` | boolean | True | Pose keypoint files exist | MediaPipe or Sapiens output |
| `notes` | string | "Audio offset confirmed" | Any special conditions | Free-form field for flagging issues |

**Example row**:
```
session_id: 15_6
session_date: 2024-01-10
child_id: 15
session_type: INDIVIDUAL
session_number: 6
naomi_folder_name: 10-1-2024_#6_INDIVIDUAL_[15]
audio_available: True
time_offset_ms: 1250
coded_bei_xuan: True
coded_emily: False
eaf_available: True
mkv_files_count: 1
depth_available: False
psifx_processed: False
bounding_boxes_folder_available: True
skeletons_folder_available: True
notes: "Annotation complete, ready for SAM3 validation"
```

---

#### Sheet 3: Sessions (annotated) — DEPRECATED

**Snapshot of only the 36 manually-coded sessions** (from early project phases).

→ **Status**: Outdated. Use **Sheet 2: Sessions (all)** instead, filtering on `coded_bei_xuan=True`.

---

#### Sheet 4: Children

**One row per unique child participant** (25 total rows).

| Column | Type | Example | Definition | Notes |
|--------|------|---------|-----------|-------|
| `child_id` | integer | 15 | Unique child identifier | Primary key |
| `age_range` | string | "7–12" | Reported age bracket | All participants in this range; no exact ages for privacy |
| `session_type_primary` | string | "INDIVIDUAL" | Predominant session type | Most sessions are individual or group |
| `nb_individual_sessions` | integer | 3 | Count of individual sessions | Child attended alone |
| `nb_group_sessions` | integer | 1 | Count of group sessions | Child attended with peers |
| `total_sessions` | integer | 4 | Total participated sessions | Sum of individual + group |
| `nb_with_audio` | integer | 4 | Sessions with audio track | Fully synchronized |
| `nb_coded_bei_xuan` | integer | 3 | Sessions manually annotated | ELAN coding complete |
| `nb_psifx_processed` | integer | 0 | Sessions through pipeline | Automated mask/skeleton extraction |
| `clinical_notes` | string | "ADHD diagnosis" | Clinical context | General condition; no specific details |
| `dropout_status` | string | "Active" or "Discontinued" | Participation status | Whether child is still attending |

**Example row**:
```
child_id: 15
age_range: 7–12
session_type_primary: INDIVIDUAL
nb_individual_sessions: 3
nb_group_sessions: 1
total_sessions: 4
nb_with_audio: 4
nb_coded_bei_xuan: 3
nb_psifx_processed: 0
clinical_notes: "ADHD diagnosis"
dropout_status: "Active"
```

---

#### Sheet 5: Tracking + Features

**One row per person per session** (tracking entity level).

Specifies which computer-vision models processed each person (child or therapist) in each session.

| Column | Type | Example | Definition | Notes |
|--------|------|---------|-----------|-------|
| `track_id` | string | "15_6_c" | Unique tracking identifier | Format: `{session_id}_{role_letter}` (c=child, t=therapist) |
| `session_id` | string | "15_6" | Associated session | Links to Sessions (all) sheet |
| `child_id` | integer | 15 | Associated child | Links to Children sheet |
| `role` | string | "child" or "therapist" | Person's role in session | Determines applicable annotations |
| `mask_source` | string | "SAM3" | Model used for segmentation | Segment Anything 3 (only current option) |
| `mask_available` | boolean | True | Masks have been extracted | Binary/RLE encoded frame-by-frame |
| `pose_source` | string | "MediaPipe" | Skeleton keypoint model | MediaPipe (current) or Sapiens (planned) |
| `skeleton_available` | boolean | True | Pose keypoints available | 17pt (MediaPipe) or richer (Sapiens) |
| `gaze_face_source` | string | "OpenFace2.0" | Model for head pose / gaze | OpenFace2.0 (current) or Pierre's model (planned) |
| `gaze_available` | boolean | True | Gaze direction extracted | 3D normalized vector per frame |
| `bbox_available` | boolean | False | Bounding boxes extracted | Alternative to mask-based tracking |
| `notes` | string | "Mask track stable throughout" | QC or processing notes | Free-form |

**Example rows**:
```
track_id: 15_6_c
session_id: 15_6
child_id: 15
role: child
mask_source: SAM3
mask_available: True
pose_source: MediaPipe
skeleton_available: True
gaze_face_source: OpenFace2.0
gaze_available: True
bbox_available: False
notes: "Child mask stable. Gaze confidence avg 0.92"

---

track_id: 15_6_t
session_id: 15_6
child_id: 15
role: therapist
mask_source: SAM3
mask_available: True
pose_source: MediaPipe
skeleton_available: True
gaze_face_source: OpenFace2.0
gaze_available: True
bbox_available: False
notes: "Therapist partially out-of-frame frames 100–120. Otherwise stable."
```

---

## Section B: JSON Schemas

### 1. sessions_inventory.json

**Authoritative machine-generated inventory** of the Naomi archive.

```json
{
  "scan_timestamp": "2024-12-15T10:30:00Z",
  "scan_root": "/ROOT_Directory_Raw/SESSIONS/",
  "sessions": [
    {
      "session_id": "15_6",
      "session_date": "2024-01-10",
      "naomi_folder_name": "10-1-2024_#6_INDIVIDUAL_[15]",
      "child_id": 15,
      "session_type": "INDIVIDUAL",
      "session_number": 6,
      "all_files": ["raw_video.mkv", "audio_track.wav", "10-1-2024_#6_INDIVIDUAL_[15]_BL.txt"],
      "all_folders": ["depth_data"],
      "mkv_files": ["raw_video.mkv"],
      "audio_files": ["audio_track.wav"],
      "eaf_files": ["10-1-2024_#6_INDIVIDUAL_[15]_BL.txt"],
      "txt_files": ["10-1-2024_#6_INDIVIDUAL_[15]_BL.txt"],
      "depth_files": ["depth_data/depth_stream.bin"],
      "bounding_boxes_folder_available": true,
      "skeletons_folder_available": false,
      "config_reid_available": false,
      "time_offset_ms": 1250,
      "file_count": 5,
      "notes": ""
    }
  ],
  "summary": {
    "total_sessions": 306,
    "total_folders_scanned": 306,
    "sessions_with_video": 306,
    "sessions_with_audio": 292,
    "sessions_with_elan": 30
  }
}
```

**Field Definitions**:

| Field | Type | Definition |
|-------|------|-----------|
| `session_id` | string | Normalized ID (child_id + session_number) |
| `session_date` | string | ISO 8601 date |
| `naomi_folder_name` | string | Original folder name in archive |
| `child_id` | int/string | Child ID or comma-separated for group sessions |
| `session_type` | string | "INDIVIDUAL" or "GROUP" |
| `session_number` | int | Sequential number per child (may repeat across children/dates) |
| `all_files` | array | All file names in folder |
| `all_folders` | array | All subdirectory names |
| `mkv_files` | array | Video files (.mkv or .mp4) |
| `audio_files` | array | Audio tracks (.wav or .mp3) |
| `eaf_files` | array | ELAN exports (.eaf or .txt) |
| `depth_files` | array | Azure Kinect depth streams |
| `*_folder_available` | boolean | Whether that type of data exists |
| `time_offset_ms` | integer | Audio→Video synchronization offset |

---

### 2. Annotations JSON (`<session_id>_annotations.json`)

**Parsed ELAN tiers** in unified format.

```json
{
  "session_id": "15_6",
  "session_date": "2024-01-10",
  "child_id": 15,
  "session_type": "INDIVIDUAL",
  "tiers": {
    "c15_CG": {
      "tier_name": "c15_CG",
      "tier_meaning": "Child 15, Gaze",
      "tier_prefix": "c15",
      "tier_suffix": "CG",
      "events": [
        {
          "start_sec": 254.273,
          "end_sec": 294.079,
          "duration_sec": 39.806,
          "code": "GO",
          "code_meaning": "Gaze at objects"
        },
        {
          "start_sec": 294.079,
          "end_sec": 320.0,
          "duration_sec": 25.921,
          "code": "GT",
          "code_meaning": "Gaze at therapist"
        }
      ]
    },
    "c15_CA": {
      "tier_name": "c15_CA",
      "tier_meaning": "Child 15, Attending to",
      "events": [
        {
          "start_sec": 254.273,
          "end_sec": 310.843,
          "duration_sec": 56.57,
          "code": "AO",
          "code_meaning": "Attending to objects"
        }
      ]
    }
  },
  "metadata": {
    "total_session_duration_sec": 1800,
    "n_tiers": 2,
    "n_events_total": 45,
    "annotation_coverage_pct": 65.3,
    "annotators": ["Bei-Xuan"],
    "time_offset_ms": 1250,
    "source_file": "10-1-2024_#6_INDIVIDUAL_[15]_BL.txt"
  }
}
```

---

### 3. Bounding Boxes (`<session_id>_bboxes.json`)

**Per-frame bounding boxes** for each tracking entity (child, therapist).

```json
{
  "session_id": "15_6",
  "fps": 30,
  "total_frames": 1800,
  "video_width": 1920,
  "video_height": 1080,
  "tracks": {
    "15_6_c": {
      "track_id": "15_6_c",
      "role": "child",
      "child_id": 15,
      "mask_source": "SAM3",
      "frames": {
        "0": [120, 150, 400, 600],
        "1": [121, 149, 401, 599],
        "10": [125, 145, 405, 595]
      },
      "frames_with_bbox": 1795,
      "bbox_mean_width": 280.5,
      "bbox_mean_height": 450.2
    },
    "15_6_t": {
      "track_id": "15_6_t",
      "role": "therapist",
      "frames": {
        "0": [800, 200, 1100, 900],
        "1": [801, 199, 1101, 899]
      },
      "frames_with_bbox": 1789,
      "bbox_mean_width": 300.1,
      "bbox_mean_height": 700.0
    }
  }
}
```

**bbox format**: `[x_min, y_min, x_max, y_max]` (pixel coordinates)

---

### 4. Head Pose & Gaze

#### 4a. Head Pose (`<track_id>_head_pose.json`)

```json
{
  "track_id": "15_6_c",
  "model": "OpenFace2.0",
  "frames": {
    "0": {
      "frame_no": 0,
      "timestamp_sec": 0.0,
      "yaw": -5.2,
      "pitch": 12.1,
      "roll": 3.4,
      "confidence": 0.98,
      "is_valid": true
    },
    "1": {
      "frame_no": 1,
      "timestamp_sec": 0.033,
      "yaw": -4.8,
      "pitch": 12.3,
      "roll": 3.2,
      "confidence": 0.97,
      "is_valid": true
    }
  },
  "metadata": {
    "units": "degrees",
    "convention": "yaw: ±90° (left/right), pitch: ±90° (up/down), roll: ±90° (tilt)",
    "mean_confidence": 0.92,
    "frames_valid": 1750
  }
}
```

#### 4b. Gaze Direction (`<track_id>_gaze_3d.json`)

```json
{
  "track_id": "15_6_c",
  "model": "OpenFace2.0",
  "frames": {
    "0": {
      "frame_no": 0,
      "timestamp_sec": 0.0,
      "gaze_vector": [0.15, -0.08, 0.98],
      "gaze_confidence": 0.92,
      "is_valid": true
    },
    "1": {
      "frame_no": 1,
      "timestamp_sec": 0.033,
      "gaze_vector": [0.14, -0.09, 0.99],
      "gaze_confidence": 0.91,
      "is_valid": true
    }
  },
  "metadata": {
    "representation": "3D normalized vector (x, y, z)",
    "normalization": "L2 norm = 1.0",
    "interpretation": "Gaze direction from face center toward target",
    "mean_confidence": 0.91,
    "frames_valid": 1745
  }
}
```

---

### 5. Skeleton Keypoints (`<track_id>_skeleton.json`)

**Body pose keypoints** (17pt MediaPipe or finer Sapiens model).

```json
{
  "track_id": "15_6_c",
  "model": "mediapipe",
  "keypoint_names": [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
  ],
  "frames": {
    "0": {
      "frame_no": 0,
      "timestamp_sec": 0.0,
      "keypoints": [
        {"name": "nose", "x": 300, "y": 200, "z": 0.5, "confidence": 0.99},
        {"name": "left_eye", "x": 285, "y": 190, "z": 0.4, "confidence": 0.98},
        {"name": "right_eye", "x": 315, "y": 190, "z": 0.4, "confidence": 0.98}
      ],
      "frame_confidence": 0.97
    }
  },
  "metadata": {
    "n_keypoints": 17,
    "coordinate_system": "image (x: left→right, y: top→bottom)",
    "mean_frame_confidence": 0.89,
    "frames_valid": 1750
  }
}
```

---

### 6. Tracking Metadata (`tracks.json`)

**Links track IDs to roles** and specifies processing status.

```json
{
  "session_id": "15_6",
  "tracks": [
    {
      "track_id": "15_6_c",
      "role": "child",
      "child_id": 15,
      "mask_source": "SAM3",
      "mask_file": "masks/15_6_c_mask_frames.npz",
      "mask_status": "complete",
      "pose_source": "MediaPipe",
      "pose_file": "skeleton/15_6_c_skeleton.json",
      "pose_status": "complete",
      "gaze_face_source": "OpenFace2.0",
      "gaze_files": {
        "head_pose": "heads/15_6_c_head_pose.json",
        "gaze_3d": "heads/15_6_c_gaze_3d.json"
      },
      "gaze_status": "complete"
    },
    {
      "track_id": "15_6_t",
      "role": "therapist",
      "mask_source": "SAM3",
      "mask_status": "complete",
      "pose_status": "complete",
      "gaze_status": "complete"
    }
  ]
}
```

---

### 7. Validation Report (`validation_report.json`)

**QA summary** generated during visualization step.

```json
{
  "session_id": "15_6",
  "validation_timestamp": "2024-12-15T14:30:00Z",
  "validation_checks": {
    "mask_completeness": {
      "child": {
        "n_frames_total": 1800,
        "n_frames_with_mask": 1795,
        "coverage_pct": 99.7,
        "status": "PASS"
      },
      "therapist": {
        "n_frames_total": 1800,
        "n_frames_with_mask": 1789,
        "coverage_pct": 99.4,
        "status": "PASS"
      }
    },
    "track_continuity": {
      "child": {
        "n_track_breaks": 3,
        "mean_break_duration_frames": 2,
        "mean_break_duration_sec": 0.067,
        "status": "PASS_WITH_MINOR_ISSUES"
      },
      "therapist": {
        "n_track_breaks": 0,
        "status": "PASS"
      }
    },
    "annotation_alignment": {
      "n_annotation_events": 42,
      "n_events_with_masks": 40,
      "coverage_pct": 95.2,
      "status": "PASS"
    },
    "gaze_quality": {
      "child": {
        "n_frames_with_gaze": 1750,
        "mean_confidence": 0.91,
        "min_confidence": 0.72,
        "frames_high_confidence_pct": 92.3,
        "status": "PASS"
      },
      "therapist": {
        "n_frames_with_gaze": 1760,
        "mean_confidence": 0.89,
        "frames_high_confidence_pct": 88.1,
        "status": "PASS"
      }
    }
  },
  "overall_status": "VALID",
  "flagged_issues": [],
  "recommended_next_step": "Ready for feature analysis"
}
```

---

### 8. Tracking Validation Registry (`tracking_validation.json`)

**Master status file** tracking all sessions' processing progress.

```json
{
  "validation_records": [
    {
      "session_id": "15_6",
      "status": "valid",
      "reviewer": "alice",
      "review_timestamp": "2024-12-15T14:45:00Z",
      "notes": "Masks stable, gaze tracking excellent. Ready for analysis.",
      "manual_flags": [],
      "pipeline_stages_complete": [
        "annotation_parsing",
        "sam3_tracking",
        "pose_extraction",
        "gaze_extraction",
        "visualization",
        "validation_report"
      ],
      "validation_report_file": "/ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_report.json",
      "validation_video_file": "/ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4"
    },
    {
      "session_id": "1-3_6",
      "status": "needs_correction",
      "reviewer": "bob",
      "review_timestamp": "2024-12-15T16:00:00Z",
      "notes": "Child 1 mask lost frames 500–520. Need to re-run SAM3 with adjusted parameters.",
      "manual_flags": ["mask_discontinuity", "child_1_tracking_loss"],
      "pipeline_stages_complete": [
        "annotation_parsing",
        "sam3_tracking",
        "pose_extraction",
        "visualization"
      ],
      "next_action": "Re-run SAM3 tracking with manual mask correction"
    }
  ]
}
```

---

## Section C: CSV Formats (Frame-by-Frame Alignment)

### Annotations CSV (`<session_id>_annotations.csv`)

**Frame-by-frame tab-delimited alignment** of all tiers to video frames.

```csv
frame_no	timestamp_sec	c15_CG	c15_CA	c15_CP	c15_CSP	t1_TP	t1_TSP	JA_c15_t1
0	0.0	GU	AU	CSI	CP	TSI	T	False
1	0.033	GU	AU	CSI	CP	TSI	T	False
...
127	4.23	GO	AO	CST	TC	TST	TC	False
254	8.47	GO	AO	CST	TC	TSI	TC	True
...
1800	60.0	GT	AT	CST	TC	TST	TC	False
```

**Columns**:
- `frame_no`: 0-based frame index
- `timestamp_sec`: Seconds into video (frame_no / fps)
- Each additional column = one ELAN tier
- Cell values = ELAN codes (GU, GO, GT, AO, AT, etc.)
- Empty cell or "NA" if that tier has no event at that frame

**Purpose**: Direct alignment for video overlay and feature engineering

---

## Section D: File Naming Conventions

### Processed Output Files

All files in `ROOT_Directory_Processed/SESSIONS/<session_id>/` follow these patterns:

```
annotations/
  ├── <session_id>_annotations.json          # Parsed tiers
  ├── <session_id>_annotations.csv           # Frame-by-frame alignment
  └── <session_id>_annotations_schema.txt    # Human-readable tier legend

tracking/
  ├── masks/
  │   ├── <session_id>_c_mask_frames.npz     # Child masks (NumPy)
  │   └── <session_id>_t_mask_frames.npz     # Therapist masks (NumPy)
  ├── bboxes/
  │   └── <session_id>_bboxes.json           # All track bboxes
  └── tracks.json                             # Metadata linking track_id→role

features/
  ├── heads/
  │   ├── <track_id>_head_pose.json          # Yaw, pitch, roll
  │   └── <track_id>_gaze_3d.json            # 3D gaze vector
  └── skeleton/
      ├── <track_id>_skeleton.json           # Keypoint coordinates
      └── ...

validation/
  ├── validation_rendered.mp4                # Overlay video (masks + annotations)
  ├── validation_report.json                 # QA metrics
  └── validation_status.txt                  # Human-readable status
```

**track_id format**: `<session_id>_<role_letter>`
- `15_6_c` = Child in session 15_6
- `15_6_t` = Therapist in session 15_6

---

## Appendix: ELAN Tier Code Reference

### Individual Session Codes

**Gaze (CG)**:
- `GO` = Gaze at Objects
- `GT` = Gaze at Therapist
- `GNO` = Gaze at Non-session objects
- `GU` = Gaze Undetermined

**Attention (CA)**:
- `AO` = Attending to Objects
- `ANO` = Attending to Non-session objects
- `AU` = Attending Undetermined

**Attention to Therapist (CAT)**:
- `AT` = Attending to Therapist

**Position (CP)**:
- `CST` = Child Standing
- `CHO` = Child Hovering
- `CSI` = Child Sitting
- `CLF` = Child on Floor
- `CGO` = Child Gone (left room/not visible)
- `CRE` = Child Reaching
- `CCR` = Child Crouch

**Session Pattern (CSP)**:
- `CP` = Child creating/Playing alone
- `COB` = Child Observing (therapist creates)
- `TC` = Together Creating
- `PL` = Together Playing
- `PRC` = Preparing/Cleanup

**Common Action in Shared Engagement (CTCA)**:
- `OBJ_EXCHANGE` = Object transfer
- `ARTMAKING` = Creating/sculpting
- `SYMBOLIC_PLAY` = Role-play
- `COORDINATED_PLAY` = Rhythmic play
- `CONVERSATION` = Verbal interaction
- `CLEAN_UP_ACTIVITY` = Tidying

**Joint Attention (JA)**:
- `TC` = Joint eye contact present

**Therapist Position (TP)**, **Pattern (TSP)**, **Vocalization (TV)**:
- Follow similar conventions with `T` prefix

---

# T4 Tracking Pipeline & HowTo Guide

## Processing Steps for a New Session (Complete Walkthrough)

This guide walks through all steps to take a raw session folder and produce validation outputs.

---

## Step 1: Raw Ingestion & Inventory Check

### Goal
Locate the raw session in the Naomi archive and verify it's registered in `sessions_inventory.json`.

### Action

```bash
# 1.1 List raw session folder
ls /ROOT_Directory_Raw/SESSIONS/ | grep "10-1-2024_#6"
# Output: 10-1-2024_#6_INDIVIDUAL_[15]/

# 1.2 Verify session is in authoritative inventory
python -c "
import json
with open('GENERAL_FILES/sessions_inventory.json') as f:
    inv = json.load(f)
# Check if session_id '15_6' exists
session = [s for s in inv if s['session_id'] == '15_6'][0]
print(f'Session {session['session_id']} found:')
print(f'  - Video: {session['mkv_files']}')
print(f'  - Audio: {session['audio_files']}')
print(f'  - ELAN: {session['eaf_files']}')
print(f'  - Time offset: {session.get('time_offset_ms', 'NOT SET')}')
"
```

### Output
Session metadata summary:
```
Session 15_6 found:
  - Video: ['10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv']
  - Audio: ['10-1-2024_#6_INDIVIDUAL_[15]/audio_track.wav']
  - ELAN: ['10-1-2024_#6_INDIVIDUAL_[15]_BL.txt']
  - Time offset: 1250  # milliseconds; video is 1250ms ahead of audio
```

### Next
If video, audio, and ELAN file all exist → proceed to Step 2.  
If any are missing → flag in `GENERAL_FILES/tracking_validation.json` with status `missing_assets`.

---

## Step 2: Annotation Parsing & Audio Alignment

### Goal
Convert ELAN .txt export (tab-delimited tiers) into standardized JSON and frame-by-frame CSV.

### Background: ELAN Export Format

ELAN exports a `.eaf` file as tab-delimited text:

```
tier	start_hms	start_sec	end_hms	end_sec	dur_sec	value
t1_TP	00:04:14.273	254.273	00:04:27.530	267.53	13.257	TST
c15_CG	00:04:14.273	254.273	00:04:50.079	294.079	39.806	GO
c15_CA	00:04:14.273	254.273	00:05:10.843	310.843	56.57	AO
```

→ `c15_CG` is **child 15's gaze**, with value `GO` (gaze at objects).  
→ `c15_CA` is **child 15's attention**, with value `AO` (attending to objects).

**Important**: Audio in ELAN is often offset from video. `time_offset_ms = 1250` means video starts 1250ms *before* audio.

### Action

```bash
# 2.1 Run annotation parser
python scripts/convert_annotations.py \
  --elan_file /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/10-1-2024_#6_INDIVIDUAL_[15]_BL.txt \
  --time_offset_ms 1250 \
  --session_id 15_6 \
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/annotations/

# 2.2 Check output
ls -lh /ROOT_Directory_Processed/SESSIONS/15_6/annotations/
```

### Output Files

**File 1: `15_6_annotations.json`**  
Unified JSON format:

```json
{
  "session_id": "15_6",
  "session_date": "2024-01-10",
  "child_id": 15,
  "tiers": {
    "c15_CG": {
      "tier_name": "c15_CG",
      "tier_meaning": "Child 15, Gaze",
      "events": [
        {
          "start_sec": 254.273,
          "end_sec": 294.079,
          "duration_sec": 39.806,
          "code": "GO",
          "code_meaning": "Gaze at objects"
        },
        {
          "start_sec": 294.079,
          "end_sec": 320.0,
          "duration_sec": 25.921,
          "code": "GT",
          "code_meaning": "Gaze at therapist"
        }
      ]
    },
    "c15_CA": {
      "tier_name": "c15_CA",
      "tier_meaning": "Child 15, Attending to",
      "events": [
        {
          "start_sec": 254.273,
          "end_sec": 310.843,
          "duration_sec": 56.57,
          "code": "AO",
          "code_meaning": "Attending to objects"
        }
      ]
    }
  },
  "metadata": {
    "total_session_duration_sec": 1800,
    "annotation_coverage_pct": 65.3,
    "time_offset_ms": 1250
  }
}
```

**File 2: `15_6_annotations.csv`**  
Frame-by-frame (aligned to video):

```csv
frame_no,timestamp_sec,c15_CG,c15_CA,c15_CP,c15_CSP,t1_TP,t1_TSP,JA_c15_t1
0,0.0,GU,AU,CSI,CP,TSI,T,False
127,4.23,GO,AO,CST,TC,TST,TC,False
254,8.47,GO,AO,CST,TC,TSI,TC,True
...
1500,50.0,GT,AT,CST,TC,TST,TC,True
```

→ Each row = one frame (at video FPS, typically 30fps)  
→ Each column = one tier (child gaze, attention, position, etc.)

### Next
Verify annotations look reasonable:

```bash
# Check annotation coverage
python -c "
import json
with open('15_6_annotations.json') as f:
    annot = json.load(f)
    print(f'Coverage: {annot['metadata']['annotation_coverage_pct']}%')
    for tier_name, tier_data in annot['tiers'].items():
        total_dur = sum(e['duration_sec'] for e in tier_data['events'])
        print(f'  {tier_name}: {total_dur:.1f}s of {annot['metadata']['total_session_duration_sec']}s')
"
```

---

## Step 3: SAM3 Mask & Tracking Execution

### Goal
Run Meta's Segment Anything 3 (SAM3) to extract per-person segmentation masks and bounding boxes.

### Background

SAM3 is a foundation model that segments objects frame-by-frame. For therapy sessions, we run it to:
1. Extract masks for **child** and **therapist** independently
2. Derive **bounding boxes** (min/max X, Y per frame)
3. Map track IDs to roles: `15_6_c` = child, `15_6_t` = therapist

### Action

```bash
# 3.1 Run SAM3 tracking
python scripts/run_sam3_tracking.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --fps 30 \
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/tracking/ \
  --device cuda:0 \
  --verbose

# 3.2 Monitor progress
# (Takes ~2–4 hours for 30-min session on GPU; check logs)
tail -f logs/15_6_sam3.log
```

### Output Files

**File 1: `masks/15_6_c_mask_frames.npz`**  
Compressed NumPy array (binary masks, frame-by-frame):

```python
import numpy as np
masks_c = np.load('masks/15_6_c_mask_frames.npz')
# Shape: (n_frames, height, width) → (1800, 1080, 1920)
# dtype: uint8 (0=background, 1=child)
frame_0_mask = masks_c['frames'][0]  # First frame mask
print(frame_0_mask.shape)  # (1080, 1920)
print(frame_0_mask.sum() / frame_0_mask.size * 100)  # % pixels = child
```

**File 2: `bboxes/15_6_bboxes.json`**  
Bounding boxes per track per frame:

```json
{
  "session_id": "15_6",
  "fps": 30,
  "total_frames": 1800,
  "tracks": {
    "15_6_c": {
      "role": "child",
      "frames": {
        "0": [120, 150, 400, 600],  # [x_min, y_min, x_max, y_max]
        "1": [121, 149, 401, 599],
        "10": [125, 145, 405, 595]
      }
    },
    "15_6_t": {
      "role": "therapist",
      "frames": {
        "0": [800, 200, 1100, 900],
        "1": [801, 199, 1101, 899]
      }
    }
  }
}
```

**File 3: `tracks.json`**  
Metadata linking track IDs to roles:

```json
{
  "session_id": "15_6",
  "tracks": [
    {
      "track_id": "15_6_c",
      "role": "child",
      "child_id": 15,
      "mask_source": "SAM3",
      "mask_file": "masks/15_6_c_mask_frames.npz",
      "bbox_source": "derived_from_mask"
    },
    {
      "track_id": "15_6_t",
      "role": "therapist",
      "child_id": 15,
      "mask_source": "SAM3",
      "mask_file": "masks/15_6_t_mask_frames.npz",
      "bbox_source": "derived_from_mask"
    }
  ]
}
```

### Troubleshooting

- **Tracks get lost mid-session**: Common with SAM3. Check visualization in Step 4.
- **Therapist not detected**: May be mostly out-of-frame. Flag for manual review.
- **Performance**: GPU memory ~10–12GB for 1080p at 30fps. Use smaller batches if needed.

### Next
Proceed to Step 4 to visualize masks and check tracking quality.

---

## Step 4: Pose & Gaze Extraction

### Goal
Extract head pose (yaw, pitch, roll) and gaze direction (3D vector) for both child and therapist.

### Action

```bash
# 4.1 Extract head pose & gaze (OpenFace 2.0)
python scripts/extract_head_gaze.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --bboxes /ROOT_Directory_Processed/SESSIONS/15_6/tracking/bboxes/15_6_bboxes.json \
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/features/ \
  --device cuda:0

# 4.2 Extract skeleton keypoints (MediaPipe)
python scripts/extract_skeleton.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --bboxes /ROOT_Directory_Processed/SESSIONS/15_6/tracking/bboxes/15_6_bboxes.json \
  --model mediapipe \  # or 'sapiens' for finer poses
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/features/ \
  --device cuda:0
```

### Output Files

**File 1: `heads/15_6_c_head_pose.json`**

```json
{
  "track_id": "15_6_c",
  "frames": {
    "0": {
      "timestamp_sec": 0.0,
      "yaw": -5.2,
      "pitch": 12.1,
      "roll": 3.4,
      "confidence": 0.98
    },
    "1": {
      "timestamp_sec": 0.033,
      "yaw": -4.8,
      "pitch": 12.3,
      "roll": 3.2,
      "confidence": 0.97
    }
  }
}
```

**File 2: `heads/15_6_c_gaze_3d.json`**

```json
{
  "track_id": "15_6_c",
  "frames": {
    "0": {
      "timestamp_sec": 0.0,
      "gaze_vector": [0.15, -0.08, 0.98],  # Normalized 3D vector (x, y, z)
      "gaze_confidence": 0.92
    },
    "1": {
      "timestamp_sec": 0.033,
      "gaze_vector": [0.14, -0.09, 0.99],
      "gaze_confidence": 0.91
    }
  }
}
```

**File 3: `skeleton/15_6_c_skeleton.json`**

```json
{
  "track_id": "15_6_c",
  "model": "mediapipe",
  "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", ...],
  "frames": {
    "0": {
      "timestamp_sec": 0.0,
      "keypoints": [
        {"name": "nose", "x": 300, "y": 200, "confidence": 0.99},
        {"name": "left_eye", "x": 285, "y": 190, "confidence": 0.98}
      ]
    }
  }
}
```

---

## Step 5: Validation & Visualization

### Goal
Overlay masks, track IDs, and ELAN annotation tiers onto video for **manual QA**.

### Action

```bash
# 5.1 Render validation video (3–5 min for 30-min session)
python scripts/visualize_tracking.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --masks /ROOT_Directory_Processed/SESSIONS/15_6/tracking/masks/ \
  --bboxes /ROOT_Directory_Processed/SESSIONS/15_6/tracking/bboxes/15_6_bboxes.json \
  --annotations /ROOT_Directory_Processed/SESSIONS/15_6/annotations/15_6_annotations.csv \
  --output_video /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4 \
  --output_report /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_report.json \
  --fps 30 \
  --overlay_panels child:therapist:joint \
  --verbose
```

### Output

**File 1: `validation_rendered.mp4`**  
Video with 3 side-by-side panels:
- **Child panel**: child mask + skeleton + gaze vector + ELAN tier overlay (CSP, CG, CA)
- **Therapist panel**: therapist mask + skeleton + gaze vector + ELAN tier overlay (TSP)
- **Joint panel**: Combined view + "Mutual Gaze" highlight when both looking at each other

Text overlay at bottom:
```
Frame: 254 | Time: 00:08:30 | Session: 15_6
Child 15: CG=GO (gaze@objects) | CA=AO (attend@objects) | CSP=TC (together)
Therapist: TP=TST (standing) | TSP=TC (together)
Gaze Mutual? FALSE | Head-to-Head: 0.82m
```

**File 2: `validation_report.json`**

```json
{
  "session_id": "15_6",
  "validation_checks": {
    "mask_completeness": {
      "child": {"n_frames_with_mask": 1795, "pct": 99.7, "status": "PASS"},
      "therapist": {"n_frames_with_mask": 1789, "pct": 99.4, "status": "PASS"}
    },
    "track_continuity": {
      "child": {
        "n_track_breaks": 3,
        "avg_break_duration_frames": 2,
        "status": "PASS_WITH_MINOR_ISSUES"
      },
      "therapist": {
        "n_track_breaks": 0,
        "avg_break_duration_frames": 0,
        "status": "PASS"
      }
    },
    "annotation_alignment": {
      "child_gaze_events": 42,
      "child_attention_events": 38,
      "events_with_masks": 40,
      "pct_covered": 95.2,
      "status": "PASS"
    },
    "gaze_quality": {
      "child": {
        "frames_with_gaze": 1750,
        "mean_confidence": 0.91,
        "status": "PASS"
      },
      "therapist": {
        "frames_with_gaze": 1760,
        "mean_confidence": 0.89,
        "status": "PASS"
      }
    }
  },
  "overall_status": "VALID",
  "flagged_issues": [],
  "reviewed_by": "user",
  "review_timestamp": "2024-12-15T14:30:00Z"
}
```

### Next
Watch the validation video. Look for:
- ✓ Masks follow child/therapist throughout
- ✓ Track IDs stable (no sudden jumps)
- ✓ Gaze vectors point reasonably (toward objects/therapist)
- ✓ ELAN tier text matches visible behavior

Flag issues → mark as `needs_correction` in Step 6.

---

## Step 6: Quality Assurance & Registration

### Goal
Summarize validation results and register session status in master tracking file.

### Action

```bash
# 6.1 Manual review (watch validation_rendered.mp4)
# Open: /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4

# 6.2 If video looks good, mark as VALID
python scripts/validate_session.py \
  --session_id 15_6 \
  --status valid \
  --reviewer "your_name" \
  --notes "Masks clean, gaze tracking stable. Ready for analysis." \
  --tracking_validation_file /ROOT_Directory_Processed/GENERAL_FILES/tracking_validation.json

# 6.3 If issues found, mark as NEEDS_CORRECTION
python scripts/validate_session.py \
  --session_id 15_6 \
  --status needs_correction \
  --reviewer "your_name" \
  --notes "Therapist mask lost frames 800–810. Check SAM3 tracking." \
  --tracking_validation_file /ROOT_Directory_Processed/GENERAL_FILES/tracking_validation.json
```

### Output: `tracking_validation.json`

```json
{
  "validation_records": [
    {
      "session_id": "15_6",
      "status": "valid",
      "reviewer": "alice",
      "review_timestamp": "2024-12-15T14:45:00Z",
      "notes": "Masks clean, gaze tracking stable. Ready for analysis.",
      "validation_report": "/ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_report.json",
      "validation_video": "/ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4",
      "pipeline_stages_complete": ["annotation_parsing", "sam3_tracking", "pose_extraction", "visualization"],
      "next_stage": "automated_feature_analysis"
    }
  ]
}
```

---

## Summary: Full Pipeline Execution

| Step | Task | Input | Output | Time |
|------|------|-------|--------|------|
| 1 | Inventory Check | Session name | Confirmation | <1 min |
| 2 | Annotation Parse | ELAN .txt | JSON + CSV | 1 min |
| 3 | SAM3 Tracking | Video | Masks, bboxes | 2–4 hrs (GPU) |
| 4 | Pose & Gaze | Video + bboxes | Head pose, gaze, skeleton | 1–2 hrs (GPU) |
| 5 | Visualization | All above | MP4 video + report | 5–10 min |
| 6 | QA & Register | Validation report | Status update | <1 min |

**Total Time**: ~3–6 hours per 30-minute session (mostly GPU-bound)

---

## Checkpoints & Common Errors

### Checkpoint A: After Annotation Parsing (Step 2)
```bash
# Check annotation JSON parses and has events
python -c "
import json
with open('15_6_annotations.json') as f:
    a = json.load(f)
    if not a['tiers']:
        print('ERROR: No tiers found!')
    else:
        print(f'OK: {len(a['tiers'])} tiers, {sum(len(t['events']) for t in a['tiers'].values())} events')
"
```

### Checkpoint B: After SAM3 (Step 3)
```bash
# Verify mask files exist and have expected shape
python -c "
import numpy as np
masks_c = np.load('15_6/tracking/masks/15_6_c_mask_frames.npz')
print(f'Child mask shape: {masks_c['frames'].shape}')  # Should be (n_frames, H, W)
print(f'Frames with mask: {(masks_c['frames'] > 0).sum(axis=(1,2)).mean():.1f}% avg')
"
```

### Checkpoint C: After Pose Extraction (Step 4)
```bash
# Verify gaze confidence is reasonable
python -c "
import json
with open('15_6/features/heads/15_6_c_gaze_3d.json') as f:
    g = json.load(f)
    conf_vals = [g['frames'][str(i)]['gaze_confidence'] for i in range(10)]
    print(f'Gaze confidence (first 10 frames): {conf_vals}')
    print(f'Mean: {sum(conf_vals)/len(conf_vals):.3f}')  # Should be ~0.90+
"
```

### Checkpoint D: After Visualization (Step 5)
```bash
# Check validation report has no critical flags
python -c "
import json
with open('15_6/validation/validation_report.json') as f:
    r = json.load(f)
    print(f'Overall: {r['overall_status']}')
    if r['flagged_issues']:
        print(f'Issues: {r['flagged_issues']}')
    else:
        print('No critical issues.')
"
```

---

## Next: Automated Analysis Phase

Once a session is marked `valid`, proceed to:
- **Feature engineering** (relational metrics: G2G, G2H, H2H)
- **Statistical analysis** (early vs. late session comparisons)
- **Paper-ready outputs** (figures, tables, p-values)


            ├── validation_report.json      # QA metrics (mask completeness, track stability, annotation overlap)
            └── validation_status.txt       # Status: valid | needs_correction | in_progress
