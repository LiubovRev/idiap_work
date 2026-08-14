# CHUV Art Therapy Gaze Analysis 
**Status**: Structure & Tracking Pipeline (in progress!)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Architecture](#directory-architecture)
3. [Data Dictionary & Schemas](#data-dictionary--schemas)
4. [ELAN Annotation Codes](#elan-annotation-codes)
5. [T4 Tracking Pipeline Guide](#t4-tracking-pipeline-guide)
6. [Processing Checkpoints](#processing-checkpoints)
7. [Quick Reference](#quick-reference)

---

## Project Overview

### Clinical Context

- **Population**: Children with neurodevelopmental disorders (ADHD, ASD, learning disabilities); ages 7–12
- **Intervention**: Weekly art therapy sessions (30–45 minutes each)
- **Current State**: 306 sessions recorded (212 individual + 94 group); 30 manually coded (9.8% progress)
- **Key Finding**: Children show significant **gaze shift from object-focused to therapist-focused** between early and late sessions — a marker of increasing social engagement

### Goal

Automate gaze and interaction analysis to:
- Track behavioral changes over therapy progression
- Validate therapy outcomes with objective metrics
- Replace manual frame-by-frame coding (currently 10% complete, very time-intensive)

### Scope & Deliverables

| Aspect | Details |
|--------|---------|
| **Attention behaviors** | Gaze, blink/saccade, eye contact, joint attention |
| **Linguistic conditioning** | Whisper ASR transcripts (stretch goal) |
| **Duration** | 4 months (15/07 – 15/11) |
| **Primary deliverable** | Research paper with automated validation |

---

## Directory Architecture

### Raw Data: `ROOT_Directory_Raw/`

**Immutable source** — read-only Naomi archive with original session recordings.

```
ROOT_Directory_Raw/
└── SESSIONS/
    └── <day>-<month>-<year>_#<session_number>_<TYPE>_[<child_ids>]/
        ├── raw_video.mkv                  # Main recording (typically 30–45 min)
        ├── audio_track.wav                # Separate audio (iPhone capture)
        ├── depth_data.bin                 # Azure Kinect depth stream (optional)
        └── <session_id>_BL.txt            # ELAN .txt export (tab-delimited annotations)
```

#### Naming Convention

Format: `<day>-<month>-<year>_#<session_number>_<TYPE>_[<child_ids>]`

| Example | Meaning |
|---------|---------|
| `10-1-2024_#6_INDIVIDUAL_[15]` | Child 15, Jan 10, 2024, individual session #6 |
| `11-1-2024_#6_GROUP_[1-3]` | Children 1 & 3, Jan 11, 2024, group session #6 |
| `12-12-2023_#5_GROUP_[12-13-19]` | Children 12, 13, 19, Dec 12, 2023, group session #5 |

**Key Quirks** (handled by parsing scripts):
- Session numbers can repeat per child on different dates → use `session_id` (child_id + session_number) as true unique key
- A few sessions have "correct"/"error" suffixes or 4-digit anomalies (#1001, #1002, #1003)

---

### Processed Outputs: `ROOT_Directory_Processed/`

**Derived data** — all extracted features, masks, annotations, and validation outputs.

```
ROOT_Directory_Processed/
│
├── GENERAL_FILES/                         # Centralized metadata & tables
│   ├── CHUV_data_tables.xlsx               # Primary Excel workbook (5 sheets)
│   │   ├── Overview                        # Summary statistics
│   │   ├── Sessions (all)                  # 306 rows: one per session + flags
│   │   ├── Sessions (annotated)            # DEPRECATED: 36-row snapshot
│   │   ├── Children                        # 25 rows: one per child
│   │   └── Tracking + Features             # Person-level metadata
│   │
│   ├── sessions_inventory.json             # Authoritative file listing (machine-generated)
│   ├── children_table.csv                  # Quick reference: child_id, age, diagnosis
│   ├── sessions_list.csv                   # Session registry with time offsets
│   └── tracking_validation.json            # QA status tracker (valid|needs_correction|in_progress)
│
└── SESSIONS/
    └── <session_id>/                       # Normalized ID (e.g., 15_6)
        │
        ├── tracking/                       # CV masks & bounding boxes
        │   ├── masks/
        │   │   ├── <session_id>_c_mask_frames.npz     # Child mask (NumPy)
        │   │   └── <session_id>_t_mask_frames.npz     # Therapist mask (NumPy)
        │   │
        │   ├── bboxes/
        │   │   └── <session_id>_bboxes.json           # Per-track bboxes per frame
        │   │
        │   └── tracks.json                             # Tracking metadata
        │
        ├── features/                       # Pose & gaze
        │   ├── heads/
        │   │   ├── <track_id>_head_pose.json   # Yaw, pitch, roll per frame
        │   │   └── <track_id>_gaze_3d.json     # 3D gaze vector per frame
        │   │
        │   └── skeleton/
        │       ├── <track_id>_skeleton.json    # Keypoint coordinates (MediaPipe)
        │       └── ...
        │
        ├── annotations/                    # ELAN parsed formats
        │   ├── <session_id>_annotations.json       # Unified tier structure
        │   ├── <session_id>_annotations.csv        # Frame-by-frame alignment
        │   └── <session_id>_annotations_schema.txt # Tier legend
        │
        └── validation/                     # QA & visualization
            ├── validation_rendered.mp4     # Overlay video
            ├── validation_report.json      # QA metrics
            └── validation_status.txt       # Status flag
```

**Track ID Format**: `<session_id>_<role_letter>`
- `15_6_c` = Child in session 15_6
- `15_6_t` = Therapist in session 15_6

---

## Data Dictionary & Schemas

### A. Excel Metadata Tables (CHUV_data_tables.xlsx)

#### Sheet 1: Overview

High-level summary of the dataset.

```
metric | value | description
-------|-------|------------
Total Sessions | 306 | All recorded therapy sessions
Individual Sessions | 212 | One child per session
Group Sessions | 94 | Multiple children per session
Unique Children | 25 | Participants (ages 7–12)
Sessions with Audio | 292 | Synchronized audio track
Sessions Manually Coded | 30 | ELAN annotations (9.8% progress)
Sessions with Bounding Boxes | 5 | SAM3 masks extracted
Sessions with Skeletons | 5 | Pose keypoints available
```

#### Sheet 2: Sessions (all)

**One row per recorded therapy session** (306 total).

| Column | Type | Example | Definition |
|--------|------|---------|-----------|
| `session_id` | string | "15_6" | Unique ID: `{child_id}_{session_number}` |
| `session_date` | date | 2024-01-10 | Recording date (ISO 8601) |
| `child_id` | int/string | 15 or "1,3" | Child ID(s) (comma-sep for groups) |
| `session_type` | string | "INDIVIDUAL" | "INDIVIDUAL" or "GROUP" |
| `session_number` | int | 6 | Sequential per child (may repeat across children/dates) |
| `naomi_folder_name` | string | "10-1-2024_#6_INDIVIDUAL_[15]" | Raw folder name in archive |
| `audio_available` | bool | True | Has audio track |
| `time_offset_ms` | int | 1250 | Audio→Video sync offset (video starts this many ms before audio) |
| `coded_bei_xuan` | bool | True | Manually annotated (Bei-Xuan) |
| `coded_emily` | bool | False | Manually annotated (Emily, secondary) |
| `eaf_available` | bool | True | ELAN file exists |
| `mkv_files_count` | int | 1 | Number of video files |
| `depth_available` | bool | False | Has Azure Kinect depth |
| `psifx_processed` | bool | False | Mask/skeleton/gaze extraction complete |
| `bounding_boxes_folder_available` | bool | True | SAM3 bboxes exist |
| `skeletons_folder_available` | bool | True | Pose keypoints exist |
| `notes` | string | "Annotation complete, ready for SAM3" | Free-form QA notes |

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
depth_available: False
psifx_processed: False
bounding_boxes_folder_available: True
skeletons_folder_available: True
notes: "Annotation complete, ready for SAM3 validation"
```

#### Sheet 3: Sessions (annotated) — DEPRECATED

Snapshot of 36 manually-coded sessions from early phases. **Status**: Outdated.  
→ Use **Sheet 2** instead, filtering on `coded_bei_xuan=True`.

#### Sheet 4: Children

**One row per unique child participant** (25 total).

| Column | Type | Example | Definition |
|--------|------|---------|-----------|
| `child_id` | int | 15 | Unique child ID |
| `age_range` | string | "7–12" | Age bracket (no exact ages for privacy) |
| `session_type_primary` | string | "INDIVIDUAL" | Predominant session type |
| `nb_individual_sessions` | int | 3 | Count of individual sessions |
| `nb_group_sessions` | int | 1 | Count of group sessions |
| `total_sessions` | int | 4 | Total (individual + group) |
| `nb_with_audio` | int | 4 | Sessions with audio |
| `nb_coded_bei_xuan` | int | 3 | Sessions manually annotated |
| `nb_psifx_processed` | int | 0 | Sessions through pipeline |
| `clinical_notes` | string | "ADHD diagnosis" | General clinical context |
| `dropout_status` | string | "Active" | "Active" or "Discontinued" |

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
clinical_notes: ADHD diagnosis
dropout_status: Active
```

#### Sheet 5: Tracking + Features

**One row per person per session** (tracking entity level).

Specifies which CV models processed each person (child or therapist).

| Column | Type | Example | Definition |
|--------|------|---------|-----------|
| `track_id` | string | "15_6_c" | Unique tracking ID: `{session_id}_{role_letter}` |
| `session_id` | string | "15_6" | Associated session |
| `child_id` | int | 15 | Associated child |
| `role` | string | "child" | "child" or "therapist" |
| `mask_source` | string | "SAM3" | Segmentation model |
| `mask_available` | bool | True | Masks extracted |
| `pose_source` | string | "MediaPipe" | Skeleton model (MediaPipe or Sapiens) |
| `skeleton_available` | bool | True | Pose keypoints available |
| `gaze_face_source` | string | "OpenFace2.0" | Head pose/gaze model |
| `gaze_available` | bool | True | Gaze direction extracted |
| `bbox_available` | bool | False | Bounding boxes extracted |
| `notes` | string | "Mask stable throughout" | QC notes |

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
notes: Child mask stable. Gaze confidence avg 0.92

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
notes: Therapist partially out-of-frame frames 100–120. Otherwise stable.
```

---

### B. JSON Schemas

#### 1. sessions_inventory.json

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

**Key Fields**:
- `session_id`: Normalized ID (child_id + session_number)
- `all_files` / `all_folders`: Complete file tree
- `*_available`: Boolean flags for data availability
- `time_offset_ms`: Audio→Video synchronization offset

#### 2. Annotations JSON (`<session_id>_annotations.json`)

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

#### 3. Bounding Boxes (`<session_id>_bboxes.json`)

**Per-frame bounding boxes** for each tracking entity.

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

#### 5. Skeleton Keypoints (`<track_id>_skeleton.json`)

**Body pose keypoints** (17pt MediaPipe or finer Sapiens).

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
        {"name": "left_eye", "x": 285, "y": 190, "z": 0.4, "confidence": 0.98}
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

#### 6. Tracking Metadata (`tracks.json`)

**Links track IDs to roles** and processing status.

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
    }
  ]
}
```

#### 7. Validation Report (`validation_report.json`)

**QA summary** generated during visualization.

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
        "status": "PASS"
      },
      "therapist": {
        "n_frames_with_gaze": 1760,
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

#### 8. Tracking Validation Registry (`tracking_validation.json`)

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
      "notes": "Child 1 mask lost frames 500–520. Need to re-run SAM3.",
      "manual_flags": ["mask_discontinuity", "child_1_tracking_loss"],
      "pipeline_stages_complete": [
        "annotation_parsing",
        "sam3_tracking",
        "pose_extraction",
        "visualization"
      ]
    }
  ]
}
```

---

### C. CSV Formats

#### Annotations CSV (`<session_id>_annotations.csv`)

**Frame-by-frame tab-delimited alignment** of all tiers to video frames.

```
frame_no	timestamp_sec	c15_CG	c15_CA	c15_CP	c15_CSP	t1_TP	t1_TSP	JA_c15_t1
0	0.0	GU	AU	CSI	CP	TSI	T	False
1	0.033	GU	AU	CSI	CP	TSI	T	False
127	4.23	GO	AO	CST	TC	TST	TC	False
254	8.47	GO	AO	CST	TC	TSI	TC	True
1500	50.0	GT	AT	CST	TC	TST	TC	True
```

**Columns**:
- `frame_no`: 0-based frame index
- `timestamp_sec`: Seconds into video (frame_no / fps)
- Each additional column = one ELAN tier
- Cell values = ELAN codes (GU, GO, GT, AO, AT, etc.)
- Empty or "NA" if tier has no event at that frame

---

## ELAN Annotation Codes

### Individual Session Schema

The ELAN `.eaf` file (exported as `.txt`) contains **tiers** — horizontal tracks coding specific behaviors.

#### Gaze Tier (CG)

| Code | Meaning |
|------|---------|
| `GO` | Gaze at Objects (session-related materials) |
| `GT` | Gaze at Therapist (look at therapist's face/body) |
| `GNO` | Gaze at Non-session objects (sink, shoes, jacket) |
| `GU` | Gaze Undetermined |

#### Attention Tier (CA)

| Code | Meaning |
|------|---------|
| `AO` | Attending to Objects (session-related materials) |
| `ANO` | Attending to Non-session objects |
| `AU` | Attending Undetermined |

#### Attention to Therapist (CAT)

| Code | Meaning |
|------|---------|
| `AT` | Attending to Therapist |

#### Position (CP)

| Code | Meaning |
|------|---------|
| `CST` | Standing (two feet on ground) |
| `CHO` | Hovering/leaning over table |
| `CSI` | Sitting |
| `CLF` | On the floor (lying down) |
| `CGO` | Child gone (left room/not visible) |
| `CRE` | Reaching (partial rise from seated) |
| `CCR` | Crouch (knees bent, upper body forward) |

#### Session Pattern (CSP)

| Code | Meaning |
|------|---------|
| `CP` | Child creating/playing alone |
| `COB` | Child observing (therapist creates without participation) |
| `TC` | Together creating/discussing |
| `PL` | Together playing |
| `PRC` | Preparing/cleaning up |

#### Common Action in Shared Engagement (CTCA)

Used within CSP:TC, CSP:PL, or CSP:PRC:

| Code | Meaning |
|------|---------|
| `OBJ_EXCHANGE` | Transfer of materials between child and therapist |
| `ARTMAKING` | Object-based creation or sensory exploration |
| `SYMBOLIC_PLAY` | Pretending/role-play |
| `COORDINATED_PLAY` | Rule-governed or rhythmic physical interaction |
| `CONVERSATION` | Reciprocal verbal interaction |
| `CLEAN_UP_ACTIVITY` | Organizing/tidying |

#### Joint Attention (JA)

| Code | Meaning |
|------|---------|
| `TC` | Joint eye contact present (therapist ↔ child) |

#### Therapist Tiers (TP, TSP, TV)

Follow similar conventions with `T` prefix:
- `TP` = Therapist Position (TST, TSI, TGO, TLF, TRE, TCR)
- `TSP` = Therapist Session Pattern (T, TOB, TC, PL, PRC)
- `TV` = Therapist Vocalization (TS, TNS)

### Key Insight: Attention vs. Gaze Divergence

**This is the paper's headline finding:**

- **Attention (CA, CAT)**: Coded from behavioral orientation — is child engaging with object or therapist?
- **Gaze (CG)**: Coded from eye direction only — where are they actually looking?

These can **diverge**:

```
Child is ATTENDING to therapist (CAT: AT)
But GAZING at objects (CG: GO)
→ Behavioral engagement ≠ visual focus
```

**Statistical Finding**:
- Interactive activity time increased early→late: 72.6% → 82.8% (p=0.042)
- **Gaze to therapist** increased: 13.6% → 24.6% (p=0.029) ← **Stronger signal**
- Gaze to objects decreased: 80.6% → 67.0% (p=0.052, n.s.)
- **Context-dependent**: Object-gaze while attending to therapist decreased (p=0.037), but while NOT attending to therapist unchanged (p=0.844)

→ **Conclusion**: Gaze is a more sensitive marker of increasing social engagement than attention/interaction time alone.

### Group Sessions (Different Schema)

Group sessions use per-individual-prefixed tiers (X = child/individual number, Z = other party):

```
cX_CP     Child X position
cX_CA     Child X attention (AO=object, AT_Z, AC_Z)
cX_CI     Child X interaction (T_Z, C_Z)
cX_CRE    Child X creating (TC_TZ, PL_TZ)
cX_JA     Joint eye contact (T_Z)
```

---

## T4 Tracking Pipeline Guide

Complete step-by-step walkthrough for processing a single session from raw to validated outputs.

### Overview

| Step | Task | Input | Output | Duration |
|------|------|-------|--------|----------|
| 1 | Inventory Check | Session name | Confirmation | <1 min |
| 2 | Annotation Parse | ELAN .txt | JSON + CSV | 1 min |
| 3 | SAM3 Tracking | Video | Masks, bboxes | 2–4 hrs (GPU) |
| 4 | Pose & Gaze | Video + bboxes | Head pose, gaze, skeleton | 1–2 hrs (GPU) |
| 5 | Visualization | All above | MP4 video + report | 5–10 min |
| 6 | QA & Register | Validation report | Status update | <1 min |

**Total**: ~3–6 hours per 30-minute session (GPU-bound)

---

### Step 1: Raw Ingestion & Inventory Check

#### Goal
Locate raw session in Naomi archive and verify it's registered in `sessions_inventory.json`.

#### Action

```bash
# 1.1 List raw session folder
ls /ROOT_Directory_Raw/SESSIONS/ | grep "10-1-2024_#6"
# Output: 10-1-2024_#6_INDIVIDUAL_[15]/

# 1.2 Verify session in inventory
python -c "
import json
with open('GENERAL_FILES/sessions_inventory.json') as f:
    inv = json.load(f)
session = [s for s in inv if s['session_id'] == '15_6'][0]
print(f'Session {session['session_id']} found:')
print(f'  - Video: {session['mkv_files']}')
print(f'  - Audio: {session['audio_files']}')
print(f'  - ELAN: {session['eaf_files']}')
print(f'  - Time offset: {session.get('time_offset_ms', 'NOT SET')} ms')
"
```

#### Output
```
Session 15_6 found:
  - Video: ['10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv']
  - Audio: ['10-1-2024_#6_INDIVIDUAL_[15]/audio_track.wav']
  - ELAN: ['10-1-2024_#6_INDIVIDUAL_[15]_BL.txt']
  - Time offset: 1250 ms
```

#### Next
If video, audio, and ELAN file exist → proceed to Step 2.  
If missing → flag in `tracking_validation.json` with status `missing_assets`.

---

### Step 2: Annotation Parsing & Audio Alignment

#### Goal
Convert ELAN .txt export (tab-delimited tiers) into standardized JSON and frame-by-frame CSV.

#### Background

ELAN exports `.eaf` as tab-delimited text:

```
tier	start_hms	start_sec	end_hms	end_sec	dur_sec	value
t1_TP	00:04:14.273	254.273	00:04:27.530	267.53	13.257	TST
c15_CG	00:04:14.273	254.273	00:04:50.079	294.079	39.806	GO
c15_CA	00:04:14.273	254.273	00:05:10.843	310.843	56.57	AO
```

- `c15_CG` = child 15's gaze, value `GO` (gaze at objects)
- `c15_CA` = child 15's attention, value `AO` (attending to objects)

**Important**: Audio is often offset from video. `time_offset_ms = 1250` means video starts 1250ms *before* audio.

#### Action

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

#### Outputs

**15_6_annotations.json** (unified JSON):
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
    }
  },
  "metadata": {
    "total_session_duration_sec": 1800,
    "annotation_coverage_pct": 65.3,
    "time_offset_ms": 1250
  }
}
```

**15_6_annotations.csv** (frame-by-frame):
```csv
frame_no,timestamp_sec,c15_CG,c15_CA,c15_CP,c15_CSP,t1_TP,t1_TSP,JA_c15_t1
0,0.0,GU,AU,CSI,CP,TSI,T,False
127,4.23,GO,AO,CST,TC,TST,TC,False
254,8.47,GO,AO,CST,TC,TSI,TC,True
1500,50.0,GT,AT,CST,TC,TST,TC,True
```

#### Verification

```bash
python -c "
import json
with open('15_6_annotations.json') as f:
    annot = json.load(f)
    print(f'Coverage: {annot['metadata']['annotation_coverage_pct']}%')
    for tier_name, tier_data in annot['tiers'].items():
        total_dur = sum(e['duration_sec'] for e in tier_data['events'])
        pct = total_dur / annot['metadata']['total_session_duration_sec'] * 100
        print(f'  {tier_name}: {total_dur:.1f}s ({pct:.1f}%)')
"
```

---

### Step 3: SAM3 Mask & Tracking Execution

#### Goal
Run Meta's Segment Anything 3 (SAM3) to extract per-person segmentation masks and bounding boxes.

#### Background

SAM3 is a foundation model that segments objects frame-by-frame:
1. Extract masks for **child** and **therapist** independently
2. Derive **bounding boxes** (min/max X, Y per frame)
3. Map track IDs to roles: `15_6_c` = child, `15_6_t` = therapist

#### Action

```bash
# 3.1 Run SAM3 tracking
python scripts/run_sam3_tracking.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --fps 30 \
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/tracking/ \
  --device cuda:0 \
  --verbose

# 3.2 Monitor progress (2–4 hours for 30-min session)
tail -f logs/15_6_sam3.log
```

#### Outputs

**masks/15_6_c_mask_frames.npz** (binary masks, frame-by-frame):
```python
import numpy as np
masks_c = np.load('masks/15_6_c_mask_frames.npz')
# Shape: (1800, 1080, 1920) — frames × height × width
# dtype: uint8 (0=background, 1=child)
frame_0_mask = masks_c['frames'][0]
print(frame_0_mask.shape)  # (1080, 1920)
```

**bboxes/15_6_bboxes.json**:
```json
{
  "session_id": "15_6",
  "fps": 30,
  "total_frames": 1800,
  "tracks": {
    "15_6_c": {
      "role": "child",
      "frames": {
        "0": [120, 150, 400, 600],
        "1": [121, 149, 401, 599]
      },
      "frames_with_bbox": 1795
    },
    "15_6_t": {
      "role": "therapist",
      "frames": {
        "0": [800, 200, 1100, 900]
      },
      "frames_with_bbox": 1789
    }
  }
}
```

**tracks.json** (metadata):
```json
{
  "session_id": "15_6",
  "tracks": [
    {
      "track_id": "15_6_c",
      "role": "child",
      "mask_source": "SAM3",
      "mask_file": "masks/15_6_c_mask_frames.npz"
    },
    {
      "track_id": "15_6_t",
      "role": "therapist",
      "mask_source": "SAM3",
      "mask_file": "masks/15_6_t_mask_frames.npz"
    }
  ]
}
```

#### Troubleshooting

- **Tracks lost mid-session**: Common with SAM3. Check visualization in Step 5.
- **Therapist not detected**: May be out-of-frame. Flag for manual review.
- **Performance**: GPU memory ~10–12GB for 1080p at 30fps.

---

### Step 4: Pose & Gaze Extraction

#### Goal
Extract head pose (yaw, pitch, roll) and gaze direction (3D vector) for both child and therapist.

#### Action

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
  --model mediapipe \
  --output_dir /ROOT_Directory_Processed/SESSIONS/15_6/features/ \
  --device cuda:0
```

#### Outputs

**heads/15_6_c_head_pose.json**:
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
    }
  }
}
```

**heads/15_6_c_gaze_3d.json**:
```json
{
  "track_id": "15_6_c",
  "frames": {
    "0": {
      "timestamp_sec": 0.0,
      "gaze_vector": [0.15, -0.08, 0.98],
      "gaze_confidence": 0.92
    }
  }
}
```

**skeleton/15_6_c_skeleton.json**:
```json
{
  "track_id": "15_6_c",
  "model": "mediapipe",
  "keypoint_names": ["nose", "left_eye", "right_eye", ...],
  "frames": {
    "0": {
      "timestamp_sec": 0.0,
      "keypoints": [
        {"name": "nose", "x": 300, "y": 200, "confidence": 0.99}
      ]
    }
  }
}
```

---

### Step 5: Validation & Visualization

#### Goal
Overlay masks, track IDs, and ELAN annotation tiers onto video for manual QA.

#### Action

```bash
# 5.1 Render validation video
python scripts/visualize_tracking.py \
  --session_id 15_6 \
  --video_path /ROOT_Directory_Raw/SESSIONS/10-1-2024_#6_INDIVIDUAL_[15]/raw_video.mkv \
  --masks /ROOT_Directory_Processed/SESSIONS/15_6/tracking/masks/ \
  --bboxes /ROOT_Directory_Processed/SESSIONS/15_6/tracking/bboxes/15_6_bboxes.json \
  --annotations /ROOT_Directory_Processed/SESSIONS/15_6/annotations/15_6_annotations.csv \
  --output_video /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4 \
  --output_report /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_report.json \
  --fps 30 \
  --verbose
```

#### Outputs

**validation_rendered.mp4**: Video with 3 side-by-side panels:
- **Child panel**: child mask + skeleton + gaze vector + ELAN tier overlay
- **Therapist panel**: therapist mask + skeleton + gaze vector + ELAN tier overlay
- **Joint panel**: Combined view + "Mutual Gaze" highlight

**validation_report.json**:
```json
{
  "session_id": "15_6",
  "validation_timestamp": "2024-12-15T14:30:00Z",
  "validation_checks": {
    "mask_completeness": {
      "child": {"n_frames_with_mask": 1795, "coverage_pct": 99.7, "status": "PASS"},
      "therapist": {"n_frames_with_mask": 1789, "coverage_pct": 99.4, "status": "PASS"}
    },
    "track_continuity": {
      "child": {"n_track_breaks": 3, "status": "PASS_WITH_MINOR_ISSUES"},
      "therapist": {"n_track_breaks": 0, "status": "PASS"}
    },
    "annotation_alignment": {
      "pct_covered": 95.2,
      "status": "PASS"
    },
    "gaze_quality": {
      "child": {"mean_confidence": 0.91, "status": "PASS"},
      "therapist": {"mean_confidence": 0.89, "status": "PASS"}
    }
  },
  "overall_status": "VALID",
  "flagged_issues": []
}
```

#### Quality Checklist

Watch the validation video. Look for:
- ✓ Masks follow child/therapist throughout
- ✓ Track IDs stable (no sudden jumps)
- ✓ Gaze vectors point reasonably (toward objects/therapist)
- ✓ ELAN tier text matches visible behavior

---

### Step 6: Quality Assurance & Registration

#### Goal
Summarize validation results and register session status in master tracking file.

#### Action

```bash
# 6.1 Watch validation video
# Open: /ROOT_Directory_Processed/SESSIONS/15_6/validation/validation_rendered.mp4

# 6.2 If video looks good → VALID
python scripts/validate_session.py \
  --session_id 15_6 \
  --status valid \
  --reviewer "your_name" \
  --notes "Masks clean, gaze tracking stable. Ready for analysis." \
  --tracking_validation_file /ROOT_Directory_Processed/GENERAL_FILES/tracking_validation.json

# 6.3 If issues found → NEEDS_CORRECTION
python scripts/validate_session.py \
  --session_id 15_6 \
  --status needs_correction \
  --reviewer "your_name" \
  --notes "Therapist mask lost frames 800–810. Check SAM3 tracking." \
  --tracking_validation_file /ROOT_Directory_Processed/GENERAL_FILES/tracking_validation.json
```

#### Output: tracking_validation.json

```json
{
  "validation_records": [
    {
      "session_id": "15_6",
      "status": "valid",
      "reviewer": "alice",
      "review_timestamp": "2024-12-15T14:45:00Z",
      "notes": "Masks clean, gaze tracking stable. Ready for analysis.",
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
    }
  ]
}
```

---

## Processing Checkpoints

### Checkpoint A: After Annotation Parsing (Step 2)

```bash
python -c "
import json
with open('15_6_annotations.json') as f:
    a = json.load(f)
    if not a['tiers']:
        print('ERROR: No tiers found!')
    else:
        n_events = sum(len(t['events']) for t in a['tiers'].values())
        print(f'OK: {len(a['tiers'])} tiers, {n_events} events')
"
```

**Expected**: ≥2 tiers (gaze, attention), ≥20 events

### Checkpoint B: After SAM3 (Step 3)

```bash
python -c "
import numpy as np
masks_c = np.load('15_6/tracking/masks/15_6_c_mask_frames.npz')
print(f'Child mask shape: {masks_c['frames'].shape}')
pct_frames = (masks_c['frames'].max(axis=(1,2)) > 0).mean() * 100
print(f'Frames with mask: {pct_frames:.1f}%')
"
```

**Expected**: >95% frame coverage

### Checkpoint C: After Pose Extraction (Step 4)

```bash
python -c "
import json
with open('15_6/features/heads/15_6_c_gaze_3d.json') as f:
    g = json.load(f)
    confs = [g['frames'][str(i)]['gaze_confidence'] for i in range(10)]
    print(f'Gaze confidence (first 10): {sum(confs)/len(confs):.3f}')
"
```

**Expected**: Mean confidence >0.85

### Checkpoint D: After Visualization (Step 5)

```bash
python -c "
import json
with open('15_6/validation/validation_report.json') as f:
    r = json.load(f)
    print(f'Overall: {r['overall_status']}')
    if r['flagged_issues']:
        print(f'Issues: {r['flagged_issues']}')
"
```

**Expected**: `overall_status` = "VALID"

---

## Quick Reference

### File Locations

| What | Where |
|------|-------|
| Raw sessions | `ROOT_Directory_Raw/SESSIONS/<folder>/` |
| Excel metadata | `GENERAL_FILES/CHUV_data_tables.xlsx` |
| Session inventory | `GENERAL_FILES/sessions_inventory.json` |
| Processing status | `GENERAL_FILES/tracking_validation.json` |
| Parsed annotations | `ROOT_Directory_Processed/SESSIONS/<session_id>/annotations/` |
| Masks (SAM3) | `ROOT_Directory_Processed/SESSIONS/<session_id>/tracking/masks/` |
| Bounding boxes | `ROOT_Directory_Processed/SESSIONS/<session_id>/tracking/bboxes/` |
| Gaze vectors | `ROOT_Directory_Processed/SESSIONS/<session_id>/features/heads/` |
| Skeleton keypoints | `ROOT_Directory_Processed/SESSIONS/<session_id>/features/skeleton/` |
| Validation video | `ROOT_Directory_Processed/SESSIONS/<session_id>/validation/validation_rendered.mp4` |

### Useful Commands

**Find sessions ready for processing**:
```bash
python -c "
import pandas as pd
df = pd.read_excel('GENERAL_FILES/CHUV_data_tables.xlsx', sheet_name='Sessions (all)')
ready = df[(df['coded_bei_xuan'] == True) & (df['psifx_processed'] == False)]
print(f'Ready to process: {len(ready)} sessions')
for _, row in ready.head(5).iterrows():
    print(f\"  {row['session_id']}: {row['naomi_folder_name']}\")
"
```

**Check processing status**:
```bash
python -c "
import json
with open('GENERAL_FILES/tracking_validation.json') as f:
    records = json.load(f)['validation_records']
    statuses = {}
    for r in records:
        s = r['status']
        statuses[s] = statuses.get(s, 0) + 1
    for status, count in sorted(statuses.items()):
        print(f'{status}: {count}')
"
```

**List all processed files for a session**:
```bash
find ROOT_Directory_Processed/SESSIONS/15_6 -type f | sort
```

---

## Summary

This documentation provides:

1. **Complete directory architecture** — where all data lives
2. **Full schema reference** — all JSON, CSV, and Excel structures with examples
3. **ELAN tier definitions** — complete behavioral coding scheme with key insight
4. **Step-by-step T4 pipeline** — exact commands and expected outputs for each processing stage
5. **Quality checkpoints** — verification steps at each stage
6. **Quick reference** — file locations and useful commands

For questions on specific schemas or processing steps, cross-reference the table of contents above.

**Maintained by**: Idiap × CHUV Project Team  
**Status**: T3/T4 Complete (Structure & Tracking Pipeline Documented)  
**Last Updated**: 2024-12-TBD
