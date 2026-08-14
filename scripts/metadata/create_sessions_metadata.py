#!/usr/bin/env python3
"""
create_sessions_metadata.py
===========================

Build/update sessions_metadata.json from:

1. sessions_inventory.json
   - authoritative dataset/session information

2. Raw session directories
   - actual files currently present

3. Existing sessions_metadata.json
   - preserves manually added metadata such as:
       * synchronization offsets
       * calibration
       * tracking validation
       * processing information
       * notes

The resulting sessions_metadata.json is the central registry for
raw data, annotations, tracking, feature extraction and future processing.

Example:

python create_sessions_metadata.py \
    --sessions-dir ~/Documents/chuv_machine_downloads/videos \
    --inventory ~/Documents/idiap_work/tables/sessions_inventory.json \
    --output ~/Documents/idiap_work/tables/sessions_metadata.json \
    --existing-metadata ~/Documents/idiap_work/tables/sessions_metadata.json
"""

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

SCHEMA_VERSION = "1.0"

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac"}
DEPTH_EXTENSIONS = {".bin", ".raw", ".depth"}

ANNOTATION_EXTENSIONS = {".eaf"}
TEXT_EXTENSIONS = {".txt"}

PROCESSING_STATUS_VALUES = {
    "not_processed",
    "processing",
    "done",
    "failed",
    "blocked",
    "unknown",
}


# ============================================================================
# Generic helpers
# ============================================================================

def now_iso():
    return datetime.now().isoformat()


def load_json(path):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def merge_dict_preserve_existing(old, new):
    """
    Recursively merge dictionaries.

    Values already present in `old` are preserved when `new` contains
    None/empty values.

    This is important because manually entered metadata such as
    synchronization offsets must not be destroyed by regeneration.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return deepcopy(old)

    result = deepcopy(old)

    for key, value in new.items():

        if key not in result:
            result[key] = deepcopy(value)
            continue

        old_value = result[key]

        if isinstance(old_value, dict) and isinstance(value, dict):
            result[key] = merge_dict_preserve_existing(
                old_value,
                value
            )

        elif value is not None and value != "":
            result[key] = deepcopy(value)

    return result


def relative_path(path, root):
    """Return path relative to root using POSIX separators."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ============================================================================
# Session ID helpers
# ============================================================================

def normalize_session_id(session_id):
    """
    Create a stable normalized ID.

    Examples:
        1-2-2024_#10_INDIVIDUAL_[14]
            -> 10_individual_14

        1-2-2024_#9_GROUP_[1-3]
            -> 9_group_1_3
    """

    if not session_id:
        return None

    session_id = str(session_id)

    match = re.search(
        r"#(\d+)_([A-Z]+)_\[(.*?)\]",
        session_id
    )

    if match:
        session_number = match.group(1)
        session_type = match.group(2).lower()
        children = re.sub(
            r"[^0-9]+",
            "_",
            match.group(3)
        ).strip("_")

        return f"{session_number}_{session_type}_{children}"

    # Fallback
    normalized = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        session_id
    ).strip("_")

    return normalized.lower()


def find_matching_session_directory(session_id, sessions_dir):
    """
    Find actual session directory.

    Exact match is preferred.

    Inventory IDs can contain dates and # symbols, while some downloaded
    directories may have simplified names, so a few fallback strategies
    are used.
    """

    if not sessions_dir.exists():
        return None

    # Exact match
    exact = sessions_dir / session_id
    if exact.is_dir():
        return exact

    # Case-insensitive exact match
    for directory in sessions_dir.iterdir():
        if directory.is_dir():
            if directory.name.lower() == session_id.lower():
                return directory

    # Normalize both sides
    target = normalize_session_id(session_id)

    for directory in sessions_dir.iterdir():

        if not directory.is_dir():
            continue

        if normalize_session_id(directory.name) == target:
            return directory

    return None


# ============================================================================
# File classification
# ============================================================================

def classify_file(file):
    """
    Classify one file.

    Returns:
        category
    """

    name = file.name.lower()
    suffix = file.suffix.lower()

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    if suffix in VIDEO_EXTENSIONS:

        if "camera_a" in name or re.search(r"\bcam[_-]?a\b", name):
            return "camera_a"

        if "camera_b" in name or re.search(r"\bcam[_-]?b\b", name):
            return "camera_b"

        # Processed / visualization videos
        if any(
            keyword in name
            for keyword in [
                "tracking",
                "annotated",
                "visual",
                "openpose",
                "output",
                "mask",
                "validation"
            ]
        ):
            return "visualization"

        return "video_other"

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    if suffix in AUDIO_EXTENSIONS:
        return "audio"

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------

    if suffix in DEPTH_EXTENSIONS:
        if "depth" in name:
            return "depth"

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    if suffix == ".eaf":
        return "elan"

    if suffix == ".txt":
        if any(
            keyword in name
            for keyword in [
                "annotation",
                "audioTime".lower(),
                "wakEE".lower(),
                "session",
                "individual"
            ]
        ):
            return "text_annotation"

    # ------------------------------------------------------------------
    # Metadata/config
    # ------------------------------------------------------------------

    if name == "metadata.json":
        return "metadata_json"

    if name == "config_reid.json":
        return "reid_config"

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    if "mask_decision" in name:
        return "mask_decision"

    if "mask" in name:
        return "mask"

    if "tracking" in name or "track" in name:
        return "tracking"

    if "box" in name or "bbox" in name:
        return "bounding_box"

    if "skeleton" in name or "pose" in name:
        return "skeleton"

    if "head" in name:
        return "head"

    if "gaze" in name:
        return "gaze"

    return "other"


# ============================================================================
# Directory scanning
# ============================================================================

def scan_session_directory(session_dir):
    """
    Scan a session directory recursively.

    Returns a structured inventory of files actually present.
    """

    result = {
        "camera_a": [],
        "camera_b": [],
        "audio": [],
        "depth": [],
        "elan": [],
        "text": [],
        "metadata": [],
        "reid_config": [],
        "tracking": [],
        "masks": [],
        "mask_decisions": [],
        "bounding_boxes": [],
        "head": [],
        "skeleton": [],
        "head_pose": [],
        "gaze": [],
        "visualizations": [],
        "other": [],
    }

    if session_dir is None or not session_dir.exists():
        return result

    for file in sorted(session_dir.rglob("*")):

        if not file.is_file():
            continue

        category = classify_file(file)

        path = relative_path(file, session_dir)

        if category in result:
            result[category].append(path)
        else:
            result["other"].append(path)

    return result


# ============================================================================
# Existing resources
# ============================================================================

def build_existing_resources(inventory_entry):
    """
    Information already available according to sessions_inventory.json.
    """

    return {
        "bounding_boxes": bool(
            inventory_entry.get(
                "bounding_boxes_folder_available",
                False
            )
        ),
        "skeletons": bool(
            inventory_entry.get(
                "skeletons_folder_available",
                False
            )
        ),
        "visualizations": bool(
            inventory_entry.get(
                "Visualizations_folder_available",
                False
            )
        ),
        "reid_config": bool(
            inventory_entry.get(
                "config_reid_json_available",
                False
            )
        )
    }


# ============================================================================
# Recording metadata
# ============================================================================

def build_recording_metadata(scan):

    return {
        "cameras": {
            "camera_a": {
                "available": bool(scan["camera_a"]),
                "files": scan["camera_a"],
                "fps": None,
                "resolution": None,
                "duration_s": None
            },
            "camera_b": {
                "available": bool(scan["camera_b"]),
                "files": scan["camera_b"],
                "fps": None,
                "resolution": None,
                "duration_s": None
            }
        },

        "audio": {
            "available": bool(scan["audio"]),
            "files": scan["audio"],
            "sample_rate_hz": None,
            "duration_s": None
        },

        "depth": {
            "available": bool(scan["depth"]),
            "files": scan["depth"]
        }
    }


# ============================================================================
# Annotations
# ============================================================================

def build_annotations(scan):

    return {
        "elan": {
            "available": bool(scan["elan"]),
            "files": scan["elan"],
            "format": "eaf",
            "converted": False,
            "converted_files": [],
            "version": None,
            "annotation_types": []
        },

        "text_annotations": {
            "available": bool(scan["text"]),
            "files": scan["text"]
        },

        "clinical": {
            "available": False,
            "files": [],
            "version": None,
            "annotation_types": []
        }
    }


# ============================================================================
# Tracking
# ============================================================================

def build_tracking(scan):

    # Important:
    # presence of mask-related files does NOT automatically mean
    # tracking is completed or validated.

    has_masks = bool(scan["masks"])
    has_tracking = bool(scan["tracking"])
    has_boxes = bool(scan["bounding_boxes"])

    tracking_status = "not_processed"

    if has_tracking or has_masks:
        tracking_status = "done"

    return {
        "method": "SAM3",

        "status": tracking_status,

        "tracks": [],

        "masks": {
            "status": "done" if has_masks else "not_processed",
            "files": scan["masks"],
            "video": None
        },

        "bounding_boxes": {
            "status": "done" if has_boxes else "not_processed",
            "files": scan["bounding_boxes"]
        },

        "validation": {
            "status": "not_validated",
            "decisions_file": (
                scan["mask_decisions"][0]
                if scan["mask_decisions"]
                else None
            ),
            "validation_video": None,
            "validated_by": None,
            "validated_at": None,
            "notes": ""
        }
    }


# ============================================================================
# Features
# ============================================================================

def build_features(scan):

    return {

        "head": {
            "status": "done" if scan["head"] else "not_processed",
            "method": None,
            "files": scan["head"]
        },

        "skeleton": {
            "status": (
                "done"
                if scan["skeleton"]
                else "not_processed"
            ),
            "method": None,
            "files": scan["skeleton"]
        },

        "head_pose": {
            "status": (
                "done"
                if scan["head_pose"]
                else "not_processed"
            ),
            "method": None,
            "files": scan["head_pose"]
        },

        "gaze": {
            "status": (
                "done"
                if scan["gaze"]
                else "not_processed"
            ),
            "method": None,
            "input": "head_boxes",
            "files": scan["gaze"]
        },

        "gaze_follow": {
            "status": "not_processed",
            "method": None,
            "files": []
        },

        "clinical_markers": {
            "status": "not_processed",
            "files": []
        }
    }


# ============================================================================
# Processing
# ============================================================================

def build_processing(scan):

    psifx_detected = any(
        "psifx" in path.lower()
        for category in scan.values()
        if isinstance(category, list)
        for path in category
    )

    sam3_detected = bool(
        scan["masks"] or
        scan["tracking"] or
        scan["mask_decisions"]
    )

    mediapipe_detected = any(
        "mediapipe" in path.lower()
        for category in scan.values()
        if isinstance(category, list)
        for path in category
    )

    return {
        "psifx": {
            "status": "done" if psifx_detected else "not_processed",
            "version": None,
            "processed_at": None,
            "outputs": []
        },

        "sam3": {
            "status": "done" if sam3_detected else "not_processed",
            "version": None,
            "processed_at": None,
            "outputs": (
                scan["masks"] +
                scan["tracking"] +
                scan["mask_decisions"]
            )
        },

        "mediapipe": {
            "status": (
                "done"
                if mediapipe_detected
                else "not_processed"
            ),
            "version": None,
            "processed_at": None,
            "outputs": []
        },

        "overall_status": "pending"
    }


# ============================================================================
# Paths
# ============================================================================

def build_paths(session_dir, scan):

    raw_session_directory = (
        str(session_dir)
        if session_dir is not None
        else None
    )

    return {
        "raw_session_directory": raw_session_directory,

        "processed_session_directory": None,

        "raw": {
            "camera_a": scan["camera_a"],
            "camera_b": scan["camera_b"],
            "audio": scan["audio"],
            "depth": scan["depth"]
        },

        "annotations": {
            "elan": scan["elan"],
            "text": scan["text"]
        },

        "processed": {
            "tracking": scan["tracking"],
            "masks": scan["masks"],
            "mask_decisions": scan["mask_decisions"],
            "bounding_boxes": scan["bounding_boxes"],
            "head": scan["head"],
            "skeleton": scan["skeleton"],
            "head_pose": scan["head_pose"],
            "gaze": scan["gaze"],
            "gaze_follow": [],
            "clinical_markers": [],
            "visualizations": scan["visualizations"]
        }
    }


# ============================================================================
# QC
# ============================================================================

def build_quality_control(
    inventory_entry,
    session_dir,
    scan
):

    missing_assets = []
    warnings = []

    if not scan["camera_a"]:
        missing_assets.append("camera_a")

    if not scan["camera_b"]:
        missing_assets.append("camera_b")

    if not scan["audio"]:
        warnings.append("No audio file found")

    if not session_dir:
        warnings.append(
            "Raw session directory could not be matched"
        )

    if inventory_entry.get("eaf_files") and not scan["elan"]:
        warnings.append(
            "ELAN files reported by sessions_inventory "
            "but not found in current session directory"
        )

    return {
        "missing_assets": missing_assets,
        "processing_errors": [],
        "warnings": warnings,
        "notes": ""
    }


# ============================================================================
# Session builder
# ============================================================================

def build_session_metadata(
    inventory_entry,
    sessions_dir,
    old_session=None
):

    session_id = inventory_entry["session_id"]

    session_dir = find_matching_session_directory(
        session_id,
        sessions_dir
    )

    scan = scan_session_directory(session_dir)

    child_ids = [
        str(child)
        for child in str(
            inventory_entry.get("child_id", "")
        ).split("-")
        if child
    ]

    session_type = inventory_entry.get(
        "session_type"
    )

    session_number_raw = inventory_entry.get(
        "session_number"
    )

    try:
        session_number = int(session_number_raw)
    except (TypeError, ValueError):
        session_number = session_number_raw

    # ------------------------------------------------------------
    # Children
    # ------------------------------------------------------------

    children = [
        {
            "child_id": child_id,
            "track_ids": []
        }
        for child_id in child_ids
    ]

    # ------------------------------------------------------------
    # Base metadata
    # ------------------------------------------------------------

    metadata = {

        "identity": {
            "session_id": session_id,
            "normalized_session_id":
                normalize_session_id(session_id),

            "original_folder_name": (
                session_dir.name
                if session_dir
                else None
            ),

            "session_type": session_type,
            "session_number": session_number,
            "child_ids": child_ids,
            "date": inventory_entry.get("session_date")
        },

        "participants": {
            "children": children,
            "clinicians": [],
            "other_participants": []
        },

        "recording": build_recording_metadata(scan),

        "existing_resources":
            build_existing_resources(inventory_entry),

        "synchronization": {
            "status": "unknown",

            "offsets_ms": {
                "audio_to_camera_a": None,
                "camera_b_to_camera_a": None,
                "elan_to_camera_a": None
            },

            "method": None,
            "reference_event": None,
            "confidence": None,
            "notes": ""
        },

        "calibration": {

            "spatial": {
                "status": "unknown",
                "camera_pair": [
                    "camera_a",
                    "camera_b"
                ],
                "method": None,
                "file": None,
                "notes": ""
            },

            "temporal": {
                "status": "unknown",
                "method": None,
                "file": None,
                "notes": ""
            }
        },

        "annotations":
            build_annotations(scan),

        "tracking":
            build_tracking(scan),

        "features":
            build_features(scan),

        "processing":
            build_processing(scan),

        "paths":
            build_paths(
                session_dir,
                scan
            ),

        "quality_control":
            build_quality_control(
                inventory_entry,
                session_dir,
                scan
            ),

        "provenance": {
            "metadata_created_at": now_iso(),
            "metadata_updated_at": now_iso(),
            "last_processed_at": None
        }
    }

    # ------------------------------------------------------------
    # Preserve manually maintained metadata
    # ------------------------------------------------------------

    if old_session:

        metadata = merge_dict_preserve_existing(
            old_session,
            metadata
        )

        # These are fields that should ALWAYS reflect current filesystem
        # rather than stale metadata.

        metadata["identity"]["original_folder_name"] = (
            session_dir.name
            if session_dir
            else metadata["identity"].get(
                "original_folder_name"
            )
        )

        metadata["recording"] = build_recording_metadata(scan)

        metadata["existing_resources"] = \
            build_existing_resources(
                inventory_entry
            )

        metadata["paths"] = build_paths(
            session_dir,
            scan
        )

        # Preserve manual tracking validation.
        old_tracking = old_session.get(
            "tracking",
            {}
        )

        if old_tracking.get("validation"):
            metadata["tracking"]["validation"] = \
                old_tracking["validation"]

        # Preserve manual synchronization.
        if old_session.get("synchronization"):
            metadata["synchronization"] = \
                merge_dict_preserve_existing(
                    old_session["synchronization"],
                    metadata["synchronization"]
                )

        # Preserve manual calibration.
        if old_session.get("calibration"):
            metadata["calibration"] = \
                merge_dict_preserve_existing(
                    old_session["calibration"],
                    metadata["calibration"]
                )

        # Preserve provenance creation timestamp.
        metadata["provenance"][
            "metadata_created_at"
        ] = old_session.get(
            "provenance",
            {}
        ).get(
            "metadata_created_at",
            now_iso()
        )

        metadata["provenance"][
            "metadata_updated_at"
        ] = now_iso()

    return metadata


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--sessions-dir",
        required=True,
        type=Path,
        help="Directory containing raw session folders"
    )

    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="sessions_inventory.json"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output sessions_metadata.json"
    )

    parser.add_argument(
        "--existing-metadata",
        type=Path,
        default=None,
        help="Existing sessions_metadata.json to preserve manual metadata"
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("CHUV SESSION METADATA")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load inventory
    # ------------------------------------------------------------------

    inventory = load_json(args.inventory)

    if not isinstance(inventory, list):
        raise ValueError(
            "sessions_inventory.json must contain a JSON list."
        )

    print(
        f"[info] Loaded inventory: "
        f"{len(inventory)} sessions"
    )

    # ------------------------------------------------------------------
    # Load existing metadata
    # ------------------------------------------------------------------

    existing_data = {}

    if (
        args.existing_metadata
        and args.existing_metadata.exists()
    ):

        existing_data = load_json(
            args.existing_metadata
        )

        if not isinstance(existing_data, dict):
            raise ValueError(
                "Existing sessions_metadata.json "
                "must contain a JSON object."
            )

        print(
            f"[info] Loaded existing metadata: "
            f"{args.existing_metadata}"
        )

    existing_sessions = existing_data.get(
        "sessions",
        {}
    )

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    created_at = (
        existing_data
        .get("dataset", {})
        .get("created_at")
        or now_iso()
    )

    root_processed = (
        existing_data
        .get("dataset", {})
        .get("root_directory_processed")
    )

    metadata = {

        "schema_version": SCHEMA_VERSION,

        "dataset": {
            "name": "CHUV-ADHDArtTherapy",

            "created_at": created_at,

            "updated_at": now_iso(),

            "root_directory_raw":
                str(args.sessions_dir),

            "root_directory_processed":
                root_processed
        },

        "sessions": {}
    }

    # ------------------------------------------------------------------
    # Build sessions
    # ------------------------------------------------------------------

    matched = 0
    unmatched = 0

    for index, inventory_entry in enumerate(inventory, start=1):

        session_id = inventory_entry.get(
            "session_id"
        )

        if not session_id:
            print(
                f"[warning] Inventory entry #{index} "
                f"has no session_id - skipped"
            )
            continue

        old_session = existing_sessions.get(
            session_id
        )

        session = build_session_metadata(
            inventory_entry=inventory_entry,
            sessions_dir=args.sessions_dir,
            old_session=old_session
        )

        metadata["sessions"][session_id] = session

        raw_dir = session["paths"][
            "raw_session_directory"
        ]

        if raw_dir:
            matched += 1
            marker = "✓"
        else:
            unmatched += 1
            marker = "⚠"

        recording = session["recording"]

        camera_a = recording["cameras"]["camera_a"][
            "available"
        ]

        camera_b = recording["cameras"]["camera_b"][
            "available"
        ]

        audio = recording["audio"]["available"]

        print(
            f"{marker} {session_id}"
            f" | A={camera_a}"
            f" B={camera_b}"
            f" audio={audio}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_json(
        metadata,
        args.output
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    sessions = metadata["sessions"]

    total = len(sessions)

    camera_a = sum(
        1
        for s in sessions.values()
        if s["recording"]["cameras"]["camera_a"]["available"]
    )

    camera_b = sum(
        1
        for s in sessions.values()
        if s["recording"]["cameras"]["camera_b"]["available"]
    )

    audio = sum(
        1
        for s in sessions.values()
        if s["recording"]["audio"]["available"]
    )

    elan = sum(
        1
        for s in sessions.values()
        if s["annotations"]["elan"]["available"]
    )

    reid = sum(
        1
        for s in sessions.values()
        if s["existing_resources"]["reid_config"]
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Total sessions:       {total}")
    print(f"Matched raw folders:  {matched}")
    print(f"Unmatched folders:    {unmatched}")
    print(f"Camera A:             {camera_a}")
    print(f"Camera B:             {camera_b}")
    print(f"Audio:                {audio}")
    print(f"ELAN annotations:     {elan}")
    print(f"ReID configs:         {reid}")

    print()
    print(
        f"[ok] Saved metadata: {args.output}"
    )


if __name__ == "__main__":
    main()
