#!/usr/bin/env python3
"""
create_sessions_metadata.py

Create or update the central sessions_metadata.json file.

sessions_inventory.json:
    Describes which files/folders exist.

sessions_metadata.json:
    Stores authoritative session interpretation metadata:
    - session identity
    - timing
    - synchronization
    - calibration
    - tracking roles
    - processing status
    - notes

Existing metadata is preserved when updating the file.

Usage:

    python create_sessions_metadata.py \
        --inventory GENERAL_FILES/sessions_inventory.json

Optional:

    python create_sessions_metadata.py \
        --inventory GENERAL_FILES/sessions_inventory.json \
        --output GENERAL_FILES/sessions_metadata.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "1.0"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[ok] Saved: {path}")


def build_session_metadata(session):
    """
    Create the initial metadata record from sessions_inventory.json.
    """

    session_id = session["session_id"]

    child_id = session.get("child_id")

    if child_id is None:
        child_ids = []
    elif isinstance(child_id, list):
        child_ids = [str(x) for x in child_id]
    else:
        # GROUP sessions may contain "1-3".
        child_ids = [str(child_id)]

    return {
        "session": {
            "session_id": session_id,
            "session_date": session.get("session_date"),
            "child_ids": child_ids,
            "session_type": session.get("session_type"),
            "session_number": (
                int(session["session_number"])
                if str(session.get("session_number", "")).isdigit()
                else session.get("session_number")
            )
        },

        "timing": {
            "master_clock": "raw_video",

            "sources": {
                "camera_a": {
                    "file": "camera_a.mkv",
                    "start_sec": 0.0,
                    "fps": None
                },

                "camera_b": {
                    "file": "camera_b.mkv",
                    "start_sec": 0.0,
                    "fps": None
                },

                "audio": {
                    "file": None,
                    "start_sec": 0.0
                }
            },

            "offsets": {
                "audio_to_camera_a_sec": None,
                "camera_b_to_camera_a_sec": None,
                "elan_to_camera_a_sec": None
            },

            "processed_video": {}
        },

        "synchronization": {
            "status": "not_calibrated",

            "video_to_audio": {
                "offset_sec": None,
                "method": None,
                "reference": None
            },

            "camera_a_to_camera_b": {
                "offset_sec": None,
                "method": None,
                "reference": None
            }
        },

        "calibration": {
            "spatial": {
                "status": "not_calibrated",
                "file": None,
                "notes": ""
            },

            "temporal": {
                "status": "not_calibrated",
                "reference_event": None,
                "notes": ""
            }
        },

        "tracking": {
            "status": "not_processed",

            "roles": {
                "child": None,
                "therapist": None
            },

            "mask_decisions_file": None
        },

        "processing": {
            "status": "raw",
            "last_updated": None
        },

        "notes": ""
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create/update sessions_metadata.json"
    )

    parser.add_argument(
        "--inventory",
        required=True,
        help="Path to sessions_inventory.json"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output sessions_metadata.json"
    )

    args = parser.parse_args()

    inventory_path = Path(args.inventory)

    if not inventory_path.exists():
        print(f"[error] Inventory not found: {inventory_path}")
        return

    # Default output next to inventory.
    output_path = (
        Path(args.output)
        if args.output
        else inventory_path.parent / "sessions_metadata.json"
    )

    inventory = load_json(inventory_path)

    if not isinstance(inventory, list):
        raise ValueError(
            "sessions_inventory.json must contain a list of sessions."
        )

    # Load existing metadata if it exists.
    if output_path.exists():
        metadata = load_json(output_path)

        if "sessions" not in metadata:
            metadata["sessions"] = {}

        print(f"[info] Updating existing metadata: {output_path}")

    else:
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "updated": None,
            "sessions": {}
        }

        print(f"[info] Creating new metadata file: {output_path}")

    # Add missing sessions.
    added = 0
    existing = 0

    for session in inventory:
        session_id = session.get("session_id")

        if not session_id:
            print("[warn] Session without session_id -- skipped")
            continue

        if session_id in metadata["sessions"]:
            existing += 1
            continue

        metadata["sessions"][session_id] = build_session_metadata(session)
        added += 1

    metadata["schema_version"] = SCHEMA_VERSION
    metadata["updated"] = datetime.now().isoformat(timespec="seconds")

    save_json(output_path, metadata)

    print()
    print("=" * 70)
    print("Sessions metadata")
    print("=" * 70)
    print(f"Sessions in inventory : {len(inventory)}")
    print(f"Existing metadata     : {existing}")
    print(f"New sessions added    : {added}")
    print(f"Total metadata        : {len(metadata['sessions'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
