#!/usr/bin/env python3

"""
create_sessions_metadata.py

Create a clean sessions_metadata.json from sessions_inventory.json.

Principles:
- sessions_inventory.json is the source of truth for discovered files/resources.
- Use inventory filenames directly; do not try to match annotations by session_id.
- Paths stored in metadata are relative to the raw data root.
- Supports .m4a audio files.
- Supports recursively discovered resources from the inventory.
- Preserves existing metadata where appropriate.
- Does not create script/code entries.
- Keeps metadata focused on identity, recording, resources, annotations and paths.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SCHEMA_VERSION = "2.2"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def unique(items: List[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def normalize_child_ids(session: Dict[str, Any]) -> List[str]:
    """
    Inventory currently uses child_id, including values such as:
        "14"
        "1-3"

    Convert them into the normalized list:
        ["14"]
        ["1", "3"]
    """

    value = session.get("child_id")

    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    value = str(value).strip()

    if not value:
        return []

    # Group sessions can appear as "1-3"
    if session.get("session_type") == "GROUP" and "-" in value:
        return [x for x in value.split("-") if x]

    return [value]


def normalize_session_number(value: Any) -> Any:
    if value is None:
        return None

    value = str(value).strip()

    if value.isdigit():
        return int(value)

    return value


def make_normalized_session_id(
    session_number: Any,
    session_type: str,
    child_ids: List[str]
) -> str:

    number = str(session_number)
    session_type = str(session_type).lower()

    children = "_".join(child_ids)

    if children:
        return f"{number}_{session_type}_{children}"

    return f"{number}_{session_type}"


def relative_path(
    session_id: str,
    filename: str
) -> str:
    """
    Build a portable path relative to the raw data root.

    Example:
        1-2-2024_#10_INDIVIDUAL_[14]/camera_a.mkv
    """

    return str(Path(session_id) / filename)


# ---------------------------------------------------------------------
# Inventory → recording
# ---------------------------------------------------------------------

def build_recording(
    session: Dict[str, Any]
) -> Dict[str, Any]:

    mkv_files = [
        str(x)
        for x in session.get("mkv_files", [])
    ]

    audio_files = [
        str(x)
        for x in session.get("audio_files", [])
    ]

    # Camera files are explicitly named in the inventory.
    camera_a = [
        f for f in mkv_files
        if Path(f).stem.lower() == "camera_a"
    ]

    camera_b = [
        f for f in mkv_files
        if Path(f).stem.lower() == "camera_b"
    ]

    # Fallback in case inventory contains unusual capitalization.
    if not camera_a:
        camera_a = [
            f for f in mkv_files
            if "camera_a" in Path(f).stem.lower()
        ]

    if not camera_b:
        camera_b = [
            f for f in mkv_files
            if "camera_b" in Path(f).stem.lower()
        ]

    depth_files = []

    for filename in session.get("all_files", []):
        name = str(filename).lower()

        if (
            "depth" in name
            or name.endswith(".depth")
            or ".depth." in name
        ):
            depth_files.append(str(filename))

    return {
        "camera_a": {
            "available": bool(camera_a),
            "files": camera_a
        },

        "camera_b": {
            "available": bool(camera_b),
            "files": camera_b
        },

        "audio": {
            "available": bool(audio_files),
            "files": audio_files
        }
    }


# ---------------------------------------------------------------------
# Existing resources
# ---------------------------------------------------------------------

def build_existing_resources(
    session: Dict[str, Any]
) -> Dict[str, bool]:

    return {
        "bounding_boxes": bool(
            session.get(
                "bounding_boxes_folder_available",
                False
            )
        ),

        "skeleton": bool(
            session.get(
                "skeletons_folder_available",
                False
            )
        ),

        "visualizations": bool(
            session.get(
                "Visualizations_folder_available",
                False
            )
        ),

        "reid_config": bool(
            session.get(
                "config_reid_json_available",
                False
            )
        )
    }


# ---------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------

def build_annotations(
    session: Dict[str, Any]
) -> Dict[str, Any]:

    eaf_files = unique([
        str(x)
        for x in session.get("eaf_files", [])
        if x
    ])

    txt_files = unique([
        str(x)
        for x in session.get("txt_files", [])
        if x
    ])

    return {
        "elan": {
            "available": bool(eaf_files),
            "files": eaf_files
        },

        "text_annotations": {
            "available": bool(txt_files),
            "files": txt_files
        }
    }


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def build_paths(
    session: Dict[str, Any],
    recording: Dict[str, Any],
    annotations: Dict[str, Any]
) -> Dict[str, Any]:

    session_id = session["session_id"]

    raw = {
        "camera_a": [
            relative_path(session_id, f)
            for f in recording["camera_a"]["files"]
        ],

        "camera_b": [
            relative_path(session_id, f)
            for f in recording["camera_b"]["files"]
        ],

        "audio": [
            relative_path(session_id, f)
            for f in recording["audio"]["files"]
        ]
    }

    processed = {
        "tracking": [],
        "masks": [],
        "mask_decisions": [],
        "bounding_boxes": [],
        "skeleton": [],
        "head": [],
        "head_pose": [],
        "gaze": [],
        "gaze_follow": [],
        "clinical_markers": [],
        "visualizations": []
    }

    # -----------------------------------------------------------------
    # Existing resource files from the inventory.
    #
    # We inspect all_files/folder_structure recursively so that nested
    # resources can be represented without hard-coding their structure.
    # -----------------------------------------------------------------

    all_files = [
        str(x)
        for x in session.get("all_files", [])
    ]

    folder_structure = session.get(
        "folder_structure",
        {}
    )

    folders = folder_structure.get(
        "folders",
        {}
    )

    recursive_files = []

    def collect_files(
        node: Dict[str, Any],
        prefix: str = ""
    ) -> None:

        for filename in node.get("files", []):
            if prefix:
                recursive_files.append(
                    str(Path(prefix) / filename)
                )
            else:
                recursive_files.append(str(filename))

        for folder_name, child in node.get(
            "folders",
            {}
        ).items():

            child_prefix = (
                str(Path(prefix) / folder_name)
                if prefix
                else folder_name
            )

            collect_files(
                child,
                child_prefix
            )

    collect_files(folders)

    all_resource_files = unique(
        all_files + recursive_files
    )

    # Don't duplicate raw recordings/annotations into processed paths.
    raw_names = set(
        recording["camera_a"]["files"]
        + recording["camera_b"]["files"]
        + recording["audio"]["files"]
        + session.get("eaf_files", [])
        + session.get("txt_files", [])
    )

    for file_path in all_resource_files:

        filename = Path(file_path).name.lower()

        if file_path in raw_names:
            continue

        if filename.endswith(".eaf") or filename.endswith(".txt"):
            continue

        relative = relative_path(
            session_id,
            file_path
        )

        lower_path = file_path.lower()

        if "bounding_boxes" in lower_path:
            processed["bounding_boxes"].append(relative)

        elif "skeleton" in lower_path:
            processed["skeleton"].append(relative)

        elif "visualization" in lower_path:
            processed["visualizations"].append(relative)

        elif "mask" in lower_path:
            processed["masks"].append(relative)

    return {
        "raw_session_directory": session_id,

        "raw": raw,

        "processed": processed
    }


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def build_session_metadata(
    session: Dict[str, Any]
) -> Dict[str, Any]:

    session_id = session["session_id"]

    session_type = session.get(
        "session_type",
        "UNKNOWN"
    )

    session_number = normalize_session_number(
        session.get("session_number")
    )

    child_ids = normalize_child_ids(session)

    recording = build_recording(session)

    annotations = build_annotations(session)

    existing_resources = build_existing_resources(
        session
    )

    paths = build_paths(
        session,
        recording,
        annotations
    )

    return {
        "session_id": session_id,

        "identity": {
            "date": session.get("session_date"),
            "session_type": session_type,
            "session_number": session_number,
            "child_ids": child_ids
        },

        "recording": recording,

        "existing_resources": existing_resources,

        "annotations": annotations,

        "paths": paths
    }


# ---------------------------------------------------------------------
# Safe merge
# ---------------------------------------------------------------------

def merge_session(
    old: Dict[str, Any],
    new: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Inventory-derived fields are refreshed.

    Unknown fields from an existing metadata file are preserved,
    so manually added research information is not destroyed.

    The clean inventory structure remains authoritative for:
      identity
      recording
      existing_resources
      annotations
      paths/raw
    """

    if not old:
        return new

    merged = dict(old)

    # These sections are generated from the inventory and should
    # always be refreshed.
    authoritative_sections = [
        "session_id",
        "identity",
        "recording",
        "existing_resources",
        "annotations"
    ]

    for key in authoritative_sections:
        if key in new:
            merged[key] = new[key]

    # Paths are partly inventory-derived.
    if "paths" not in merged:
        merged["paths"] = new["paths"]
    else:
        merged_paths = dict(merged["paths"])

        merged_paths["raw_session_directory"] = (
            new["paths"]["raw_session_directory"]
        )

        merged_paths["raw"] = new["paths"]["raw"]

        # Preserve existing processed outputs.
        if "processed" not in merged_paths:
            merged_paths["processed"] = new["paths"]["processed"]

        merged["paths"] = merged_paths

    return merged


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create clean sessions_metadata.json "
            "from sessions_inventory.json"
        )
    )

    parser.add_argument(
        "--inventory",
        required=True,
        help="Path to sessions_inventory.json"
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output metadata path. "
            "Default: next to inventory."
        )
    )

    args = parser.parse_args()

    inventory_path = Path(
        args.inventory
    )

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory not found: {inventory_path}"
        )

    output_path = (
        Path(args.output)
        if args.output
        else inventory_path.parent
        / "sessions_metadata.json"
    )

    inventory = load_json(
        inventory_path
    )

    if not isinstance(inventory, list):
        raise ValueError(
            "sessions_inventory.json must contain a list."
        )

    # ---------------------------------------------------------------
    # Existing metadata
    # ---------------------------------------------------------------

    if output_path.exists():

        existing_metadata = load_json(
            output_path
        )

        existing_sessions = existing_metadata.get(
            "sessions",
            {}
        )

        created_at = existing_metadata.get(
            "dataset",
            {}
        ).get(
            "created_at",
            datetime.now().isoformat(
                timespec="milliseconds"
            )
        )

        logger.info(
            f"Updating existing metadata: {output_path}"
        )

    else:

        existing_sessions = {}

        created_at = datetime.now().isoformat(
            timespec="milliseconds"
        )

        logger.info(
            f"Creating metadata: {output_path}"
        )

    # ---------------------------------------------------------------
    # Build sessions
    # ---------------------------------------------------------------

    sessions = {}

    added = 0
    updated = 0

    for inventory_session in inventory:

        session_id = inventory_session.get(
            "session_id"
        )

        if not session_id:
            logger.warning(
                "Skipping inventory entry without session_id"
            )
            continue

        new_metadata = build_session_metadata(
            inventory_session
        )

        if session_id in existing_sessions:

            new_metadata = merge_session(
                existing_sessions[session_id],
                new_metadata
            )

            updated += 1

        else:
            added += 1

        sessions[session_id] = new_metadata

    # ---------------------------------------------------------------
    # Dataset metadata
    # ---------------------------------------------------------------

    now = datetime.now().isoformat(
        timespec="milliseconds"
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,

        "dataset": {
            "name": "CHUV-ADHDArtTherapy",
            "created_at": created_at,
            "updated_at": now,
            "root_directory_raw": None,
            "root_directory_processed": None
        },

        "sessions": sessions
    }

    save_json(
        output_path,
        metadata
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Sessions metadata")
    logger.info("=" * 60)
    logger.info(
        f"Inventory sessions : {len(inventory)}"
    )
    logger.info(
        f"Added              : {added}"
    )
    logger.info(
        f"Updated            : {updated}"
    )
    logger.info(
        f"Output sessions    : {len(sessions)}"
    )
    logger.info(
        f"Output             : {output_path}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
