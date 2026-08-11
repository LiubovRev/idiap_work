#!/usr/bin/env python3
"""
mask_tool.py

Unified mask inspection, mask decision application, and annotation tool.

Workflow
--------
1. Inspect mask tracks:
       python mask_tool.py inspect --session_dir /path/to/session

2. Edit decisions if necessary, then apply them:
       python mask_tool.py apply --session_dir /path/to/session

3. Create the annotated tracking visualization:
       python mask_tool.py annotate --session_dir /path/to/session \
           --child_track 0 \
           --therapist_track therapist_merged

Or run the complete workflow:
       python mask_tool.py all --session_dir /path/to/session

Expected session structure
--------------------------
session/
├── MaskDir/
│   ├── 0.mp4
│   ├── 1.mp4
│   ├── 2.mp4
│   └── ...
├── VisualizationVideos/
│   └── visualization_a.mp4
├── *.txt
└── mask_decisions.json

Dependencies
------------
    pip install opencv-python numpy

Controls during inspection
--------------------------
SPACE       pause/resume
LEFT/RIGHT  frame-by-frame while paused
C           child
T           therapist
M           merge
D           delete/ignore
K           keep without role
N           next track
Q           quit and save
"""


import argparse
import bisect
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Annotation settings
# =============================================================================

TIER_INFO = {
    "ST":  ("joint",     "Session time"),
    "CP":  ("child",     "Position"),
    "CAO": ("child",     "Attending to objects"),
    "CAT": ("child",     "Attention to therapist"),
    "CG":  ("child",     "Gaze"),
    "CSP": ("child",     "Session pattern"),
    "CV":  ("child",     "Vocalization"),
    "TP":  ("therapist", "Position"),
    "TSP": ("therapist", "Session pattern"),
    "TV":  ("therapist", "Vocalization"),
    "JA":  ("joint",     "Joint eye contact"),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
LINE_HEIGHT = 22
PANEL_PADDING = 10
PANEL_ALPHA = 0.55
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
HEADER_COLOR = (0, 220, 255)


# =============================================================================
# Utility functions
# =============================================================================

def get_mask_dir(session_dir, mask_subdir="MaskDir"):
    return Path(session_dir) / mask_subdir


def get_decisions_path(session_dir, decisions_file=None):
    session_dir = Path(session_dir)

    if decisions_file:
        path = Path(decisions_file)
        if path.is_absolute():
            return path
        return session_dir / path

    return session_dir / "mask_decisions.json"


def discover_tracking_video(session_dir, explicit=None):
    session_dir = Path(session_dir)

    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = session_dir / path
        return path

    visualization_dir = session_dir / "VisualizationVideos"

    preferred = visualization_dir / "visualization_a.mp4"
    if preferred.exists():
        return preferred

    candidates = sorted(visualization_dir.glob("*.mp4"))

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No tracking video found in {visualization_dir}"
        )

    raise RuntimeError(
        "Multiple visualization videos found. "
        "Specify one with --tracking_video:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def discover_annotation_file(session_dir, pattern="*.txt"):
    session_dir = Path(session_dir)

    matches = sorted(session_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No annotation file matching '{pattern}' found in {session_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            "Multiple annotation files found. "
            "Specify --annotations_file or use --annotation_pattern:\n"
            + "\n".join(f"  {p}" for p in matches)
        )

    return matches[0]


def resolve_track_path(mask_dir, track_id):
    """
    Accept either:
        0
        0.mp4
        /absolute/path/to/0.mp4
    """
    path = Path(track_id)

    if path.is_absolute():
        return path

    if path.suffix.lower() == ".mp4":
        return mask_dir / path.name

    return mask_dir / f"{track_id}.mp4"


# =============================================================================
# Mask inspection
# =============================================================================

class MaskInspector:
    def __init__(self, mask_dir, output_config):
        self.mask_dir = Path(mask_dir)
        self.output_config = Path(output_config)

        self.masks = sorted(
            [
                f
                for f in self.mask_dir.glob("*.mp4")
                if f.name and f.name[0].isdigit()
            ]
        )

        self.decisions = self.load_existing_decisions()

    def load_existing_decisions(self):
        if not self.output_config.exists():
            return {}

        try:
            with open(self.output_config, "r", encoding="utf-8") as f:
                decisions = json.load(f)

            print(
                f"[info] Loaded {len(decisions)} existing decisions "
                f"from {self.output_config}"
            )
            return decisions

        except Exception as exc:
            print(f"[warn] Could not load existing decisions: {exc}")
            return {}

    def save_decisions(self):
        self.output_config.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_config, "w", encoding="utf-8") as f:
            json.dump(self.decisions, f, indent=2)

        print(f"\n[ok] Decisions saved to: {self.output_config}")

    def run(self):
        if not self.masks:
            print(f"[error] No numbered .mp4 masks found in {self.mask_dir}")
            return

        print("\n" + "=" * 70)
        print(f"Found {len(self.masks)} mask tracks:")
        for i, mask in enumerate(self.masks):
            decision = self.decisions.get(mask.stem, {})
            action = decision.get("action", "undecided")
            role = decision.get("role", "")
            print(f"  {i}: {mask.name:<25} [{action} {role}]")
        print("=" * 70)

        for idx, mask_path in enumerate(self.masks):
            result = self.play_mask_video(mask_path, idx)

            if result == "quit":
                break

        self.save_decisions()
        cv2.destroyAllWindows()

    def play_mask_video(self, mask_path, track_idx):
        cap = cv2.VideoCapture(str(mask_path))

        if not cap.isOpened():
            print(f"[error] Could not open {mask_path}")
            return "next"

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        track_id = mask_path.stem

        print("\n" + "=" * 70)
        print(f"[Track {track_idx}] {mask_path.name}")
        print(
            f"  Duration: {total_frames} frames @ "
            f"{fps:.1f} fps ({total_frames / fps:.1f} sec)"
        )
        print(f"  Resolution: {width}x{height}")
        print("\n  CONTROLS:")
        print("    SPACE       pause/resume")
        print("    LEFT/RIGHT  frame-by-frame")
        print("    C           child")
        print("    T           therapist")
        print("    K           keep / no role")
        print("    M           merge")
        print("    D           delete / ignore")
        print("    N           next without changing decision")
        print("    Q           quit and save")
        print("=" * 70)

        existing = self.decisions.get(track_id)

        if existing:
            print(
                f"[existing decision] "
                f"{existing.get('action')} "
                f"{existing.get('role', '')}"
            )

        frame_idx = 0
        paused = False

        window_name = f"Mask Inspector - {track_id}"

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

                if not ret:
                    print(f"[error] Could not read frames from {mask_path}")
                    cap.release()
                    return "next"

            display = frame.copy()

            time_sec = frame_idx / fps

            status = "PAUSED" if paused else "PLAYING"

            info = [
                (
                    f"Track: {track_id} | "
                    f"Frame: {frame_idx}/{max(total_frames - 1, 0)} | "
                    f"Time: {time_sec:.2f}s | {status}"
                ),
                "C=Child  T=Therapist  K=Keep  M=Merge  D=Delete  "
                "N=Next  Q=Quit",
            ]

            for i, text in enumerate(info):
                y = 30 + i * 25
                cv2.putText(
                    display,
                    text,
                    (10, y),
                    FONT,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, display)

            wait_ms = max(1, int(1000 / fps)) if not paused else 30
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                cap.release()
                return "quit"

            elif key == ord(" "):
                paused = not paused

            elif key == 83 or key == 255:
                # Windows/OpenCV sometimes reports special keys differently.
                # Arrow handling is also implemented through waitKeyEx below.
                pass

            elif key == ord("n"):
                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            elif key == ord("c"):
                self.decisions[track_id] = {
                    "action": "keep",
                    "role": "child",
                    "notes": "Assigned as child during mask inspection.",
                }

                print(f"[child] {track_id}")
                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            elif key == ord("t"):
                self.decisions[track_id] = {
                    "action": "keep",
                    "role": "therapist",
                    "notes": "Assigned as therapist during mask inspection.",
                }

                print(f"[therapist] {track_id}")
                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            elif key == ord("k"):
                self.decisions[track_id] = {
                    "action": "keep",
                    "role": "unknown",
                    "notes": "Kept without assigning child/therapist role.",
                }

                print(f"[keep] {track_id}")
                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            elif key == ord("m"):
                print(
                    f"\n[merge] {track_id} marked for merging."
                )
                print(
                    "Enter target track ID, e.g. 0, or press ENTER "
                    "to leave target unspecified."
                )

                target = input("Merge target: ").strip()

                self.decisions[track_id] = {
                    "action": "merge",
                    "merge_with": target if target else None,
                    "notes": "Marked for merge during inspection.",
                }

                print(
                    f"[merge] {track_id} -> "
                    f"{target if target else '(target not specified)'}"
                )

                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            elif key == ord("d"):
                self.decisions[track_id] = {
                    "action": "delete",
                    "reason": "Track marked for deletion/ignore.",
                    "notes": "",
                }

                print(f"[delete] {track_id}")
                cap.release()
                cv2.destroyWindow(window_name)
                return "next"

            # OpenCV arrow keys are easier to handle using waitKeyEx.
            # The main single-byte controls above continue to work normally.

            if not paused:
                frame_idx += 1

                if frame_idx >= total_frames:
                    frame_idx = 0

        cap.release()
        cv2.destroyWindow(window_name)
        return "next"


# =============================================================================
# Mask merging / applying decisions
# =============================================================================

def merge_mask_videos(video_paths, output_path, fps=None):
    """
    Merge mask videos using pixel-wise OR.

    Frames are processed sequentially instead of loading every frame
    from every video into memory.
    """

    video_paths = [Path(p) for p in video_paths]

    if not video_paths:
        raise ValueError("No videos supplied for merging.")

    print(f"  Merging {len(video_paths)} videos...")

    captures = []
    fps_ref = None
    width = None
    height = None

    for path in video_paths:
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            print(f"    [warn] Could not open {path}")
            continue

        if fps_ref is None:
            fps_ref = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        captures.append(cap)

    if not captures:
        print("    [error] None of the source videos could be opened.")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps_ref or fps or 30.0,
        (width, height),
        True,
    )

    if not writer.isOpened():
        for cap in captures:
            cap.release()
        print(f"    [error] Could not create {output_path}")
        return False

    frame_count = 0

    while True:
        merged = np.zeros((height, width), dtype=np.uint8)
        any_frame = False

        for cap in captures:
            ret, frame = cap.read()

            if not ret:
                continue

            any_frame = True

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if gray.shape[:2] != (height, width):
                gray = cv2.resize(
                    gray,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            merged = cv2.bitwise_or(merged, gray)

        if not any_frame:
            break

        bgr = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)

        frame_count += 1

    for cap in captures:
        cap.release()

    writer.release()

    print(
        f"    [ok] Merged video: {output_path} "
        f"({frame_count} frames)"
    )

    return True


def apply_decisions(
    session_dir,
    decisions_file=None,
    mask_subdir="MaskDir",
):
    session_dir = Path(session_dir)
    mask_dir = get_mask_dir(session_dir, mask_subdir)
    decisions_path = get_decisions_path(
        session_dir,
        decisions_file,
    )

    if not mask_dir.exists():
        print(f"[error] Mask directory not found: {mask_dir}")
        return False

    if not decisions_path.exists():
        print(f"[error] Decisions file not found: {decisions_path}")
        return False

    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = json.load(f)

    print(f"\n[info] Applying decisions from: {decisions_path}\n")

    to_merge = {}
    to_delete = []
    to_keep = {}

    for track_id, decision in decisions.items():
        action = decision.get("action")

        if action == "merge":
            target = decision.get("merge_with")

            if target:
                to_merge.setdefault(target, []).append(track_id)
                print(f"[merge] {track_id} -> {target}")
            else:
                print(
                    f"[warn] {track_id} is marked for merge "
                    "but has no merge_with target."
                )

        elif action == "delete":
            to_delete.append(track_id)
            print(f"[delete] {track_id}")

        elif action == "keep":
            role = decision.get("role", "unknown")
            to_keep[track_id] = role
            print(f"[keep] {track_id} ({role})")

    print("\n" + "=" * 70)

    # -------------------------------------------------------------------------
    # Apply merges
    # -------------------------------------------------------------------------

    for merge_target, source_ids in to_merge.items():
        print(f"\n[action] Merging into '{merge_target}':")

        target_path = resolve_track_path(mask_dir, merge_target)

        source_paths = []

        # Include target itself if it exists.
        if target_path.exists():
            source_paths.append(target_path)

        for source_id in source_ids:
            source_path = resolve_track_path(mask_dir, source_id)

            if source_path.exists():
                source_paths.append(source_path)
            else:
                print(
                    f"  [warn] Source track not found: "
                    f"{source_path}"
                )

        # Remove duplicates while preserving order.
        unique_paths = []
        seen = set()

        for path in source_paths:
            if path.resolve() not in seen:
                unique_paths.append(path)
                seen.add(path.resolve())

        if not unique_paths:
            print("  [warn] No source videos found.")
            continue

        output_path = mask_dir / f"{merge_target}_merged.mp4"

        merge_mask_videos(
            unique_paths,
            output_path,
        )

    # -------------------------------------------------------------------------
    # Apply deletes
    # -------------------------------------------------------------------------

    for track_id in to_delete:
        video_path = resolve_track_path(mask_dir, track_id)

        if not video_path.exists():
            print(
                f"[warn] Cannot delete {track_id}: "
                f"{video_path} does not exist."
            )
            continue

        backup_path = mask_dir / f"{video_path.stem}_DELETED.mp4"

        print(f"[action] Moving {video_path.name} -> {backup_path.name}")

        shutil.move(
            str(video_path),
            str(backup_path),
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("[summary] Final tracks:")

    for track_id, role in to_keep.items():
        path = resolve_track_path(mask_dir, track_id)

        if not path.exists():
            # A merged output may have been created.
            merged_path = mask_dir / f"{track_id}_merged.mp4"

            if merged_path.exists():
                path = merged_path
            else:
                print(
                    f"  [missing] {track_id} "
                    f"({role})"
                )
                continue

        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(
            f"  {path.name} -> {role} "
            f"({frames} frames)"
        )

    merged_files = sorted(mask_dir.glob("*_merged.mp4"))

    if merged_files:
        print("\n[merged outputs]")

        for path in merged_files:
            cap = cv2.VideoCapture(str(path))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"  {path.name} ({frames} frames)")

    print("\n[ok] Decisions applied.")

    return True


# =============================================================================
# Annotation parser
# =============================================================================

class TierTrack:
    def __init__(self):
        self.starts = []
        self.ends = []
        self.codes = []

    def add(self, start, end, code):
        self.starts.append(start)
        self.ends.append(end)
        self.codes.append(code)

    def finalize(self):
        order = sorted(
            range(len(self.starts)),
            key=lambda i: self.starts[i],
        )

        self.starts = [self.starts[i] for i in order]
        self.ends = [self.ends[i] for i in order]
        self.codes = [self.codes[i] for i in order]

    def get_code(self, t):
        idx = bisect.bisect_right(self.starts, t) - 1

        if idx < 0:
            return None

        if self.starts[idx] <= t <= self.ends[idx]:
            return self.codes[idx]

        return None


def parse_annotations(path, clip_id_filter=None):
    tiers = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 8:
                print(
                    f"[warn] line {lineno}: expected >=8 fields, "
                    f"got {len(parts)} -- skipped",
                    file=sys.stderr,
                )
                continue

            id_tier = parts[0]
            start_sec = parts[2]
            end_sec = parts[4]
            code = parts[7]

            if "*" not in id_tier:
                print(
                    f"[warn] line {lineno}: can't split clip/tier "
                    f"from '{id_tier}' -- skipped",
                    file=sys.stderr,
                )
                continue

            clip_id, tier_code = id_tier.rsplit("*", 1)

            if clip_id_filter and clip_id != clip_id_filter:
                continue

            try:
                start_f = float(start_sec)
                end_f = float(end_sec)
            except ValueError:
                print(
                    f"[warn] line {lineno}: bad time values -- skipped",
                    file=sys.stderr,
                )
                continue

            tiers.setdefault(
                tier_code,
                TierTrack(),
            ).add(
                start_f,
                end_f,
                code,
            )

    for track in tiers.values():
        track.finalize()

    return tiers


def build_panel_lines(tiers, t):
    panels = {
        "child": [],
        "therapist": [],
        "joint": [],
    }

    for tier_code, track in tiers.items():
        group, display_name = TIER_INFO.get(
            tier_code,
            ("joint", tier_code),
        )

        panels[group].append(
            (
                display_name,
                track.get_code(t),
            )
        )

    # Correct ordering by tier code rather than display name.
    order_index = {
        code: i
        for i, code in enumerate(TIER_INFO.keys())
    }

    for group in panels:
        panels[group].sort(
            key=lambda item: order_index.get(
                next(
                    (
                        code
                        for code, info in TIER_INFO.items()
                        if info[1] == item[0]
                    ),
                    "",
                ),
                999,
            )
        )

    return panels


# =============================================================================
# Annotation drawing
# =============================================================================

def get_mask_bbox(mask_frame):
    """
    Return (x, y, w, h) for the foreground region of a B/W mask.
    """

    if mask_frame is None:
        return None

    if mask_frame.ndim == 3:
        gray = cv2.cvtColor(
            mask_frame,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = mask_frame

    ys, xs = (gray > 127).nonzero()

    if len(xs) == 0:
        return None

    x = int(xs.min())
    y = int(ys.min())

    w = int(xs.max() - x)
    h = int(ys.max() - y)

    return x, y, w, h


def estimate_panel_size(header, lines):
    n_lines = len(lines) + 1

    panel_h = (
        n_lines * LINE_HEIGHT
        + PANEL_PADDING * 2
    )

    texts = [header]

    texts.extend(
        f"{name}: {code if code else ''}"
        for name, code in lines
    )

    max_w = max(
        cv2.getTextSize(
            text,
            FONT,
            FONT_SCALE,
            FONT_THICKNESS,
        )[0][0]
        for text in texts
    )

    panel_w = max_w + PANEL_PADDING * 2

    return panel_w, panel_h


def anchor_near_bbox(
    bbox,
    panel_w,
    panel_h,
    frame_w,
    frame_h,
):
    x, y, w, h = bbox

    cx = x + w // 2

    px = cx - panel_w // 2
    py = y - panel_h - 8

    if py < 0:
        py = y + h + 8

    px = max(
        4,
        min(px, frame_w - panel_w - 4),
    )

    py = max(
        4,
        min(py, frame_h - panel_h - 4),
    )

    return px, py


def draw_panel(frame, origin, header, lines):
    x, y = origin

    panel_w, panel_h = estimate_panel_size(
        header,
        lines,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_w, y + panel_h),
        BG_COLOR,
        -1,
    )

    cv2.addWeighted(
        overlay,
        PANEL_ALPHA,
        frame,
        1 - PANEL_ALPHA,
        0,
        frame,
    )

    ty = (
        y
        + PANEL_PADDING
        + LINE_HEIGHT
        - 6
    )

    cv2.putText(
        frame,
        header,
        (x + PANEL_PADDING, ty),
        FONT,
        FONT_SCALE,
        HEADER_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    for name, code in lines:
        ty += LINE_HEIGHT

        if code:
            cv2.putText(
                frame,
                f"{name}: {code}",
                (x + PANEL_PADDING, ty),
                FONT,
                FONT_SCALE,
                TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )


# =============================================================================
# Annotation generation
# =============================================================================

def annotate_video(
    session_dir,
    tracking_video=None,
    child_track=None,
    therapist_track=None,
    annotations_file=None,
    annotation_pattern="*.txt",
    clip_id_filter=None,
    output_video=None,
    time_offset_sec=0.0,
    mask_subdir="MaskDir",
):
    session_dir = Path(session_dir)
    mask_dir = get_mask_dir(
        session_dir,
        mask_subdir,
    )

    # -------------------------------------------------------------------------
    # Resolve input/output files
    # -------------------------------------------------------------------------

    tracking_path = discover_tracking_video(
        session_dir,
        tracking_video,
    )

    if annotations_file:
        annotations_path = Path(annotations_file)

        if not annotations_path.is_absolute():
            annotations_path = session_dir / annotations_path
    else:
        annotations_path = discover_annotation_file(
            session_dir,
            annotation_pattern,
        )

    if output_video:
        output_path = Path(output_video)

        if not output_path.is_absolute():
            output_path = session_dir / output_path
    else:
        output_path = (
            session_dir
            / "visualization_annotated.mp4"
        )

    if child_track:
        child_mask_path = resolve_track_path(
            mask_dir,
            child_track,
        )
    else:
        child_mask_path = None

    if therapist_track:
        therapist_mask_path = resolve_track_path(
            mask_dir,
            therapist_track,
        )
    else:
        therapist_mask_path = None

    # -------------------------------------------------------------------------
    # Parse annotations
    # -------------------------------------------------------------------------

    print(f"[info] Tracking video: {tracking_path}")
    print(f"[info] Annotation file: {annotations_path}")

    tiers = parse_annotations(
        annotations_path,
        clip_id_filter=clip_id_filter,
    )

    if not tiers:
        print(
            "[error] No tiers parsed from annotation file."
        )
        return False

    print(
        f"[info] Parsed tiers: "
        f"{list(tiers.keys())}"
    )

    # -------------------------------------------------------------------------
    # Open tracking video
    # -------------------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(tracking_path)
    )

    if not cap.isOpened():
        print(
            f"[error] Could not open "
            f"{tracking_path}"
        )
        return False

    fps = cap.get(
        cv2.CAP_PROP_FPS
    ) or 15.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # -------------------------------------------------------------------------
    # Open mask videos
    # -------------------------------------------------------------------------

    child_cap = None

    if child_mask_path:
        child_cap = cv2.VideoCapture(
            str(child_mask_path)
        )

        if not child_cap.isOpened():
            print(
                f"[warn] Could not open child mask "
                f"{child_mask_path}"
            )
            child_cap.release()
            child_cap = None
        else:
            print(
                f"[info] Child mask: "
                f"{child_mask_path}"
            )

    therapist_cap = None

    if therapist_mask_path:
        therapist_cap = cv2.VideoCapture(
            str(therapist_mask_path)
        )

        if not therapist_cap.isOpened():
            print(
                f"[warn] Could not open therapist mask "
                f"{therapist_mask_path}"
            )
            therapist_cap.release()
            therapist_cap = None
        else:
            print(
                f"[info] Therapist mask: "
                f"{therapist_mask_path}"
            )

    # -------------------------------------------------------------------------
    # Writer
    # -------------------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        if child_cap:
            child_cap.release()

        if therapist_cap:
            therapist_cap.release()

        print(
            f"[error] Could not create "
            f"{output_path}"
        )
        return False

    # -------------------------------------------------------------------------
    # Process frames
    # -------------------------------------------------------------------------

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        child_mask_frame = None

        if child_cap:
            ok, child_mask_frame = (
                child_cap.read()
            )

            if not ok:
                child_mask_frame = None

        therapist_mask_frame = None

        if therapist_cap:
            ok, therapist_mask_frame = (
                therapist_cap.read()
            )

            if not ok:
                therapist_mask_frame = None

        # Convert tracking-video frame number to annotation time.
        t = (
            frame_idx / fps
            + time_offset_sec
        )

        panels = build_panel_lines(
            tiers,
            t,
        )

        # ---------------------------------------------------------------------
        # Child panel
        # ---------------------------------------------------------------------

        if panels["child"]:
            bbox = get_mask_bbox(
                child_mask_frame
            )

            panel_w, panel_h = (
                estimate_panel_size(
                    "CHILD",
                    panels["child"],
                )
            )

            if bbox:
                origin = anchor_near_bbox(
                    bbox,
                    panel_w,
                    panel_h,
                    width,
                    height,
                )
            else:
                origin = (10, 10)

            draw_panel(
                frame,
                origin,
                "CHILD",
                panels["child"],
            )

        # ---------------------------------------------------------------------
        # Therapist panel
        # ---------------------------------------------------------------------

        if panels["therapist"]:
            bbox = get_mask_bbox(
                therapist_mask_frame
            )

            panel_w, panel_h = (
                estimate_panel_size(
                    "THERAPIST",
                    panels["therapist"],
                )
            )

            if bbox:
                origin = anchor_near_bbox(
                    bbox,
                    panel_w,
                    panel_h,
                    width,
                    height,
                )
            else:
                origin = (
                    width - panel_w - 10,
                    10,
                )

            draw_panel(
                frame,
                origin,
                "THERAPIST",
                panels["therapist"],
            )

        # ---------------------------------------------------------------------
        # Joint panel
        # ---------------------------------------------------------------------

        if panels["joint"]:
            panel_w, panel_h = (
                estimate_panel_size(
                    "JOINT",
                    panels["joint"],
                )
            )

            origin = (
                max(
                    10,
                    (width - panel_w) // 2,
                ),
                height - panel_h - 10,
            )

            draw_panel(
                frame,
                origin,
                "JOINT",
                panels["joint"],
            )

        writer.write(frame)

        frame_idx += 1

        if frame_idx % 500 == 0:
            print(
                f"[info] processed "
                f"{frame_idx}/{total_frames} frames"
            )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    cap.release()

    if child_cap:
        child_cap.release()

    if therapist_cap:
        therapist_cap.release()

    writer.release()

    print(
        f"[ok] Annotated video written to: "
        f"{output_path}"
    )

    return True


# =============================================================================
# Automatic role discovery from decisions
# =============================================================================

def get_role_tracks(
    session_dir,
    decisions_file=None,
    mask_subdir="MaskDir",
):
    decisions_path = get_decisions_path(
        session_dir,
        decisions_file,
    )

    if not decisions_path.exists():
        return None, None

    with open(
        decisions_path,
        "r",
        encoding="utf-8",
    ) as f:
        decisions = json.load(f)

    child_track = None
    therapist_track = None

    for track_id, decision in decisions.items():
        if decision.get("action") != "keep":
            continue

        role = decision.get("role")

        if role == "child":
            child_track = track_id

        elif role == "therapist":
            therapist_track = track_id

    return child_track, therapist_track


# =============================================================================
# CLI commands
# =============================================================================

def command_inspect(args):
    session_dir = Path(args.session_dir)

    mask_dir = get_mask_dir(
        session_dir,
        args.mask_subdir,
    )

    if not mask_dir.exists():
        print(
            f"[error] Mask directory not found: "
            f"{mask_dir}"
        )
        return 1

    decisions_path = get_decisions_path(
        session_dir,
        args.decisions,
    )

    inspector = MaskInspector(
        mask_dir,
        decisions_path,
    )

    inspector.run()

    return 0


def command_apply(args):
    success = apply_decisions(
        session_dir=args.session_dir,
        decisions_file=args.decisions,
        mask_subdir=args.mask_subdir,
    )

    return 0 if success else 1


def command_annotate(args):
    child_track = args.child_track
    therapist_track = args.therapist_track

    # If roles weren't explicitly supplied, try decisions JSON.
    if not child_track or not therapist_track:
        auto_child, auto_therapist = (
            get_role_tracks(
                args.session_dir,
                args.decisions,
                args.mask_subdir,
            )
        )

        child_track = (
            child_track or auto_child
        )

        therapist_track = (
            therapist_track
            or auto_therapist
        )

    if not child_track:
        print(
            "[warn] No child track specified "
            "or found in decisions."
        )

    if not therapist_track:
        print(
            "[warn] No therapist track specified "
            "or found in decisions."
        )

    success = annotate_video(
        session_dir=args.session_dir,
        tracking_video=args.tracking_video,
        child_track=child_track,
        therapist_track=therapist_track,
        annotations_file=args.annotations_file,
        annotation_pattern=args.annotation_pattern,
        clip_id_filter=args.clip_id,
        output_video=args.output_video,
        time_offset_sec=args.time_offset,
        mask_subdir=args.mask_subdir,
    )

    return 0 if success else 1


def command_all(args):
    """
    Run inspect -> apply -> annotate.

    The inspect stage is interactive. Once it finishes,
    decisions are immediately applied and the annotation
    video is generated.
    """

    print("\n" + "=" * 70)
    print("STEP 1/3 — INSPECT MASKS")
    print("=" * 70)

    result = command_inspect(args)

    if result != 0:
        return result

    print("\n" + "=" * 70)
    print("STEP 2/3 — APPLY DECISIONS")
    print("=" * 70)

    result = command_apply(args)

    if result != 0:
        return result

    print("\n" + "=" * 70)
    print("STEP 3/3 — CREATE ANNOTATED VIDEO")
    print("=" * 70)

    result = command_annotate(args)

    if result == 0:
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETE")
        print("=" * 70)

    return result


# =============================================================================
# Argument parser
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Unified mask inspection, decision application, "
            "and annotation tool."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------------------------------------------------------------
    # Shared arguments
    # -------------------------------------------------------------------------

    def add_shared_arguments(sub):
        sub.add_argument(
            "--session_dir",
            required=True,
            help="Session directory.",
        )

        sub.add_argument(
            "--mask_subdir",
            default="MaskDir",
            help="Mask subdirectory. Default: MaskDir",
        )

        sub.add_argument(
            "--decisions",
            default=None,
            help=(
                "Decision JSON filename/path. "
                "Default: mask_decisions.json inside session."
            ),
        )

    # -------------------------------------------------------------------------
    # inspect
    # -------------------------------------------------------------------------

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Interactively inspect mask tracks.",
    )

    add_shared_arguments(inspect_parser)

    inspect_parser.set_defaults(
        func=command_inspect
    )

    # -------------------------------------------------------------------------
    # apply
    # -------------------------------------------------------------------------

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply saved mask decisions.",
    )

    add_shared_arguments(apply_parser)

    apply_parser.set_defaults(
        func=command_apply
    )

    # -------------------------------------------------------------------------
    # annotate
    # -------------------------------------------------------------------------

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Create annotated visualization video.",
    )

    add_shared_arguments(annotate_parser)

    annotate_parser.add_argument(
        "--tracking_video",
        default=None,
        help=(
            "Tracking visualization video. "
            "Automatically discovered if omitted."
        ),
    )

    annotate_parser.add_argument(
        "--child_track",
        default=None,
        help=(
            "Child mask track, e.g. 0 or 0.mp4. "
            "If omitted, read role from decisions JSON."
        ),
    )

    annotate_parser.add_argument(
        "--therapist_track",
        default=None,
        help=(
            "Therapist mask track, e.g. 3 or therapist_merged. "
            "If omitted, read role from decisions JSON."
        ),
    )

    annotate_parser.add_argument(
        "--annotations_file",
        default=None,
        help=(
            "Annotation TXT file. "
            "Automatically discovered if omitted."
        ),
    )

    annotate_parser.add_argument(
        "--annotation_pattern",
        default="*.txt",
        help="Annotation file glob. Default: *.txt",
    )

    annotate_parser.add_argument(
        "--clip_id",
        default=None,
        help=(
            "Optional clip ID filter, e.g. c14."
        ),
    )

    annotate_parser.add_argument(
        "--output_video",
        default=None,
        help=(
            "Output video path. "
            "Default: visualization_annotated.mp4 "
            "inside session."
        ),
    )

    annotate_parser.add_argument(
        "--time_offset",
        type=float,
        default=0.0,
        help=(
            "Raw-video timestamp corresponding to "
            "frame 0 of the tracking video. Default: 0."
        ),
    )

    annotate_parser.set_defaults(
        func=command_annotate
    )

    # -------------------------------------------------------------------------
    # all
    # -------------------------------------------------------------------------

    all_parser = subparsers.add_parser(
        "all",
        help=(
            "Run inspect -> apply -> annotate."
        ),
    )

    add_shared_arguments(all_parser)

    all_parser.add_argument(
        "--tracking_video",
        default=None,
        help="Tracking visualization video.",
    )

    all_parser.add_argument(
        "--child_track",
        default=None,
        help="Child mask track.",
    )

    all_parser.add_argument(
        "--therapist_track",
        default=None,
        help="Therapist mask track.",
    )

    all_parser.add_argument(
        "--annotations_file",
        default=None,
        help="Annotation TXT file.",
    )

    all_parser.add_argument(
        "--annotation_pattern",
        default="*.txt",
        help="Annotation file glob.",
    )

    all_parser.add_argument(
        "--clip_id",
        default=None,
        help="Optional clip ID filter.",
    )

    all_parser.add_argument(
        "--output_video",
        default=None,
        help="Output annotated video.",
    )

    all_parser.add_argument(
        "--time_offset",
        type=float,
        default=0.0,
        help=(
            "Raw-video timestamp corresponding to "
            "frame 0 of tracking video."
        ),
    )

    all_parser.set_defaults(
        func=command_all
    )

    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        return args.func(args)

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        return 130

    except Exception as exc:
        print(
            f"\n[error] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())


