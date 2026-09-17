# coding=utf-8
"""
Convert ELAN .txt annotations to frame-level CSV for pipeline processing.

ELAN format (tab-separated):
    tier_id [tab] [empty] [tab] begin_hms [tab] begin_sec [tab] end_hms [tab]
    end_sec [tab] duration_hms [tab] duration_sec [tab] value

Example:
    c14_CP		00:02:47.571	167.571	00:18:38.787	1118.787	00:15:51.216	951.216	CST

Alignment model
---------------
ELAN times are expressed on the ORIGINAL recording. The pipeline works on a
trimmed / re-encoded video. Frame indices are therefore computed as:

    t_trimmed   = t_original - trim_start_s + annotation_offset_s
    frame_index = round(t_trimmed * fps_processed)

fps_processed is read from the processed video itself, not from the original
recording: if the two differ, using the original fps introduces a drift that
grows linearly with time.

Outputs
-------
    <output>.csv            one row per frame per active code
    <output>_intervals.csv  one row per annotation interval (for event
                            rate / mean duration statistics)
"""

import argparse
import json
import os
import re

import cv2
import pandas as pd


# Tier whose codes apply to both participants (joint attention / contact).
JOINT_TIER_PREFIXES = ("JA",)


def parse_elan_txt(txt_file):
    """
    Parse ELAN .txt output (tab-separated).

    Field positions vary between exports, so the two timestamps are located by
    picking the numeric fields rather than by fixed index: the hms columns do
    not parse as floats, the seconds columns do.

    Returns list of dicts: tier_id, begin_time_sec, end_time_sec, value.
    """
    rows = []
    n_skipped = 0

    with open(txt_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split("\t")]
            parts = [p for p in parts if p]

            if len(parts) < 4:
                n_skipped += 1
                print(f"Warning: line {lineno}: only {len(parts)} fields, skipped")
                continue

            tier_id = parts[0]
            value = parts[-1]

            numbers = []
            for p in parts[1:-1]:
                try:
                    numbers.append(float(p))
                except ValueError:
                    continue

            if len(numbers) < 2:
                n_skipped += 1
                print(f"Warning: line {lineno}: no timestamp pair found, skipped")
                continue

            begin_time_sec, end_time_sec = numbers[0], numbers[1]

            if end_time_sec < begin_time_sec:
                n_skipped += 1
                print(
                    f"Warning: line {lineno}: end {end_time_sec} < begin "
                    f"{begin_time_sec}, skipped"
                )
                continue

            rows.append({
                "tier_id": tier_id,
                "begin_time_sec": begin_time_sec,
                "end_time_sec": end_time_sec,
                "duration_sec": end_time_sec - begin_time_sec,
                "value": value,
            })

    if n_skipped:
        print(f"Skipped {n_skipped} malformed line(s)")

    return rows


def resolve_participants(tier_id, individual_id, therapist_pid=0):
    """
    Map a tier id to the participants its codes belong to.

    The TIER decides the participant, not the behaviour code. Codes are
    ambiguous across roles: TC ("builds/discusses together") and TCR ("crouch")
    are child codes on child tiers, so dispatching on the code's first letter
    misfiles them as therapist behaviour.

    Individual sessions:  CP, CAO, CAT, CG, CSP, CTCA, CV, CI -> child
                          TP, TSP, TV, TI                     -> therapist
                          JA                                  -> both
    Group sessions:       cX_*, tX_*  (X = participant number)

    Returns list of (pid, role) tuples, possibly empty.
    """
    tier_id = tier_id.strip()

    # Group-session tiers: cX_CP, tX_TI, cX_JA, ...
    match = re.match(r"^([ctCT])(\d+)_(.*)$", tier_id)
    if match:
        letter = match.group(1).lower()
        number = int(match.group(2))
        base = match.group(3).upper()
        child_pid = number if letter == "c" else individual_id

        if base.startswith(JOINT_TIER_PREFIXES):
            return [(child_pid, "child"), (therapist_pid, "therapist")]
        if letter == "c":
            return [(number, "child")]
        return [(therapist_pid, "therapist")]

    # Individual-session tiers: CP, CAO, TSP, JA, ...
    base = tier_id.upper()
    if base.startswith(JOINT_TIER_PREFIXES):
        return [(individual_id, "child"), (therapist_pid, "therapist")]
    if base.startswith("C"):
        return [(individual_id, "child")]
    if base.startswith("T"):
        return [(therapist_pid, "therapist")]

    return []


def load_processing_metadata(metadata_json):
    """
    Read trimming info from the pipeline metadata.

    Returns (fps_original, trim_start_s, trim_end_s). Any value may be None.
    """
    if not metadata_json or not os.path.exists(metadata_json):
        print(f"No metadata at {metadata_json}, assuming no trimming")
        return None, 0.0, None

    try:
        with open(metadata_json, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read metadata: {e}")
        return None, 0.0, None

    fps = meta.get("input_video", {}).get("fps")
    window = meta.get("time_window", {}) or {}
    trim_start_s = window.get("start_s", 0.0) or 0.0
    trim_end_s = window.get("end_s")

    print("Metadata:")
    print(f"  original fps : {fps}")
    print(f"  trim start   : {trim_start_s}s")
    print(f"  trim end     : {trim_end_s if trim_end_s is not None else 'n/a'}")

    return fps, float(trim_start_s), trim_end_s


def probe_video(video_path):
    """Return (fps, num_frames) of the video frame indices refer to."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps is None or fps <= 0:
        raise RuntimeError(f"Invalid FPS {fps} from video: {video_path}")
    return float(fps), num_frames


def main(args):
    print(f"ELAN annotation file : {args.elan_txt}")
    print(f"Processed video      : {args.video}")
    print(f"Individual ID        : {args.individual_id}")
    print(f"Session ID           : {args.session_id}")
    print(f"Output CSV           : {args.output}")
    print()

    annotations = parse_elan_txt(args.elan_txt)
    print(f"Parsed {len(annotations)} annotations from ELAN")
    if not annotations:
        print("ERROR: No annotations parsed. Check file format.")
        return

    # ---------------------------------------------------------------
    # Timing model
    # ---------------------------------------------------------------
    fps_meta, trim_start_s, trim_end_s = load_processing_metadata(args.sam3_metadata)
    fps_video, num_frames = probe_video(args.video)

    # fps of the video the frame indices refer to wins. Using the original
    # recording's fps here is the classic source of a drift that grows with
    # time and looks exactly like "the annotations slide out of sync".
    fps = float(args.fps) if args.fps else fps_video

    if fps_meta and abs(fps_meta - fps) > 1e-6:
        print(
            f"Note: original fps {fps_meta} != processed fps {fps}. "
            "Using the processed value; a mismatch here would drift by "
            f"{abs(fps_meta - fps):.3f} frames per second of video."
        )

    if args.trim_start_seconds is not None:
        trim_start_s = float(args.trim_start_seconds)
        print(f"Trim start overridden: {trim_start_s}s")

    offset_s = float(args.annotation_offset_ms) / 1000.0

    # Window covered by the processed video, in ORIGINAL recording seconds.
    window_start_s = trim_start_s
    window_end_s = trim_start_s + num_frames / fps
    if trim_end_s is not None:
        window_end_s = min(window_end_s, float(trim_end_s))

    print()
    print("Timing:")
    print(f"  fps (processed)   : {fps}")
    print(f"  frames            : {num_frames}")
    print(f"  annotation offset : {offset_s:+.3f}s")
    print(f"  covered window    : {window_start_s:.3f}s .. {window_end_s:.3f}s "
          f"(original timeline)")
    print()

    def to_frame(t_original):
        return int(round((t_original - trim_start_s + offset_s) * fps))

    # ---------------------------------------------------------------
    # Conversion
    # ---------------------------------------------------------------
    frame_rows = []
    interval_rows = []

    n_outside = 0
    n_clipped = 0
    n_no_participant = 0
    tier_counts = {}

    for ann in annotations:
        participants = resolve_participants(
            ann["tier_id"], args.individual_id, args.therapist_pid
        )
        if not participants:
            n_no_participant += 1
            print(f"Warning: tier '{ann['tier_id']}' maps to no participant, skipped")
            continue

        begin_frame = to_frame(ann["begin_time_sec"])
        end_frame = to_frame(ann["end_time_sec"])

        # Drop annotations that fall entirely outside the processed video.
        if end_frame < 0 or begin_frame > num_frames - 1:
            n_outside += 1
            continue

        # Clip the ones that straddle a boundary.
        clipped_begin = max(0, begin_frame)
        clipped_end = min(num_frames - 1, end_frame)
        if clipped_begin != begin_frame or clipped_end != end_frame:
            n_clipped += 1

        if clipped_end < clipped_begin:
            n_outside += 1
            continue

        code = ann["value"]
        tier_counts[ann["tier_id"]] = tier_counts.get(ann["tier_id"], 0) + 1

        for pid, role in participants:
            interval_rows.append({
                "tier_id": ann["tier_id"],
                "behavior_code": code,
                "pid": pid,
                "role": role,
                "begin_frame": clipped_begin,
                "end_frame": clipped_end,
                "begin_time_sec": clipped_begin / fps,
                "end_time_sec": clipped_end / fps,
                "duration_sec": (clipped_end - clipped_begin + 1) / fps,
                "begin_time_sec_original": ann["begin_time_sec"],
                "end_time_sec_original": ann["end_time_sec"],
            })

            for frame_idx in range(clipped_begin, clipped_end + 1):
                frame_rows.append({
                    "frame_index": frame_idx,
                    "frame_index_original": frame_idx + int(round(trim_start_s * fps)),
                    "time_sec": frame_idx / fps,
                    "time_sec_original": frame_idx / fps + trim_start_s,
                    "pid": pid,
                    "role": role,
                    "tier_id": ann["tier_id"],
                    "behavior_code": code,
                })

    if not frame_rows:
        print("ERROR: no annotation overlaps the processed video window.")
        print("Check --trim_start_seconds and that --video is the PROCESSED video.")
        return

    columns = [
        "frame_index",
        "frame_index_original",
        "time_sec",
        "time_sec_original",
        "pid",
        "role",
        "tier_id",
        "behavior_code",
    ]
    df = pd.DataFrame(frame_rows)[columns]
    df = df.drop_duplicates(subset=["frame_index", "pid", "behavior_code"])
    df = df.sort_values(["frame_index", "pid", "behavior_code"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    intervals_path = os.path.splitext(args.output)[0] + "_intervals.csv"
    pd.DataFrame(interval_rows).to_csv(intervals_path, index=False)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    covered = df["frame_index"].nunique()

    print("Conversion summary")
    print(f"  intervals kept        : {len(interval_rows)}")
    print(f"  intervals outside     : {n_outside}")
    print(f"  intervals clipped     : {n_clipped}")
    print(f"  tiers without pid     : {n_no_participant}")
    print(f"  frame rows            : {len(df)}")
    print(f"  frame range           : {df['frame_index'].min()} .. "
          f"{df['frame_index'].max()} (video has 0 .. {num_frames - 1})")
    print(f"  frames with any code  : {covered}/{num_frames} "
          f"({100.0 * covered / max(num_frames, 1):.1f}%)")
    print(f"  pids                  : {sorted(df['pid'].unique())}")
    print(f"  roles                 : {sorted(df['role'].unique())}")
    print(f"  behaviour codes       : {sorted(df['behavior_code'].unique())}")
    print()
    print("  per tier:")
    for tier in sorted(tier_counts):
        print(f"    {tier:<16} {tier_counts[tier]}")
    print()
    print(f"Saved: {args.output}")
    print(f"Saved: {intervals_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ELAN .txt annotations to frame-level ground truth CSV."
    )
    parser.add_argument(
        "--elan_txt",
        required=True,
        help="ELAN .txt annotation file",
    )
    parser.add_argument(
        "--video",
        required=True,
        help=(
            "The PROCESSED video that frame indices refer to. Its fps and frame "
            "count define the alignment window."
        ),
    )
    parser.add_argument(
        "--sam3_metadata",
        default=None,
        help="Pipeline metadata JSON (trim start/end)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the fps read from the processed video",
    )
    parser.add_argument(
        "--trim_start_seconds",
        type=float,
        default=None,
        help="Trim start in seconds (overrides metadata)",
    )
    parser.add_argument(
        "--annotation_offset_ms",
        type=float,
        default=0.0,
        help=(
            "Shift all annotations by this many milliseconds. Positive moves "
            "them later. Use for a known audio/video offset."
        ),
    )
    parser.add_argument(
        "--individual_id",
        type=int,
        required=True,
        help="Child / participant ID",
    )
    parser.add_argument(
        "--therapist_pid",
        type=int,
        default=0,
        help="Annotation pid used for the therapist (default 0)",
    )
    parser.add_argument(
        "--session_id",
        type=int,
        required=True,
        help="Session ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file",
    )
    args = parser.parse_args()
    main(args)
