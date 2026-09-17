#!/usr/bin/env python3

"""Overlay ELAN behaviour annotations on a tracked video.

Reads the CSV from elan_to_csv_converter.py, which carries `role` and
`tier_id`. The role of a code comes from the annotation itself, never from the
code's first letter: `TC` sits on both the child and the therapist
session-pattern tiers and means a different thing on each.

Draws a block of active codes beside each bounding box, and a bottom legend
with their definitions.
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from humantracker.utils import create_video_writer, video_capture, check_file

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX = ("xmin", "ymin", "xmax", "ymax")

# Tiers describing where attention or gaze is directed. In the individual
# session schema these are child-only, plus the shared JA tier: the therapist
# tiers cover position, session pattern, vocalisation and interaction.
ATTENTION_TIERS = {"CG", "CAO", "CAT", "JA"}
TIER_PRESETS = {"attention": ATTENTION_TIERS, "gaze": {"CG", "JA"}, "all": None}

# BGR, muted so the label text stays readable on top.
ROLE_COLORS = {
    "child": (80, 175, 76),
    "therapist": (0, 110, 230),
    "joint": (200, 100, 21),
    "unknown": (150, 150, 150),
}
HEAD_TINT = 60  # head box drawn lighter than the body box

# Definitions per tier, because the same code means different things on
# different tiers (CSP:TC vs TSP:TC, CI:T vs TI:C).
TIER_CODES = {
    "CP": {"CST": "standing", "CHO": "hovering over table", "CSI": "sitting",
           "CGO": "gone / not visible", "CLF": "on the floor",
           "CRE": "reaching", "CCR": "crouching"},
    "CAO": {"AO": "with session objects", "ANO": "with non-session objects",
            "AU": "undetermined"},
    "CAT": {"AT": "with the therapist"},
    "CG": {"GO": "gaze: session objects", "GT": "gaze: therapist",
           "GNO": "gaze: non-session objects", "GU": "gaze: undetermined"},
    "CSP": {"CP": "plays alone", "COB": "observes, does not join",
            "TC": "builds / discusses with therapist",
            "PL": "plays with therapist", "PRC": "preparing / cleaning up"},
    "CTCA": {"OBJ_EXCHANGE": "object exchange", "ARTMAKING": "art making",
             "SYMBOLIC_PLAY": "symbolic play",
             "COORDINATED_PLAY": "coordinated play",
             "CONVERSATION": "conversation",
             "CLEAN_UP_ACTIVITY": "clean-up activity"},
    "CV": {"CS": "speaking", "CNS": "non-speech sound"},
    "CI": {"T": "interacting with therapist", "T_V": "verbal only, no contact",
           "T_P": "physical only, no speech"},
    "TP": {"TST": "standing", "TSI": "sitting", "TCR": "crouching",
           "TGO": "gone / not visible"},
    "TSP": {"T": "creates alone", "TOB": "observes",
            "TC": "builds / discusses with child", "PL": "plays with child",
            "PRC": "preparing / cleaning up"},
    "TV": {"TS": "speaking", "TNS": "non-speech sound"},
    "TI": {"C": "interacting with child", "C_V": "verbal only, no contact",
           "C_P": "physical only, no speech"},
    "JA": {"TC": "joint attention / contact"},
}

# Used when the tier is unknown or missing from the table above.
CODE_FALLBACK = {
    "CST": "standing", "CSI": "sitting", "CCR": "crouching",
    "TST": "standing", "TSI": "sitting", "TCR": "crouching",
    "AO": "session objects", "ANO": "non-session objects", "AT": "therapist",
    "GO": "gaze: objects", "GT": "gaze: therapist", "GNO": "gaze: elsewhere",
    "CS": "speaking", "CNS": "non-speech sound",
    "TS": "speaking", "TNS": "non-speech sound",
    "PRC": "preparing / cleaning up",
}


def tier_suffix(tier_id):
    """c14_CSP -> CSP, CAO -> CAO."""
    if not isinstance(tier_id, str):
        return ""
    head, _, tail = tier_id.partition("_")
    if tail and head[:1].lower() in "ct" and head[1:].isdigit():
        return tail.upper()
    return tier_id.upper()


def describe(tier_id, code):
    return (TIER_CODES.get(tier_suffix(tier_id), {}).get(code)
            or CODE_FALLBACK.get(code, ""))


def box(row, prefix):
    """(x1, y1, x2, y2) from <prefix>_bbox_* columns, or None.

    Only a missing value disqualifies a box: a coordinate of 0 is a person at
    the frame edge, not a bad detection.
    """
    values = [row.get(f"{prefix}_bbox_{k}") for k in BOX]
    if any(pd.isna(v) for v in values):
        return None
    x1, y1, x2, y2 = (int(float(v)) for v in values)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_role_to_pid(path):
    """role -> tracking pid, from processing_log.json."""
    with open(path) as f:
        meta = json.load(f)

    block = meta.get("identity_mapping") or meta.get("participants")
    if not block:
        raise ValueError(f"No identity_mapping or participants in {path}")

    return {role: int(entry["tracking_pid"])
            for role in ("child", "therapist")
            for entry in [block.get(role) or {}]
            if "tracking_pid" in entry}


def load_annotations(path, tiers=None):
    """frame -> role -> sorted list of (tier_id, code)."""
    df = pd.read_csv(path)

    if "role" not in df.columns:
        raise ValueError(
            f"{path} has no 'role' column. Re-run elan_to_csv_converter.py; "
            "older CSVs cannot be attributed reliably."
        )
    df["tier_id"] = df.get("tier_id", "")

    if tiers:
        keep, before = set(tiers), len(df)
        df = df[df.tier_id.isin(keep) | df.tier_id.map(tier_suffix).isin(keep)]
        print(f"[info] tier filter kept {len(df)}/{before} rows")

    annotations = defaultdict(lambda: defaultdict(list))
    for frame, role, tier, code in zip(
        df.frame_index, df.role, df.tier_id, df.behavior_code
    ):
        bucket = annotations[int(frame)][str(role)]
        entry = (str(tier), str(code))
        if entry not in bucket:
            bucket.append(entry)

    for per_role in annotations.values():
        for bucket in per_role.values():
            bucket.sort()

    print(f"[info] annotations on {len(annotations)} frames, "
          f"roles: {sorted(df.role.unique())}")
    return annotations


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def put(img, text, org, scale, color, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_label_block(frame, lines, bbox, color, side="right",
                     scale=0.45, alpha=0.55, gap=6):
    """Translucent block beside a bounding box, clamped inside the frame.

    side: "right" / "left" pin it; "auto" prefers the right and falls back.
    """
    if not lines:
        return

    h, w = frame.shape[:2]
    line_h = int(round(20 * scale / 0.45))
    pad = 4
    box_w = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t in lines) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad

    x1, y1, x2, _ = bbox
    right, left = x2 + gap, x1 - gap - box_w
    if side == "left":
        x = left
    elif side == "auto" and right + box_w > w and left >= 0:
        x = left
    else:
        x = right

    x = max(0, min(int(x), w - box_w))
    y = max(0, min(int(y1), h - box_h))

    roi = frame[y:y + box_h, x:x + box_w]
    if roi.size == 0:
        return
    cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), color, 1)

    for i, text in enumerate(lines):
        put(frame, text, (x + pad, y + pad + line_h * (i + 1) - 5), scale,
            (245, 245, 245))


def draw_legend(canvas, frame_annotations, height, width, legend_height,
                show_tier=True):
    """One column per role, code + definition."""
    scale, line_h, margin = 0.48, 19, 15

    roles = [r for r in ("child", "therapist", "joint") if frame_annotations.get(r)]
    roles += [r for r in sorted(frame_annotations) if r not in roles]
    if not roles:
        return

    col_w = width // len(roles)
    y_start = height + 20

    for i, role in enumerate(roles):
        x = margin + i * col_w
        put(canvas, role.upper(), (x, y_start), scale + 0.12,
            ROLE_COLORS.get(role, ROLE_COLORS["unknown"]), 2)

        y = y_start + line_h + 3
        for tier, code in frame_annotations[role]:
            if y > height + legend_height - 4:
                put(canvas, "...", (x, y), scale, (170, 170, 170))
                break
            head = f"{tier_suffix(tier)}:{code}" if show_tier else code
            definition = describe(tier, code)
            put(canvas, f"{head}  {definition}" if definition else head,
                (x, y), scale, (225, 225, 225))
            y += line_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_video(video_file, annotations_file, tracking_data_file,
                  metadata_file, output_file, tiers=None,
                  label_tiers=ATTENTION_TIERS, label_side="right",
                  show_legend=True, show_labels=True, show_tier=True,
                  legend_height=180, max_frames=None, draw_unknown=False):

    role_to_pid = load_role_to_pid(metadata_file)
    pid_to_role = {pid: role for role, pid in role_to_pid.items()}
    print("[info] participant mapping:")
    for role, pid in role_to_pid.items():
        print(f"  {role:<10} -> tracking pid {pid}")

    annotations = load_annotations(annotations_file, tiers=tiers)
    df_tracking = pd.read_csv(tracking_data_file)

    if show_labels and label_tiers is not None:
        print(f"[info] bbox labels limited to tiers: {sorted(label_tiers)}")
        labelled = {role
                    for per_role in annotations.values()
                    for role, codes in per_role.items()
                    if any(tier_suffix(t) in label_tiers for t, _ in codes)}
        silent = sorted(set(role_to_pid) - labelled)
        if silent:
            print(f"[warn] no tier in that set belongs to: {', '.join(silent)}"
                  " - that participant will have an empty label block")

    cap, fps, width, height, num_frames = video_capture(video_file)
    fps = 15.0 if fps <= 0 else fps
    if max_frames:
        num_frames = min(num_frames, int(max_frames))

    output_height = height + legend_height if show_legend else height
    writer = create_video_writer(output_file, fps, width, output_height)

    has_head = "head_bbox_xmin" in df_tracking.columns
    unknown_pids = set()

    for frame_index in tqdm(range(num_frames), desc="Annotating video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_annotations = annotations.get(frame_index, {})

        for _, row in df_tracking[df_tracking.frame_index == frame_index].iterrows():
            pid = int(row["pid"])
            role = pid_to_role.get(pid)
            if role is None:
                unknown_pids.add(pid)
                if not draw_unknown:
                    continue
                role = "unknown"

            body = box(row, "body")
            if body is None:
                continue

            color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
            cv2.rectangle(frame, body[:2], body[2:], color, 2)

            if has_head:
                try:
                    confidence = float(row.get("head_confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                head = box(row, "head") if confidence > 0.3 else None
                if head:
                    light = tuple(min(255, c + HEAD_TINT) for c in color)
                    cv2.rectangle(frame, head[:2], head[2:], light, 2)

            if show_labels:
                codes = frame_annotations.get(role, [])
                if label_tiers is not None:
                    codes = [(t, c) for t, c in codes
                             if tier_suffix(t) in label_tiers]
                lines = [f"{role} #{pid}"]
                lines += [f"{tier_suffix(t)}:{c}" if show_tier else c
                          for t, c in codes]
                draw_label_block(frame, lines, body, color, side=label_side)

        if show_legend:
            canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
            canvas[:height] = frame
            draw_legend(canvas, frame_annotations, height, width,
                        legend_height, show_tier=show_tier)
            frame = canvas

        writer.write(frame)

    cap.release()
    writer.release()

    if unknown_pids:
        action = "drawn as unknown" if draw_unknown else "skipped"
        print(f"[warn] tracking pids with no role in the metadata: "
              f"{sorted(unknown_pids)} ({action})")
    print(f"[ok] annotated video: {output_file}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay ELAN annotations on a tracked video")
    p.add_argument("--video", required=True)
    p.add_argument("--annotations", required=True,
                   help="Annotations CSV from elan_to_csv_converter.py")
    p.add_argument("--body-detections", required=True, help="Tracking CSV")
    p.add_argument("--metadata", required=True, help="processing_log.json")
    p.add_argument("--output", required=True)
    p.add_argument("--tiers", default=None,
                   help="Tiers to load at all, e.g. CG,CV,JA. Affects both the "
                        "labels and the legend.")
    p.add_argument("--label-tiers", default="attention",
                   help="Tiers in the block beside each bbox: a preset "
                        "(attention, gaze, all) or a tier list. The legend is "
                        "not affected - use --tiers for that.")
    p.add_argument("--label-side", default="right",
                   choices=("right", "left", "auto"))
    p.add_argument("--no-legend", action="store_true")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--no-tier", action="store_true",
                   help="Show bare codes instead of TIER:CODE")
    p.add_argument("--legend-height", type=int, default=180)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--draw-unknown", action="store_true",
                   help="Also draw tracks with no role in the metadata")
    return p.parse_args()


def split_tiers(value):
    return {t.strip().upper() for t in value.split(",") if t.strip()} or None


def main():
    args = parse_args()

    for path in (args.video, args.annotations, args.body_detections,
                 args.metadata):
        check_file(path)

    preset = args.label_tiers.strip().lower()
    label_tiers = (TIER_PRESETS[preset] if preset in TIER_PRESETS
                   else split_tiers(args.label_tiers))

    process_video(
        args.video, args.annotations, args.body_detections, args.metadata,
        args.output,
        tiers=split_tiers(args.tiers) if args.tiers else None,
        label_tiers=label_tiers,
        label_side=args.label_side,
        show_legend=not args.no_legend,
        show_labels=not args.no_labels,
        show_tier=not args.no_tier,
        legend_height=args.legend_height,
        max_frames=args.max_frames,
        draw_unknown=args.draw_unknown,
    )


if __name__ == "__main__":
    main()
