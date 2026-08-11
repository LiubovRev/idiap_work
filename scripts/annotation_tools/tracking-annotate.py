#!/usr/bin/env python3
"""
annotate_tracking_video.py

Overlays time-coded tier annotation labels (child / therapist / joint
panels) onto an EXISTING tracking/mask visualization video -- no raw video,
trimming, or mask extraction needed.

If the tracking video was itself trimmed from a longer raw video (so frame 0
of the tracking video does NOT correspond to time 0:00 of the annotation
txt's timestamps), set `time_offset_sec` below to the raw-video time (in
seconds) that frame 0 corresponds to.
"""

import sys
import bisect
from pathlib import Path

import cv2

# ============================== Settings ====================================
base_path = Path("/home/lrevutska/Documents/chuv_machine_downloads/videos/7_INDIVIDUAL_14")
visualization_dir = base_path / "VisualizationVideos"
tracking_video = visualization_dir / "visualization_a.mp4"   # <-- existing tracking/mask video
mask_dir = base_path / "MaskDir"
child_mask_video = mask_dir / "0.mp4"        # <-- CHECK: set to whichever object number is the child
therapist_mask_video = mask_dir / "therapist_merged.mp4"    # <-- CHECK: set to whichever object number is the therapist
annotations_txt_pattern = "*.txt"                             # glob resolved against base_path
clip_id_filter = None                                          # e.g. "c14" if the txt mixes clips
output_video = base_path / "visualization_annotated.mp4"

time_offset_sec = 167.0   # <-- SET THIS: raw-video time (sec) that frame 0 of tracking_video corresponds to

# tier_code -> (panel_group, display_name); unknown tiers fall back to "joint"
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


class TierTrack:
    def __init__(self):
        self.starts, self.ends, self.codes = [], [], []

    def add(self, start, end, code):
        self.starts.append(start)
        self.ends.append(end)
        self.codes.append(code)

    def finalize(self):
        order = sorted(range(len(self.starts)), key=lambda i: self.starts[i])
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


def resolve_annotations_txt():
    matches = sorted(base_path.glob(annotations_txt_pattern))
    if not matches:
        print(f"[error] no file matching '{annotations_txt_pattern}' found in {base_path}")
        sys.exit(1)
    if len(matches) > 1:
        print(f"[error] multiple files match '{annotations_txt_pattern}' in {base_path}:")
        for m in matches:
            print(f"    {m}")
        print("Narrow the pattern (e.g. 'c14*.txt') or set annotations_txt_pattern to an exact filename.")
        sys.exit(1)
    print(f"[info] using annotations file: {matches[0]}")
    return matches[0]


def parse_annotations(path, clip_id_filter=None):
    tiers = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                print(f"[warn] line {lineno}: expected >=8 fields, got {len(parts)} -- skipped", file=sys.stderr)
                continue
            id_tier, start_sec, end_sec, code = parts[0], parts[2], parts[4], parts[7]
            if "_" not in id_tier:
                print(f"[warn] line {lineno}: can't split clip/tier from '{id_tier}' -- skipped", file=sys.stderr)
                continue
            clip_id, tier_code = id_tier.rsplit("_", 1)
            if clip_id_filter and clip_id != clip_id_filter:
                continue
            try:
                start_f, end_f = float(start_sec), float(end_sec)
            except ValueError:
                print(f"[warn] line {lineno}: bad time values -- skipped", file=sys.stderr)
                continue
            tiers.setdefault(tier_code, TierTrack()).add(start_f, end_f, code)
    for track in tiers.values():
        track.finalize()
    return tiers


def build_panel_lines(tiers, t):
    panels = {"child": [], "therapist": [], "joint": []}
    for tier_code, track in tiers.items():
        group, display_name = TIER_INFO.get(tier_code, ("joint", tier_code))
        panels[group].append((display_name, track.get_code(t)))
    order_index = {code: i for i, code in enumerate(TIER_INFO.keys())}
    for group in panels:
        panels[group].sort(key=lambda item: order_index.get(item[0], 999))
    return panels


def get_mask_bbox(mask_frame):
    """Returns (x, y, w, h) of the foreground region in a B/W mask frame, or None if empty."""
    if mask_frame is None:
        return None
    gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if mask_frame.ndim == 3 else mask_frame
    ys, xs = (gray > 127).nonzero()
    if len(xs) == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x), int(ys.max() - y)
    return x, y, w, h


def estimate_panel_size(header, lines):
    n_lines = len(lines) + 1
    panel_h = n_lines * LINE_HEIGHT + PANEL_PADDING * 2
    texts = [header] + [f"{name}: {code if code else ''}" for name, code in lines]
    max_w = max(cv2.getTextSize(t, FONT, FONT_SCALE, FONT_THICKNESS)[0][0] for t in texts)
    panel_w = max_w + PANEL_PADDING * 2
    return panel_w, panel_h


def anchor_near_bbox(bbox, panel_w, panel_h, frame_w, frame_h):
    """Places the panel just above the bbox (or below, if too close to the top),
    horizontally centered on the bbox, clamped to stay fully on-screen."""
    x, y, w, h = bbox
    cx = x + w // 2
    px = cx - panel_w // 2
    py = y - panel_h - 8
    if py < 0:
        py = y + h + 8  # not enough room above -> place below the person instead
    px = max(4, min(px, frame_w - panel_w - 4))
    py = max(4, min(py, frame_h - panel_h - 4))
    return px, py


def draw_panel(frame, origin, header, lines):
    x, y = origin
    panel_w, panel_h = estimate_panel_size(header, lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), BG_COLOR, -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)
    ty = y + PANEL_PADDING + LINE_HEIGHT - 6
    cv2.putText(frame, header, (x + PANEL_PADDING, ty), FONT, FONT_SCALE, HEADER_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    for name, code in lines:
        ty += LINE_HEIGHT
        if code:
            cv2.putText(frame, f"{name}: {code}", (x + PANEL_PADDING, ty), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def main():
    annotations_txt = resolve_annotations_txt()
    tiers = parse_annotations(str(annotations_txt), clip_id_filter=clip_id_filter)
    if not tiers:
        print("[error] no tiers parsed from annotation file -- check format / clip_id_filter")
        sys.exit(1)
    print(f"[info] parsed tiers: {list(tiers.keys())}")

    cap = cv2.VideoCapture(str(tracking_video))
    if not cap.isOpened():
        print(f"[error] could not open {tracking_video}")
        sys.exit(1)

    child_cap = cv2.VideoCapture(str(child_mask_video))
    therapist_cap = cv2.VideoCapture(str(therapist_mask_video))
    if not child_cap.isOpened():
        print(f"[warn] could not open child mask video {child_mask_video} -- child panel will use a fixed position")
        child_cap = None
    if not therapist_cap.isOpened():
        print(f"[warn] could not open therapist mask video {therapist_mask_video} -- therapist panel will use a fixed position")
        therapist_cap = None

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        child_mask_frame = None
        if child_cap is not None:
            ok, child_mask_frame = child_cap.read()
            if not ok:
                child_mask_frame = None
        therapist_mask_frame = None
        if therapist_cap is not None:
            ok, therapist_mask_frame = therapist_cap.read()
            if not ok:
                therapist_mask_frame = None

        t = frame_idx / fps + time_offset_sec
        panels = build_panel_lines(tiers, t)

        if panels["child"]:
            bbox = get_mask_bbox(child_mask_frame)
            panel_w, panel_h = estimate_panel_size("CHILD", panels["child"])
            origin = anchor_near_bbox(bbox, panel_w, panel_h, width, height) if bbox else (10, 10)
            draw_panel(frame, origin, "CHILD", panels["child"])
        if panels["therapist"]:
            bbox = get_mask_bbox(therapist_mask_frame)
            panel_w, panel_h = estimate_panel_size("THERAPIST", panels["therapist"])
            origin = anchor_near_bbox(bbox, panel_w, panel_h, width, height) if bbox else (width - panel_w - 10, 10)
            draw_panel(frame, origin, "THERAPIST", panels["therapist"])
        if panels["joint"]:
            panel_w, panel_h = estimate_panel_size("JOINT", panels["joint"])
            origin = (max(10, (width - panel_w) // 2), height - panel_h - 10)
            draw_panel(frame, origin, "JOINT", panels["joint"])

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"[info] processed {frame_idx}/{total_frames} frames")

    cap.release()
    if child_cap is not None:
        child_cap.release()
    if therapist_cap is not None:
        therapist_cap.release()
    writer.release()
    print(f"[ok] Annotated video written: {output_video}")


if __name__ == "__main__":
    main()
