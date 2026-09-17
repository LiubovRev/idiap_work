# coding=utf-8

"""
SAM3 mask videos -> body bounding boxes.

Pipeline per frame
------------------
1. Binarise at mask_threshold.
   Masks are stored as lossily compressed mp4, so a mask written with values
   {0, 255} comes back carrying grey levels 1-11 around every edge and along
   the frame borders. A ``> 0`` test accepts all of it: it inflates every box
   and promotes compression noise into components large enough to pass
   min_area. Any threshold between the noise floor and 255 works.

2. Close gaps of up to close_px.
   SAM3 masks fragment: a leg or an arm separates from the torso, each piece
   falls below min_area, and the absolute area filter then deletes part of the
   body. Closing reconnects those pieces before anything is measured.

3. Drop components below min_area.

4. Drop border strips (the intersection criterion).
   A band of width band_px is laid along every image border and intersected
   with each component. What matters is the fraction of the component that
   falls inside that band:

       ratio = (component & band).area / component.area

   A strip lying along a border is contained in the band and scores 1.0; a
   person scores 0.0 unless they stand right against the edge. A component is
   an edge artifact when it is almost entirely inside the band and negligible
   with respect to the image:

       ratio >= min_band_ratio  and  area / (H * W) < max_image_ratio

   The band is never erased from the mask, so a person at the edge keeps the
   full extent of theirs. No reference person is involved, so a frame holding
   nothing but streaks yields no box at all.

   Measuring the fraction rather than the component's width across the frame
   is what keeps the decision stable: a component either lies in the band or
   it does not, and flipping the verdict takes moving half of it rather than a
   few pixels at its edge. A width threshold sitting at 40 px flipped between
   adjacent frames on values like 44 and 39.

5. Union of the survivors becomes the box, so a body split across several
   blobs is not truncated. Only components that overlap the main one
   horizontally are merged (min_x_overlap): an occluder splits a body
   top/bottom and the pieces keep a shared x range, whereas two separate
   objects sit side by side. The size of the gap does not separate the two
   cases; the axis of the split does.

Known limitation: a thin diagonal streak running from a corner reaches far
past the band, so only part of it lands inside and it is never classed as a
strip. On the frames checked it scored 0.43 against 1.00 for real strips and
0.00 for people.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


CSV_COLUMNS = [
    "frame_index", "pid",
    "body_bbox_xmin", "body_bbox_ymin", "body_bbox_xmax", "body_bbox_ymax",
    "num_components", "num_kept", "num_artifacts", "num_small", "num_unmerged",
    "artifact_area", "main_area", "touches_border",
    "video_name", "num_frames", "frame_width", "frame_height", "fps",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def border_band(height, width, band_px):
    """Boolean band of width band_px along all four image borders."""
    b = max(1, min(int(band_px), height // 2, width // 2)) if band_px > 0 else 0
    band = np.zeros((height, width), dtype=bool)
    if b:
        band[:b, :] = band[-b:, :] = True
        band[:, :b] = band[:, -b:] = True
    return band


def band_ratios(labels, num_labels, stats, band):
    """Fraction of each component's pixels that falls inside the band."""
    inside = np.bincount(labels[band].ravel(), minlength=num_labels).astype(float)
    areas = stats[:num_labels, cv2.CC_STAT_AREA].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(areas > 0, inside / np.maximum(areas, 1), 0.0)


def union_bbox(stats, labels_list):
    """Box covering every listed component."""
    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in labels_list)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in labels_list)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] - 1
             for i in labels_list)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] - 1
             for i in labels_list)
    return float(x0), float(y0), float(x1), float(y1)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #

def mask_to_bbox(
    mask,
    mask_threshold=127,
    close_px=9,
    min_area=1000,
    band_px=40,
    min_band_ratio=0.9,
    max_image_ratio=0.02,
    min_x_overlap=0.2,
):
    """
    Body box for one mask frame.

    Returns (bbox | None, info, debug | None) where debug carries
    (labels, stats, kept, artifacts, small, ratios).
    """
    binary = (mask > mask_threshold).astype(np.uint8)

    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    height, width = binary.shape
    image_area = float(height * width)

    info = {
        "num_components": 0, "num_kept": 0, "num_artifacts": 0,
        "num_small": 0, "num_unmerged": 0, "artifact_area": 0,
        "main_area": 0, "touches_border": False,
    }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return None, info, None

    band = border_band(height, width, band_px)
    ratios = band_ratios(labels, num_labels, stats, band)

    # --- filter 1: absolute area -------------------------------------------
    candidates, small = [], []
    for i in range(1, num_labels):
        (candidates if stats[i, cv2.CC_STAT_AREA] >= min_area else small).append(i)

    info["num_components"] = len(candidates)
    info["num_small"] = len(small)

    if not candidates:
        return None, info, (labels, stats, [], [], small, ratios)

    # --- filter 2: border strips -------------------------------------------
    kept, artifacts = [], []
    for i in candidates:
        in_band = ratios[i] >= min_band_ratio
        tiny = stats[i, cv2.CC_STAT_AREA] / image_area < max_image_ratio
        (artifacts if (in_band and tiny) else kept).append(i)

    info["num_artifacts"] = len(artifacts)
    info["artifact_area"] = int(sum(stats[i, cv2.CC_STAT_AREA] for i in artifacts))

    if not kept:
        return None, info, (labels, stats, [], artifacts, small, ratios)

    main = max(kept, key=lambda i: stats[i, cv2.CC_STAT_AREA])

    # Merge only what lines up vertically with the main component. A body cut
    # by an occluder splits top/bottom and keeps a shared x range; two separate
    # objects sit side by side and share none. Gap size cannot tell them apart
    # (43 px between two objects vs 53 px between a torso and its legs), the
    # axis of the split can.
    if min_x_overlap > 0 and len(kept) > 1:
        mx0 = int(stats[main, cv2.CC_STAT_LEFT])
        mx1 = mx0 + int(stats[main, cv2.CC_STAT_WIDTH]) - 1
        merged, unmerged = [main], []
        for i in kept:
            if i == main:
                continue
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            x1 = x0 + int(stats[i, cv2.CC_STAT_WIDTH]) - 1
            overlap = max(0, min(mx1, x1) - max(mx0, x0) + 1)
            narrower = min(mx1 - mx0 + 1, x1 - x0 + 1)
            (merged if overlap >= min_x_overlap * narrower else unmerged).append(i)
        info["num_unmerged"] = len(unmerged)
        kept = merged

    info["num_kept"] = len(kept)
    info["main_area"] = int(stats[main, cv2.CC_STAT_AREA])
    info["touches_border"] = bool(any(ratios[i] > 0 for i in kept))

    return union_bbox(stats, kept), info, (labels, stats, kept, artifacts, small, ratios)


# --------------------------------------------------------------------------- #
# debug rendering
# --------------------------------------------------------------------------- #

def draw_debug(shape, debug, bbox, band_px, params_line):
    """green kept | red artifact | blue below min_area | yellow box | cyan band edge"""
    labels, stats, kept, artifacts, small, ratios = debug
    canvas = np.zeros((*shape, 3), np.uint8)
    height, width = shape

    for group, color, labelled in ((small, (200, 90, 0), False),
                                   (artifacts, (0, 0, 255), True),
                                   (kept, (0, 255, 0), True)):
        for i in group:
            canvas[labels == i] = color
            if not labelled:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            ratio = ratios[i]
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 1)
            cv2.putText(canvas, f"A={int(stats[i, cv2.CC_STAT_AREA])} band={ratio:.2f}",
                        (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1, cv2.LINE_AA)

    b = max(1, min(int(band_px), height // 2, width // 2)) if band_px > 0 else 0
    if b:
        cv2.rectangle(canvas, (b - 1, b - 1), (width - b, height - b), (255, 255, 0), 1)

    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)

    for n, line in enumerate(("green kept | red artifact | blue < min_area | yellow bbox",
                              params_line)):
        cv2.putText(canvas, line, (8, 16 + 16 * n), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# per-video driver
# --------------------------------------------------------------------------- #

def process_mask(mask_file, pid, params, save_dir=None, max_debug_frames=None):
    cap = cv2.VideoCapture(str(mask_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open mask video: {mask_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    params_line = (f"thr>{params['mask_threshold']} close={params['close_px']} "
                   f"min_area={params['min_area']} band={params['band_px']}"
                   f"/{params['min_band_ratio']}")

    rows, frame_index, saved, with_artifacts = [], 0, 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bbox, info, debug = mask_to_bbox(gray, **params)

        if info["num_artifacts"]:
            with_artifacts += 1

        interesting = info["num_artifacts"] or info["num_kept"] > 1 or bbox is None
        budget = max_debug_frames is None or saved < max_debug_frames

        if interesting and debug is not None and save_dir is not None and budget:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"pid_{pid}_frame_{frame_index:06d}.png"),
                        draw_debug(gray.shape, debug, bbox, params["band_px"], params_line))
            saved += 1

        if bbox is not None:
            rows.append({
                "frame_index": frame_index, "pid": pid,
                "body_bbox_xmin": bbox[0], "body_bbox_ymin": bbox[1],
                "body_bbox_xmax": bbox[2], "body_bbox_ymax": bbox[3],
                **{k: info[k] for k in ("num_components", "num_kept", "num_artifacts",
                                        "num_small", "num_unmerged", "artifact_area",
                                        "main_area", "touches_border")},
            })

        frame_index += 1

    cap.release()

    for row in rows:
        row.update(video_name=mask_file.name, num_frames=frame_index,
                   frame_width=width, frame_height=height, fps=fps)

    return rows, frame_index, saved, with_artifacts


# --------------------------------------------------------------------------- #

def main(args):
    mask_dir = Path(args.mask_dir)
    mask_files = sorted(
        (f for f in mask_dir.iterdir()
         if f.suffix.lower() in (".mp4", ".avi", ".mkv") and f.stem.isdigit()),
        key=lambda f: int(f.stem),
    )
    if not mask_files:
        raise RuntimeError(f"No numerically named mask videos in {mask_dir}")

    params = dict(
        mask_threshold=args.mask_threshold,
        close_px=args.close_px,
        min_area=args.min_area,
        band_px=args.band_px,
        min_band_ratio=args.min_band_ratio,
        max_image_ratio=args.max_image_ratio,
        min_x_overlap=args.min_x_overlap,
    )

    print(f"Mask directory : {mask_dir}  ({len(mask_files)} files)")
    for k, v in params.items():
        print(f"  {k:18s} {v}")
    if args.save_dir:
        print(f"  debug frames       {args.save_dir}")
    print()

    save_dir = Path(args.save_dir) if args.save_dir else None
    all_rows = []

    for mask_file in mask_files:
        pid = int(mask_file.stem)
        rows, n_frames, saved, with_artifacts = process_mask(
            mask_file, pid, params, save_dir, args.max_debug_frames
        )
        all_rows.extend(rows)
        print(f"PID {pid} ({mask_file.name})")
        print(f"  frames {n_frames} | bboxes {len(rows)} ({len(rows)/max(n_frames,1):.1%})")
        print(f"  frames with artifacts removed {with_artifacts} "
              f"({with_artifacts/max(n_frames,1):.1%}) | debug saved {saved}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df[CSV_COLUMNS]
    df.to_csv(out, index=False, float_format="%.2f")

    print(f"\nSaved {len(df)} rows to {out}")
    if not df.empty:
        print(f"  PIDs {sorted(df.pid.unique().tolist())} | "
              f"boxes touching a border {int(df.touches_border.sum())} "
              f"({df.touches_border.mean():.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="SAM3 mask videos -> body bounding boxes "
                    "(min_area filter + border-band area intersection)."
    )
    p.add_argument("--mask_dir", required=True, help="Directory of mask videos named <pid>.mp4")
    p.add_argument("--output", required=True, help="Output CSV.")

    p.add_argument("--mask_threshold", type=int, default=127,
                   help="Grey level above which a pixel counts as mask (default: 127). "
                        "mp4 compression leaves grey 1-11 around every edge; >0 accepts it.")
    p.add_argument("--close_px", type=int, default=9,
                   help="Morphological closing kernel in pixels (default: 9, 0 disables). "
                        "Reconnects mask fragments so min_area does not delete a limb.")
    p.add_argument("--min_area", type=int, default=1000,
                   help="Minimum component area in pixels (default: 1000).")
    p.add_argument("--band_px", type=int, default=40,
                   help="Width of the border band intersected with each component "
                        "(default: 40). The band is never erased from the mask.")
    p.add_argument("--min_band_ratio", type=float, default=0.9,
                   help="A component with at least this fraction of its pixels inside the "
                        "band is a strip (default: 0.9). Measured strips score 1.00, people "
                        "0.00, so the verdict does not flip between adjacent frames.")
    p.add_argument("--max_image_ratio", type=float, default=0.02,
                   help="A strip smaller than this fraction of the image is rejected "
                        "(default: 0.02).")

    p.add_argument("--min_x_overlap", type=float, default=0.2,
                   help="A component is merged into the box only if it overlaps the main "
                        "component horizontally by at least this fraction of the narrower "
                        "one (default: 0.2; 0 disables). A body cut by an occluder splits "
                        "top/bottom and keeps a shared x range; two separate objects sit "
                        "side by side. An outstretched arm can be dropped by this rule.")

    p.add_argument("--save_dir", default=None, help="Directory for debug frames.")
    p.add_argument("--max_debug_frames", type=int, default=200,
                   help="Cap per PID (default: 200; 0 or less means no cap).")

    args = p.parse_args()
    if args.max_debug_frames is not None and args.max_debug_frames <= 0:
        args.max_debug_frames = None

    main(args)
