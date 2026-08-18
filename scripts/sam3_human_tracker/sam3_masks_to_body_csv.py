# coding=utf-8

import argparse
import os

import cv2
import pandas as pd

def mask_to_bbox(mask, min_area=500):
    """Return bounding box of the largest sufficiently large mask component."""

    binary = (mask > 0).astype("uint8")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    valid_labels = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]

    if not valid_labels:
        return None

    # Usually the person is the largest component
    label = max(
        valid_labels,
        key=lambda i: stats[i, cv2.CC_STAT_AREA]
    )

    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    w = stats[label, cv2.CC_STAT_WIDTH]
    h = stats[label, cv2.CC_STAT_HEIGHT]

    return (
        float(x),
        float(y),
        float(x + w - 1),
        float(y + h - 1),
    )

def process_mask(mask_file, pid):
    cap = cv2.VideoCapture(mask_file)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open mask video: {mask_file}")

    rows = []

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        # Convert to grayscale in case the mask video is RGB.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bbox = mask_to_bbox(gray)

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox

            rows.append(
                {
                    "frame_index": frame_index,
                    "pid": pid,
                    "body_bbox_xmin": xmin,
                    "body_bbox_ymin": ymin,
                    "body_bbox_xmax": xmax,
                    "body_bbox_ymax": ymax,
                }
            )

        frame_index += 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = frame_index

    cap.release()

    return rows, fps, width, height, num_frames


def main(args):
    mask_files = sorted(
        [
            f
            for f in os.listdir(args.mask_dir)
            if f.lower().endswith((".mp4", ".avi", ".mkv"))
        ],
        key=lambda x: int(os.path.splitext(x)[0])
        if os.path.splitext(x)[0].isdigit()
        else x,
    )

    if not mask_files:
        raise RuntimeError(f"No mask videos found in {args.mask_dir}")

    print(f"Mask directory: {args.mask_dir}")
    print(f"Found {len(mask_files)} mask files")

    all_rows = []

    video_info = None

    for mask_file in mask_files:
        path = os.path.join(args.mask_dir, mask_file)

        name = os.path.splitext(mask_file)[0]

        try:
            pid = int(name)
        except ValueError:
            print(f"Skipping non-numeric mask: {mask_file}")
            continue

        print(f"Processing PID {pid}: {mask_file}")

        rows, fps, width, height, num_frames = process_mask(path, pid)

        for row in rows:
            row["video_file"] = path
            row["video_name"] = mask_file
            row["num_frames"] = num_frames
            row["frame_height"] = height
            row["frame_width"] = width
            row["fps"] = fps

        all_rows.extend(rows)

        if video_info is None:
            video_info = (fps, width, height, num_frames)

    df = pd.DataFrame(all_rows)

    columns = [
        "frame_index",
        "pid",
        "body_bbox_xmin",
        "body_bbox_ymin",
        "body_bbox_xmax",
        "body_bbox_ymax",
        "video_file",
        "video_name",
        "num_frames",
        "frame_height",
        "frame_width",
        "fps",
    ]

    df = df[columns]

    df.to_csv(args.output, index=False, float_format="%.2f")

    print()
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"PIDs: {sorted(df.pid.unique())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAM3 binary mask videos to body bounding-box CSV."
    )

    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Directory containing SAM3 mask videos (e.g. 0.mp4, 1.mp4, ...)",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file",
    )

    args = parser.parse_args()

    main(args)
