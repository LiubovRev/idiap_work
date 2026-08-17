import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def mask_video_to_rows(mask_file: Path, pid: int):
    cap = cv2.VideoCapture(str(mask_file))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open mask video: {mask_file}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    rows = []

    for frame_index in tqdm(
        range(num_frames),
        desc=f"PID {pid}",
    ):
        ok, frame = cap.read()

        if not ok:
            print(
                f"Warning: could not read frame "
                f"{frame_index} from {mask_file}"
            )
            continue

        # Binary black/white mask.
        # Convert to grayscale in case the mp4 has 3 channels.
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Any non-black pixel belongs to the mask.
        ys, xs = np.where(gray > 0)

        # Empty mask for this frame.
        if len(xs) == 0:
            continue

        rows.append(
            {
                "frame_index": frame_index,
                "pid": pid,
                "body_bbox_xmin": float(xs.min()),
                "body_bbox_ymin": float(ys.min()),
                "body_bbox_xmax": float(xs.max()),
                "body_bbox_ymax": float(ys.max()),
                "video_file": str(mask_file),
                "video_name": mask_file.name,
                "num_frames": num_frames,
                "frame_height": height,
                "frame_width": width,
                "fps": fps,
            }
        )

    cap.release()

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM3 binary mask videos to body bounding boxes."
    )

    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Directory containing SAM3 mask videos, e.g. MaskDir/",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file.",
    )

    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    output_file = Path(args.output)

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # SAM3 mask files are named with their PID:
    # 0.mp4, 1.mp4, 2.mp4, ...
    mask_files = sorted(
        [
            f
            for f in mask_dir.glob("*.mp4")
            if f.stem.isdigit()
        ],
        key=lambda f: int(f.stem),
    )

    if not mask_files:
        raise RuntimeError(
            f"No PID mask videos found in {mask_dir}"
        )

    print(f"Mask directory: {mask_dir}")
    print(f"Found {len(mask_files)} SAM3 mask videos:")
    for f in mask_files:
        print(f"  PID {f.stem}: {f}")

    all_rows = []

    for mask_file in mask_files:
        pid = int(mask_file.stem)

        rows = mask_video_to_rows(
            mask_file=mask_file,
            pid=pid,
        )

        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No non-empty masks found.")

    df = pd.DataFrame(all_rows)

    # Sort by frame and PID.
    df = df.sort_values(
        ["frame_index", "pid"]
    ).reset_index(drop=True)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_file,
        index=False,
        float_format="%.2f",
    )

    print()
    print("Done.")
    print(f"Output: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Frames: {df['frame_index'].nunique()}")
    print(f"PIDs: {df['pid'].nunique()}")
    print()
    print("Rows per PID:")
    print(df["pid"].value_counts().sort_index())
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print(df.head(10))


if __name__ == "__main__":
    main()
