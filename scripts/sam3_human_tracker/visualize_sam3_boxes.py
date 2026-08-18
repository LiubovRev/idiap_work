# coding=utf-8

import argparse

import cv2
import pandas as pd


COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def main(args):
    df = pd.read_csv(args.csv)

    frames = {
        int(frame_index): group
        for frame_index, group in df.groupby("frame_index")
    }

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.video}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Frames: {num_frames}")
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0

    if not writer.isOpened():
        raise RuntimeError(
        f"Could not create output video: {args.output}"
    )    
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        detections = frames.get(frame_index)

        if detections is not None:
            for _, row in detections.iterrows():
                pid = int(row["pid"])

                x1 = int(row["body_bbox_xmin"])
                y1 = int(row["body_bbox_ymin"])
                x2 = int(row["body_bbox_xmax"])
                y2 = int(row["body_bbox_ymax"])

                color = COLORS[pid % len(COLORS)]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3,
                )

                label = f"SAM3 PID {pid}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"Frame: {frame_index}/{num_frames - 1}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_index += 1

        if frame_index % 500 == 0:
            print(f"Processed {frame_index}/{num_frames}")

    cap.release()
    writer.release()

    print()
    print(f"Done: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize SAM3 body bounding boxes on a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Original video",
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="SAM3 body detection CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output visualization video",
    )

    args = parser.parse_args()

    main(args)
