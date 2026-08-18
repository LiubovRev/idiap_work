# coding=utf-8

import argparse

import cv2
import pandas as pd


COLORS = [
    (0, 255, 0),      # PID 0
    (255, 0, 0),      # PID 1
    (0, 0, 255),      # PID 2
    (255, 255, 0),    # PID 3
    (255, 0, 255),    # PID 4
    (0, 255, 255),    # PID 5
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

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer: {args.output}"
        )

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        detections = frames.get(frame_index)

        if detections is not None:

            for _, row in detections.iterrows():

                pid = int(row["pid"])
                color = COLORS[pid % len(COLORS)]

                # -------------------------------------------------
                # Body box
                # -------------------------------------------------

                bx1 = int(row["body_bbox_xmin"])
                by1 = int(row["body_bbox_ymin"])
                bx2 = int(row["body_bbox_xmax"])
                by2 = int(row["body_bbox_ymax"])

                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    color,
                    3,
                )

                # -------------------------------------------------
                # Head box
                # -------------------------------------------------

                if pd.notna(row["head_bbox_xmin"]):

                    hx1 = int(row["head_bbox_xmin"])
                    hy1 = int(row["head_bbox_ymin"])
                    hx2 = int(row["head_bbox_xmax"])
                    hy2 = int(row["head_bbox_ymax"])

                    cv2.rectangle(
                        frame,
                        (hx1, hy1),
                        (hx2, hy2),
                        color,
                        2,
                    )

                    confidence = row["head_confidence"]

                    label = (
                        f"PID {pid} | "
                        f"head {confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (hx1, max(25, hy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                else:

                    label = f"PID {pid} | NO HEAD"

                    cv2.putText(
                        frame,
                        label,
                        (bx1, max(25, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                # PID label on body
                cv2.putText(
                    frame,
                    f"PID {pid}",
                    (bx1, min(height - 10, by1 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Frame number
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

        if frame_index % 100 == 0:
            print(
                f"Processed {frame_index}/{num_frames}"
            )

    cap.release()
    writer.release()

    print()
    print(f"Done: {args.output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Visualize SAM3 PID, body boxes and "
            "matched head boxes."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Original video",
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Combined SAM3 head/body CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output visualization video",
    )

    args = parser.parse_args()

    main(args)
