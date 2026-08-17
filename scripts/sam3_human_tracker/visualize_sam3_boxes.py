import cv2
import pandas as pd
import os

VIDEO = "video_processed_a.mp4"
CSV = "sam3_body_detections.csv"
OUTPUT = "sam3_boxes_visualization.mp4"

# Read detections
df = pd.read_csv(CSV)

# Group detections by frame for fast lookup
frames = {
    int(frame_idx): group
    for frame_idx, group in df.groupby("frame_index")
}

# Open video
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {VIDEO}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Frames: {num_frames}")

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT,
    fourcc,
    fps,
    (width, height),
)

# Different colors for different PIDs
colors = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
]

frame_idx = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    # Get detections for this frame
    detections = frames.get(frame_idx)

    if detections is not None:
        for _, row in detections.iterrows():

            pid = int(row["pid"])

            x1 = int(row["body_bbox_xmin"])
            y1 = int(row["body_bbox_ymin"])
            x2 = int(row["body_bbox_xmax"])
            y2 = int(row["body_bbox_ymax"])

            color = colors[pid % len(colors)]

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            # Label
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

    # Frame number
    cv2.putText(
        frame,
        f"Frame: {frame_idx}/{num_frames - 1}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)

    frame_idx += 1

    if frame_idx % 500 == 0:
        print(f"Processed {frame_idx}/{num_frames} frames")

cap.release()
writer.release()

print()
print(f"Done: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f} MB")
