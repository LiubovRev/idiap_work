#!/usr/bin/env python3
# coding=utf-8
# SPDX-FileCopyrightText: Copyright 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-License-Identifier: LicenseRef-HumanTracker

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from humantracker.body_tracking import BodyTrack
from humantracker.utils import (
    create_video_writer,
    video_capture,
    check_file,
)

# Participant colors
THERAPIST_COLOR = (255, 170, 64)   # Blue (BGR)
CHILD_COLOR = (102, 153, 255)      # Orange (BGR)
HEAD_COLOR_THERAPIST = (255, 213, 140)
HEAD_COLOR_CHILD = (204, 204, 255)


def load_metadata(path):
    """Load metadata from processing_log.json"""
    with open(path, "r") as f:
        metadata = json.load(f)

    if "identity_mapping" in metadata:
        identity = metadata["identity_mapping"]
        return {
            "child_annotation_pid": identity["child"].get("annotation_pid", 14),
            "child_tracking_pid": int(identity["child"]["tracking_pid"]),
            "therapist_annotation_pid": identity["therapist"].get("annotation_pid", 0),
            "therapist_tracking_pid": int(identity["therapist"]["tracking_pid"]),
        }
    else:
        participants = metadata["participants"]
        return {
            "child_annotation_pid": int(participants["child"]["annotation_pid"]),
            "child_tracking_pid": int(participants["child"]["tracking_pid"]),
            "therapist_annotation_pid": int(participants["therapist"]["annotation_pid"]),
            "therapist_tracking_pid": int(participants["therapist"]["tracking_pid"]),
        }


def load_annotations(path):
    """Load behavioral annotations from CSV, extract codes only"""
    df = pd.read_csv(path)
    annotations = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        frame = int(row["frame_index"])
        pid = int(row["pid"])
        code = str(row["behavior_code"])
        
        annotations[frame][pid].append(code)

    return annotations


def draw_legend(canvas, active_annotations, video_height, video_width):
    """Draw 2-column legend: CHILD (left) | THERAPIST (right). Show code + definition."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    margin = 15
    
    # Code definitions (from elan_to_csv_converter)
    CODE_DEFINITION = {
        # Child codes
        "CST": "Standing",
        "CHO": "Hovering/leaning",
        "CSI": "Sitting",
        "AO": "Attending objects",
        "ANO": "Attending non-session objects",
        "AT": "Attending therapist",
        "GO": "Gaze at objects",
        "GT": "Gaze at therapist",
        "GNO": "Gaze elsewhere",
        "CS": "Child speaking",
        "CNS": "Child non-speech sounds",
        "C": "Child interacting",
        "T_V": "Child verbal (no contact)",
        "T_P": "Child physical (no speech)",
        # Therapist codes
        "TST": "Standing",
        "TSI": "Sitting",
        "T": "Therapist interacting",
        "TS": "Therapist speaking",
        "TNS": "Therapist non-speech sounds",
        "TC": "Child-therapist together",
        "TCR": "Crouch",
        "PRC": "Preparing/cleaning",
    }
    
    # Split screen
    left_x = margin
    right_x = video_width // 2 + margin
    y_start = video_height + 8
    
    # CHILD (left)
    cv2.putText(canvas, "CHILD", (left_x, y_start), font, font_scale + 0.1,
                (255, 255, 255), thickness + 1, cv2.LINE_AA)
    y = y_start + line_height
    
    for code in sorted(active_annotations.get("CHILD", [])):
        definition = CODE_DEFINITION.get(code, "")
        text = f"{code}: {definition}" if definition else code
        cv2.putText(canvas, text, (left_x, y), font, font_scale, 
                   (200, 255, 200), thickness, cv2.LINE_AA)
        y += line_height
    
    # THERAPIST (right)
    cv2.putText(canvas, "THERAPIST", (right_x, y_start), font, font_scale + 0.1,
                (255, 255, 255), thickness + 1, cv2.LINE_AA)
    y = y_start + line_height
    
    for code in sorted(active_annotations.get("THERAPIST", [])):
        definition = CODE_DEFINITION.get(code, "")
        text = f"{code}: {definition}" if definition else code
        cv2.putText(canvas, text, (right_x, y), font, font_scale, 
                   (200, 255, 200), thickness, cv2.LINE_AA)
        y += line_height


def process_video(
    video_file: str,
    annotations_file: str,
    tracking_data_file: str,
    metadata_file: str,
    output_file: str,
) -> None:
    """Process video with annotation overlay and compact legend"""

    # Load metadata
    metadata = load_metadata(metadata_file)
    child_annotation_pid = metadata["child_annotation_pid"]
    child_tracking_pid = metadata["child_tracking_pid"]
    therapist_annotation_pid = metadata["therapist_annotation_pid"]
    therapist_tracking_pid = metadata["therapist_tracking_pid"]

    print("[info] Participant mapping:")
    print(f"  child: annotation PID {child_annotation_pid} → tracking PID {child_tracking_pid}")
    print(f"  therapist: annotation PID {therapist_annotation_pid} → tracking PID {therapist_tracking_pid}")

    # Load annotations and tracking
    annotations = load_annotations(annotations_file)
    df_tracking = pd.read_csv(tracking_data_file)

    # Annotation tiers
    THERAPIST_TIERS = {"TP", "TSP", "TI", "TV"}
    CHILD_TIERS = {"CP", "CAO", "CAT", "CG", "CSP", "CV", "CI"}

    # Video setup
    cap, fps, width, height, num_frames = video_capture(video_file)
    fps = 15.0 if fps <= 0 else fps

    # Output with legend space
    legend_height = 180
    output_height = height + legend_height
    writer = create_video_writer(output_file, fps, width, output_height)

    # Process frames
    for frame_index in tqdm(range(num_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_annotations = annotations.get(frame_index, {})
        df_frame = df_tracking[df_tracking["frame_index"] == frame_index]

        # Draw bounding boxes (no text labels on faces)
        for _, row in df_frame.iterrows():
            pid = int(row["pid"])
            
            body_xmin = float(row.get("body_bbox_xmin", 0))
            body_ymin = float(row.get("body_bbox_ymin", 0))
            body_xmax = float(row.get("body_bbox_xmax", 0))
            body_ymax = float(row.get("body_bbox_ymax", 0))
            
            if pd.isna(body_xmin) or body_xmin == 0:
                continue
            
            if pid == therapist_tracking_pid:
                body_color = THERAPIST_COLOR
            elif pid == child_tracking_pid:
                body_color = CHILD_COLOR
            else:
                continue
            
            cv2.rectangle(frame, (int(body_xmin), int(body_ymin)), 
                         (int(body_xmax), int(body_ymax)), body_color, 2)
            
            # Head box
            if not pd.isna(row.get("head_bbox_xmin")) and row.get("head_bbox_xmin") != "":
                try:
                    head_xmin = int(row["head_bbox_xmin"])
                    head_ymin = int(row["head_bbox_ymin"])
                    head_xmax = int(row["head_bbox_xmax"])
                    head_ymax = int(row["head_bbox_ymax"])
                    head_confidence = float(row.get("head_confidence", 1.0))
                    
                    if head_confidence > 0.3:
                        head_color = HEAD_COLOR_THERAPIST if pid == therapist_tracking_pid else HEAD_COLOR_CHILD
                        cv2.rectangle(frame, (head_xmin, head_ymin), (head_xmax, head_ymax), 
                                     head_color, 2)
                except (ValueError, TypeError):
                    pass

        # Collect active codes for this frame
        active_annotations = {}
        
        child_codes = set()
        if child_annotation_pid in frame_annotations:
            child_codes = set(frame_annotations[child_annotation_pid])
        
        therapist_codes = set()
        if therapist_annotation_pid in frame_annotations:
            therapist_codes = set(frame_annotations[therapist_annotation_pid])
        
        if child_codes:
            active_annotations["CHILD"] = child_codes
        if therapist_codes:
            active_annotations["THERAPIST"] = therapist_codes

        # Create output canvas
        canvas = np.zeros((output_height, width, 3), dtype=np.uint8)
        canvas[:height, :] = frame
        
        # Draw compact legend
        draw_legend(canvas, active_annotations, height, width)

        writer.write(canvas)

    cap.release()
    writer.release()

    print(f"[ok] Annotated video: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay annotations with compact 2-column legend"
    )
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument("--annotations", required=True, help="Annotations CSV")
    parser.add_argument("--body-detections", required=True, help="Tracking CSV")
    parser.add_argument("--metadata", required=True, help="Metadata JSON")
    parser.add_argument("--output", required=True, help="Output video")

    args = parser.parse_args()

    check_file(args.video)
    check_file(args.annotations)
    check_file(args.body_detections)
    check_file(args.metadata)

    process_video(
        args.video,
        args.annotations,
        args.body_detections,
        args.metadata,
        args.output,
    )


if __name__ == "__main__":
    main()
