# coding=utf-8
"""
Convert ELAN .txt annotations to frame-level CSV for pipeline processing.

ELAN format (tab-separated):
    tier_id  [tab] [empty] [tab] begin_time_hms [tab] begin_time_sec [tab] end_time_hms [tab] end_time_sec [tab] duration_hms [tab] duration_sec [tab] value
    
Example:
    c14_CP		00:02:47.571	167.571	00:18:38.787	1118.787	00:15:51.216	951.216	CST

Output CSV: one row per frame within annotation interval.
Accounts for video trimming using SAM3 metadata.
"""

import argparse
import json
import os
import re

import cv2
import pandas as pd


def parse_elan_txt(txt_file):
    """
    Parse ELAN .txt output (tab-separated).
    
    Returns list of dicts with keys:
        tier_id, begin_time_sec, end_time_sec, duration_sec, value
    """
    rows = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab and filter empty fields
            parts = [p.strip() for p in line.split('\t')]
            parts = [p for p in parts if p]  
            
            if len(parts) < 8:
                print(f"Warning: skipping line with {len(parts)} fields: {line[:80]}")
                continue
            
            try:
                tier_id = parts[0]
                begin_time_sec = float(parts[2])  
                end_time_sec = float(parts[4])    
                duration_sec = float(parts[6])    
                value = parts[7]
                
                rows.append({
                    'tier_id': tier_id,
                    'begin_time_sec': begin_time_sec,
                    'end_time_sec': end_time_sec,
                    'duration_sec': duration_sec,
                    'value': value,
                })
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line[:80]} — {e}")
                continue
    
    return rows


def extract_individual_from_tier(tier_id):
    """
    Extract participant ID from tier ID.
    
    Examples:
        c14_CP -> pid 14 (child)
        t1_TP -> pid 1 (therapist) 
        cX_* -> child, use pid X
        tX_* -> therapist, use pid 0
    
    Returns participant ID (int) or None
    """
    match = re.match(r'([ct])(\d+)', tier_id)
    if match:
        role = match.group(1)
        num = int(match.group(2))
        
        # c = child, t = therapist (use 0 for therapist)
        if role == 'c':
            return num 
        elif role == 't':
            return 0    
    return None


def load_sam3_metadata(metadata_json):
    """
    Load SAM3 metadata to get video trimming info.
    
    Returns (fps, trim_start_sec) where trim_start_sec is the offset in the original video.
    """
    if not metadata_json or not os.path.exists(metadata_json):
        return None, 0.0
    
    try:
        with open(metadata_json, 'r') as f:
            meta = json.load(f)
        
        fps = meta.get('input_video', {}).get('fps')
        trim_start_s = meta.get('time_window', {}).get('start_s', 0.0)
        
        print(f"Loaded SAM3 metadata:")
        print(f"  FPS: {fps}")
        print(f"  Trim start: {trim_start_s}s")
        
        return fps, trim_start_s
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load SAM3 metadata: {e}")
        return None, 0.0


def get_video_fps(video_path):
    """Extract FPS from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS {fps} from video: {video_path}")
    return fps


def time_to_frame(time_sec, fps):
    """Convert time (seconds) to frame index."""
    return int(time_sec * fps)


def main(args):
    print(f"ELAN annotation file: {args.elan_txt}")
    print(f"Video file: {args.video}")
    if args.sam3_metadata:
        print(f"SAM3 metadata: {args.sam3_metadata}")
    print(f"Individual ID: {args.individual_id}")
    print(f"Session ID: {args.session_id}")
    print(f"Output CSV: {args.output}")
    print()

    # Parse ELAN
    annotations = parse_elan_txt(args.elan_txt)
    print(f"Parsed {len(annotations)} annotations from ELAN")

    if not annotations:
        print("ERROR: No annotations parsed. Check file format.")
        return

    # Get video FPS (metadata if available)
    fps_meta, trim_start_s = load_sam3_metadata(args.sam3_metadata)
    if fps_meta:
        fps = fps_meta
        print(f"Video FPS from SAM3 metadata: {fps}")
    else:
        fps = get_video_fps(args.video)
        print(f"Video FPS from video file: {fps}")
        trim_start_s = 0.0
    
    # Override with explicit trim_start if provided
    if args.trim_start_seconds is not None:
        trim_start_s = args.trim_start_seconds
        print(f"Using explicit trim start: {trim_start_s}s")
    
    print()

    # Compute trim start frame
    trim_start_frame = int(trim_start_s * fps)
    print(f"Trim start: {trim_start_s}s = frame {trim_start_frame}")
    print()

    # Convert to CSV rows (one row per frame)
    csv_rows = []
    for ann in annotations:
        begin_frame = time_to_frame(ann['begin_time_sec'], fps)
        end_frame = time_to_frame(ann['end_time_sec'], fps)
        
        # Skip annotations outside trimming window
        if end_frame < trim_start_frame:
            continue  # Annotation ends before trim starts
        if begin_frame >= trim_start_frame + int((300 * fps)):  # Rough check
            continue  # Annotation starts after trim ends (adjust as needed)
        
        # Adjust frame indices to trimmed video (subtract trim start)
        begin_frame_trimmed = max(0, begin_frame - trim_start_frame)
        end_frame_trimmed = max(0, end_frame - trim_start_frame)
        
        # Determine participant ID from behavior code prefix
        behavior_code = ann['value']
        if behavior_code.startswith('T'):
            participant_id = 0  # Therapist
        elif behavior_code.startswith('C'):
            participant_id = args.individual_id  # Child
        else:
            participant_id = args.individual_id  # Default to child ID
        
        # One row per frame in the annotation interval
        for frame_idx in range(begin_frame_trimmed, end_frame_trimmed + 1):
            # Original frame index and time in full video
            frame_original = frame_idx + trim_start_frame
            time_original = frame_original / fps
            
            # Trimmed time
            time_trimmed = frame_idx / fps
            
            csv_rows.append({
                'frame_index': frame_idx,
                'frame_index_original': frame_original,
                'time_sec': time_trimmed,
                'time_sec_original': time_original,
                'pid': participant_id,
                'behavior_code': ann['value'],
            })

    df = pd.DataFrame(csv_rows)

    columns = [
        'frame_index',
        'frame_index_original',
        'time_sec',
        'time_sec_original',
        'pid',
        'behavior_code',
    ]
    df = df[columns]

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Frame range: {df['frame_index'].min()} — {df['frame_index'].max()}")
    print(f"Unique PIDs: {sorted(df['pid'].unique())}")
    print(f"Behavior codes: {sorted(df['behavior_code'].unique())}")
    print()
    print("Conversion complete.")


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
        help="Video file (to extract FPS)",
    )
    parser.add_argument(
        "--sam3_metadata",
        default=None,
        help="SAM3 metadata JSON (to get FPS and trimming info)",
    )
    parser.add_argument(
        "--trim_start_seconds",
        type=float,
        default=None,
        help="Trim start time in seconds (overrides metadata)",
    )
    parser.add_argument(
        "--individual_id",
        type=int,
        required=True,
        help="Individual/participant ID",
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
