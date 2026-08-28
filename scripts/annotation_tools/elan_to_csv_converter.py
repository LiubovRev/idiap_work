# coding=utf-8
"""
Convert ELAN .txt annotations to frame-level CSV for pipeline processing.

ELAN format (tab-separated):
    tier_id  [tab] [empty] [tab] begin_time_hms [tab] begin_time_sec [tab] end_time_hms [tab] end_time_sec [tab] duration_hms [tab] duration_sec [tab] value
    
Example:
    c14_CP		00:02:47.571	167.571	00:18:38.787	1118.787	00:15:51.216	951.216	CST

Output CSV: one row per annotation with frame indices computed from timestamps.
"""

import argparse
import csv
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
            parts = [p for p in parts if p]  # Remove empty strings
            
            if len(parts) < 8:
                print(f"Warning: skipping line with {len(parts)} fields: {line[:80]}")
                continue
            
            try:
                tier_id = parts[0]
                begin_time_sec = float(parts[2])  # numeric seconds
                end_time_sec = float(parts[4])    # numeric seconds
                duration_sec = float(parts[6])    # numeric seconds
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


def extract_individual_and_session_from_path(txt_file):
    """
    Extract individual ID and session ID from filename.
    
    Assumes filename format: something_individual_X_session_Y.txt
    or just use parent directory structure if available.
    
    Returns (individual_id, session_id) or (None, None) if not found.
    """
    basename = os.path.basename(txt_file)
    
    # Try pattern: Individual_X_session_Y or similar
    match = re.search(r'[Ii]ndividual.?(\d+).*[Ss]ession.?(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try pattern: just numbers
    match = re.search(r'(\d+)_(\d+)', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


def extract_individual_from_tier(tier_id):
    """
    Extract participant indicator from tier ID.
    
    Examples:
        c14_CP -> participant 14 (child)
        t1_TP -> participant 1 (therapist)
        cX_* -> child X
        tX_* -> therapist X
    
    Returns participant index (int) or None
    """
    match = re.match(r'[ct](\d+)', tier_id)
    if match:
        return int(match.group(1))
    return None


def categorize_behavior(tier_id, value):
    """
    Map ELAN tier and value to behavior category.
    
    Tiers (Type 1 - individual sessions):
        cX_CP, tX_TP -> position
        cX_CA, cX_CAO, cX_CAT -> attention
        cX_CG -> gaze
        cX_CI, tX_TI -> interaction
        cX_CV, tX_TV -> voice
        cX_CSP, tX_TSP -> session_pattern
        cX_CTCA -> common_action
        cX_JA -> joint_eye_contact
    """
    tier_parts = tier_id.lower().split('_')
    
    if len(tier_parts) < 2:
        return 'other'
    
    tier_type = tier_parts[1]
    
    if tier_type in ('cp', 'tp'):
        return 'position'
    elif tier_type in ('ca', 'cao', 'cat'):
        return 'attention'
    elif tier_type == 'cg':
        return 'gaze'
    elif tier_type in ('ci', 'ti'):
        return 'interaction'
    elif tier_type in ('cv', 'tv'):
        return 'voice'
    elif tier_type in ('csp', 'tsp'):
        return 'session_pattern'
    elif tier_type == 'ctca':
        return 'common_action'
    elif tier_type == 'ja':
        return 'joint_eye_contact'
    else:
        return 'other'


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

    # Get video FPS
    fps = get_video_fps(args.video)
    print(f"Video FPS: {fps}")
    print()

    # Convert to CSV rows
    csv_rows = []
    for ann in annotations:
        begin_frame = time_to_frame(ann['begin_time_sec'], fps)
        end_frame = time_to_frame(ann['end_time_sec'], fps)
        
        category = categorize_behavior(ann['tier_id'], ann['value'])
        
        # Determine participant info
        participant_id = extract_individual_from_tier(ann['tier_id'])
        if participant_id is None:
            participant_id = args.individual_id
        
        csv_rows.append({
            'frame_index_start': begin_frame,
            'frame_index_end': end_frame,
            'timestamp_start_sec': ann['begin_time_sec'],
            'timestamp_end_sec': ann['end_time_sec'],
            'tier_id': ann['tier_id'],
            'behavior_code': ann['value'],
            'behavior_category': category,
            'duration_sec': ann['duration_sec'],
            'individual_id': args.individual_id,
            'session_id': args.session_id,
            'participant_in_tier': participant_id,
        })

    df = pd.DataFrame(csv_rows)

    columns = [
        'frame_index_start',
        'frame_index_end',
        'timestamp_start_sec',
        'timestamp_end_sec',
        'tier_id',
        'behavior_code',
        'behavior_category',
        'duration_sec',
        'individual_id',
        'session_id',
        'participant_in_tier',
    ]
    df = df[columns]

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Frame range: {df['frame_index_start'].min()} — {df['frame_index_end'].max()}")
    print(f"Behavior categories: {sorted(df['behavior_category'].unique())}")
    print(f"Tiers: {sorted(df['tier_id'].unique())}")
    print()
    print("✓ Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ELAN .txt annotations to frame-level CSV."
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
