#!/usr/bin/env python3
"""
apply_mask_decisions.py

Reads mask_decisions.json (from inspect_masks.py) and applies the decisions:
  - merge: combine multiple tracks into one
  - keep: rename/tag for use
  - delete: remove the track file

Usage:
    # After manually editing mask_decisions.json:
    python apply_mask_decisions.py --session_dir /path/to/7_INDIVIDUAL_14 --decisions mask_decisions.json
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil


def merge_mask_videos(video_paths, output_path, fps=None):
    """Merge multiple mask videos into one."""
    print(f"  Merging {len(video_paths)} videos...")
    
    frames_list = []
    fps_ref = None
    height, width = None, None
    max_frames = 0
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"    [warn] Could not open {video_path}")
            continue
        
        if fps_ref is None:
            fps_ref = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale for merging
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frames.append(gray)
        
        cap.release()
        frames_list.append(np.array(frames))
        max_frames = max(max_frames, len(frames))
    
    # Pad all to same length and merge
    merged = np.zeros((max_frames, height, width), dtype=np.uint8)
    for frames in frames_list:
        for i, f in enumerate(frames):
            # OR operation: combine masks
            merged[i] = cv2.bitwise_or(merged[i], f)
    
    # Write output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_ref or fps or 30, (width, height))
    
    for i, frame in enumerate(merged):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"    [ok] Merged video: {output_path} ({merged.shape[0]} frames)")


def apply_decisions(session_dir, decisions_file):
    """Apply mask decisions: merge, delete, or keep."""
    session_dir = Path(session_dir)
    mask_dir = session_dir / "MaskDir"
    decisions_path = Path(decisions_file) if Path(decisions_file).is_absolute() else session_dir / decisions_file
    
    if not decisions_path.exists():
        print(f"[error] Decisions file not found: {decisions_path}")
        return
    
    with open(decisions_path, 'r') as f:
        decisions = json.load(f)
    
    print(f"\n[info] Applying decisions from: {decisions_path}\n")
    
    # Group by action
    to_merge = {}
    to_delete = []
    to_keep = {}
    
    for track_id, decision in decisions.items():
        action = decision.get("action")
        
        if action == "merge":
            merge_target = decision.get("merge_with")
            if merge_target:
                if merge_target not in to_merge:
                    to_merge[merge_target] = []
                to_merge[merge_target].append(track_id)
                print(f"[merge] {track_id} -> {merge_target}")
        
        elif action == "delete":
            to_delete.append(track_id)
            print(f"[delete] {track_id}")
        
        elif action == "keep":
            role = decision.get("role", "unknown")
            to_keep[track_id] = role
            print(f"[keep] {track_id} (role: {role})")
    
    print(f"\n{'='*70}\n")
    
    # Apply merges
    for merge_target, source_ids in to_merge.items():
        print(f"[action] Merging into '{merge_target}':")
        video_paths = [mask_dir / f"{src}.mp4" for src in source_ids]
        video_paths = [p for p in video_paths if p.exists()]
        
        if video_paths:
            output_path = mask_dir / f"{merge_target}_merged.mp4"
            merge_mask_videos(video_paths, output_path)
    
    # Apply deletes
    for track_id in to_delete:
        video_path = mask_dir / f"{track_id}.mp4"
        if video_path.exists():
            print(f"[action] Deleting {video_path.name}...")
            backup_path = mask_dir / f"{track_id}_DELETED.mp4"
            shutil.move(str(video_path), str(backup_path))
            print(f"         (backed up to {backup_path.name})")
    
    # Summary of kept tracks
    print(f"\n[summary] Final tracks to use:")
    for track_id, role in to_keep.items():
        video_path = mask_dir / f"{track_id}.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"  {track_id}.mp4 -> {role} ({frames} frames)")
    
    # Check for merged outputs
    for merged_file in mask_dir.glob("*_merged.mp4"):
        cap = cv2.VideoCapture(str(merged_file))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {merged_file.name} ({frames} frames)")
    
    print(f"\n[ok] Decisions applied. Update your annotation script with the final track filenames.")


def main():
    parser = argparse.ArgumentParser(description="Apply mask inspection decisions")
    parser.add_argument("--session_dir", required=True, help="Session directory")
    parser.add_argument("--decisions", default="mask_decisions.json", help="Decisions JSON file")
    args = parser.parse_args()
    
    apply_decisions(args.session_dir, args.decisions)


if __name__ == "__main__":
    main()
