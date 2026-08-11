#!/usr/bin/env python3
"""
merge_therapist_masks.py

Merges two fragmented therapist mask videos (e.g., 1.mp4 and 2.mp4) into a single
unified therapist mask. Uses frames from track 1 where available, fills gaps from track 2.

Usage:
    python merge_therapist_masks.py --mask1 1.mp4 --mask2 2.mp4 --output merged_therapist.mp4
"""

import cv2
import argparse
import numpy as np
from pathlib import Path


def extract_frames_as_binary(video_path, target_frames=None):
    """Extract all frames from video and convert to binary mask (0=bg, 255=fg)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for binary mask detection
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Threshold: assume non-zero pixels are foreground
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        frames.append(binary)
        frame_count += 1
    
    cap.release()
    
    if target_frames and frame_count < target_frames:
        # Pad with empty frames if this track is shorter
        padding = np.zeros((target_frames - frame_count, height, width), dtype=np.uint8)
        frames = np.array(frames)
        frames = np.vstack([frames[:, np.newaxis, :, :], padding[:, np.newaxis, :, :]])
        frames = frames.reshape(-1, height, width)
    else:
        frames = np.array(frames)
    
    return frames, fps, width, height


def merge_masks(mask1_frames, mask2_frames):
    """
    Merge two mask sequences. Prioritize mask1, fill gaps with mask2.
    Returns merged sequence, padded to the length of whichever is longer.
    """
    n1, h1, w1 = mask1_frames.shape
    n2, h2, w2 = mask2_frames.shape
    
    # Ensure same spatial dimensions
    assert h1 == h2 and w1 == w2, "Mask dimensions must match"
    
    max_frames = max(n1, n2)
    merged = np.zeros((max_frames, h1, w1), dtype=np.uint8)
    
    for i in range(max_frames):
        # Use mask1 if available and non-empty
        if i < n1 and mask1_frames[i].sum() > 0:
            merged[i] = mask1_frames[i]
        # Otherwise use mask2 if available and non-empty
        elif i < n2 and mask2_frames[i].sum() > 0:
            merged[i] = mask2_frames[i]
        # Otherwise leave as zero (empty frame)
    
    return merged


def save_merged_video(merged_frames, output_path, fps, width, height):
    """Write merged mask frames back to video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for i, frame in enumerate(merged_frames):
        # Convert binary mask back to BGR for video output
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(bgr_frame)
        if (i + 1) % 500 == 0:
            print(f"  wrote {i + 1}/{len(merged_frames)} frames")
    
    writer.release()
    print(f"[ok] Merged video written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge two fragmented therapist mask tracks")
    parser.add_argument("--mask1", required=True, help="Path to first therapist mask video (1.mp4)")
    parser.add_argument("--mask2", required=True, help="Path to second therapist mask video (2.mp4)")
    parser.add_argument("--output", default="merged_therapist.mp4", help="Output merged mask video")
    args = parser.parse_args()
    
    print(f"[info] Loading mask 1: {args.mask1}")
    mask1, fps1, w1, h1 = extract_frames_as_binary(args.mask1)
    print(f"       -> {mask1.shape[0]} frames, {w1}x{h1}, {fps1:.1f} fps")
    
    print(f"[info] Loading mask 2: {args.mask2}")
    mask2, fps2, w2, h2 = extract_frames_as_binary(args.mask2)
    print(f"       -> {mask2.shape[0]} frames, {w2}x{h2}, {fps2:.1f} fps")
    
    # Use fps and dimensions from mask1
    fps = fps1 if fps1 > 0 else fps2
    
    print(f"[info] Merging masks (prioritizing track 1, filling gaps from track 2)...")
    merged = merge_masks(mask1, mask2)
    print(f"       -> {merged.shape[0]} frames in merged output")
    
    print(f"[info] Writing merged video to: {args.output}")
    save_merged_video(merged, args.output, fps, w1, h1)


if __name__ == "__main__":
    main()
