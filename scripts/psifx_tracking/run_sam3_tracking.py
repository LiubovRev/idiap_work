#!/usr/bin/env python3

import subprocess
from pathlib import Path
from datetime import datetime
import json
import os

# -------------------------
# Configuration
# -------------------------
base_dir = Path("7_INDIVIDUAL_14")
input_video = base_dir / "camera_a.mkv"
processed_video = base_dir / "camera_a_trimmed.mkv"
mask_dir = base_dir / "MaskDir2""

device = "cuda"
model_path = "facebook/sam3"
text_prompt = "people"
confidence_threshold = 0.4  # Store locally, will be passed to tool directly

manual_time = True
manual_start = 167
manual_end = 500

# -------------------------
# Utilities
# -------------------------
def get_video_info(path):
    """Get video resolution, fps, duration."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            stream = json.loads(result.stdout)["streams"][0]
            fps_parts = stream.get("r_frame_rate", "30/1").split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1])
            duration = float(stream.get("duration", 0))
            w, h = stream.get("width"), stream.get("height")
            return {
                "resolution": f"{w}x{h}",
                "fps": round(fps, 2),
                "duration_s": round(duration, 2),
                "frames": int(duration * fps),
            }
    except: 
        pass
    return {}

def get_file_size_mb(path):
    """Get file size in MB."""
    if path.exists():
        return round(path.stat().st_size / (1024**2), 2)
    return None


# -------------------------
# Command runner
# -------------------------
def run(cmd, step):
    print(f"\n {step}...")
    t0 = datetime.now()
    try:
        subprocess.run(cmd, check=True)
        elapsed = (datetime.now() - t0).total_seconds()
        
        print(f" {step} completed ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {step} failed")
        raise

# -------------------------
# 1. Video processing
# -------------------------
cmd = [
    "ffmpeg",
    "-y",
    "-ss", str(manual_start),
    "-to", str(manual_end),
    "-i", input_video,
    "-map", "0:0",
    "-c:v", "libx264",
    processed_video,
]

subprocess.run(cmd, check=True)

# -------------------------
# 2. SAM3 inference (without --confidence_threshold if not supported)
# -------------------------
run([
    "psifx", "video", "tracking", "sam3", "inference",
    "--video", str(processed_video),
    "--mask_dir", str(mask_dir),
    "--text_prompt", text_prompt,
    "--chunk_size", "300",
    "--iou_threshold", "0.3",
    "--device", device,
    "--model_path", model_path,
], "SAM3 inference")

# -------------------------
# Final report
# -------------------------


masks = list(mask_dir.glob("*.mp4")) if mask_dir.exists() else []
mask_size = sum(f.stat().st_size for f in masks) / (1024**2) if masks else 0


