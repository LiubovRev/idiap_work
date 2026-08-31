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
input_video = base_dir / "camera_a_full.mkv"
processed_video = base_dir / "camera_a_trimmed.mkv"
mask_dir = base_dir / "MaskDir2"
metadata_dir = base_dir / "metadata"

device = "cpu"
model_path = "facebook/sam3"
text_prompt = "people"
confidence_threshold = 0.4  # Store locally, will be passed to tool directly

manual_time = True
manual_start = 167
manual_end = 350

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
# Initialize metadata
# -------------------------
metadata_dir.mkdir(parents=True, exist_ok=True)
metadata_file = metadata_dir / "processing_log.json"

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Get input video info
input_info = get_video_info(input_video)
if input_info:
    fps = input_info.get("fps", 30)
    start_frame = int(manual_start * fps)
    end_frame = int(manual_end * fps)
else:
    start_frame = end_frame = 0

metadata = {
    "session_id": session_id,
    "timestamp": datetime.now().isoformat(),
    "input_video": {
        "path": str(input_video.absolute()),
        "size_mb": get_file_size_mb(input_video),
        **input_info,
    },
    "time_window": {
        "start_s": manual_start,
        "end_s": manual_end,
        "duration_s": manual_end - manual_start,
        "frames": end_frame - start_frame,
    },
    "sam3_config": {
        "device": device,
        "model_path": model_path,
        "text_prompt": text_prompt,
        "chunk_size": 300,
        "iou_threshold": 0.3,
        "confidence_threshold": confidence_threshold,
    },
    "output_paths": {
        "processed_video": str(processed_video.absolute()),
        "mask_dir": str(mask_dir.absolute()),
        "metadata_log": str(metadata_file.absolute()),
    },
    "pipeline": {
        "video_processing": {"status": "pending"},
        "sam3_inference": {"status": "pending"},
    },
}

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)
print(f" Metadata initialized: {session_id}")

# -------------------------
# Command runner
# -------------------------
def run(cmd, step):
    print(f"\n {step}...")
    t0 = datetime.now()
    try:
        subprocess.run(cmd, check=True)
        elapsed = (datetime.now() - t0).total_seconds()
        
        with open(metadata_file) as f: 
            meta = json.load(f)
        step_key = step.lower().replace(" ", "_")
        meta["pipeline"][step_key]["status"] = "completed"
        meta["pipeline"][step_key]["elapsed_s"] = round(elapsed, 2)
        with open(metadata_file, "w") as f: 
            json.dump(meta, f, indent=2)
        
        print(f"✓ {step} completed ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        with open(metadata_file) as f: 
            meta = json.load(f)
        step_key = step.lower().replace(" ", "_")
        meta["pipeline"][step_key]["status"] = "failed"
        meta["pipeline"][step_key]["error"] = str(e)
        with open(metadata_file, "w") as f: 
            json.dump(meta, f, indent=2)
        print(f" {step} failed")
        raise

# -------------------------
# 1. Video processing
# -------------------------
run([
    "psifx", "video", "manipulation", "process",
    "--in_video", str(input_video),
    "--out_video", str(processed_video),
    "--start", str(manual_start),
    "--end", str(manual_end),
], "Video processing")

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
with open(metadata_file) as f: 
    meta = json.load(f)
meta["timestamp_end"] = datetime.now().isoformat()

masks = list(mask_dir.glob("*.mp4")) if mask_dir.exists() else []
mask_size = sum(f.stat().st_size for f in masks) / (1024**2) if masks else 0

meta["output_files"] = {
    "processed_video": str(processed_video),
    "processed_video_size_mb": get_file_size_mb(processed_video),
    "num_masks": len(masks),
    "masks_total_size_mb": round(mask_size, 2),
}

with open(metadata_file, "w") as f: 
    json.dump(meta, f, indent=2)


print(f"Session: {session_id}")
print(f"Input: {input_video} ({meta['input_video'].get('size_mb')} MB)")
print(f"Time: {manual_start}s - {manual_end}s ({meta['time_window']['frames']} frames)")
print(f"Masks: {len(masks)} objects, {meta['output_files']['masks_total_size_mb']} MB")
print(f"Log: {metadata_file}")
