import json
import cv2
import os
import argparse

def render_validation_video(session_id, processed_root="ROOT_Directory_Processed", raw_root="ROOT_Directory_Raw"):
    """
    Overlays tracking bounding boxes (SAM3) and ELAN manual annotations onto 
    the raw session video for visual validation and quality check.
    """
    session_dir = os.path.join(processed_root, "SESSIONS", session_id)
    
    # Define file paths based on the project documentation structure
    raw_video_path = os.path.join(raw_root, "SESSIONS", session_id, "raw_video.mp4")
    tracking_file = os.path.join(session_dir, "tracking", "tracks.json")
    annotation_file = os.path.join(session_dir, "annotations", f"{session_id}_annotations.json")
    out_dir = os.path.join(session_dir, "validation")
    os.makedirs(out_dir, exist_ok=True)
    
    out_video_path = os.path.join(out_dir, "validation_rendered.mp4")
    
    # Check if tracking metadata exists
    if not os.path.exists(tracking_file):
        print(f"[Error] Tracking file missing for session '{session_id}' at: {tracking_file}")
        return

    # Load SAM3 tracking output
    with open(tracking_file, 'r', encoding='utf-8') as f:
        tracking_data = json.load(f)

    # Load parsed ELAN annotations if present
    annotations_data = {}
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)

    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open raw video file: {raw_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    print(f"[Processing] Rendering validation video for session '{session_id}'...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time_sec = frame_idx / fps if fps > 0 else 0
        frame_key = str(frame_idx)
        
        # 1. Render SAM3 Bounding Boxes and Tracking IDs
        if frame_key in tracking_data.get("frames", {}):
            for obj in tracking_data["frames"][frame_key]:
                track_id = obj["track_id"]
                role = obj.get("role", "unknown")  # 'child' or 'therapist'
                bbox = obj["bbox"]                 # [x, y, w, h]
                
                # Green box for child, Blue/Orange for clinician
                color = (0, 255, 0) if role == "child" else (255, 165, 0)
                x, y, w, h = map(int, bbox)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{track_id} ({role})", (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. Render Active ELAN Tiers (Gaze, Attention, Position)
        if annotations_data:
            active_tiers = []
            for item in annotations_data.get("annotations", []):
                if item["start_sec"] <= current_time_sec <= item["end_sec"]:
                    active_tiers.append(f"{item['tier']}: {item['value']}")
            
            # Draw timestamp and active annotation tiers in the top-left corner
            y_offset = 30
            cv2.putText(frame, f"Time: {current_time_sec:.2f}s", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for tier_info in active_tiers[:5]:  # Display up to 5 active tiers
                y_offset += 25
                cv2.putText(frame, tier_info, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[Success] Validation video generated at: {out_video_path}")

def update_validation_registry(session_id, status, notes="", processed_root="ROOT_Directory_Processed"):
    """
    Updates or appends the validation state in tracking_validation.json.
    """
    val_file = os.path.join(processed_root, "GENERAL_FILES", "tracking_validation.json")
    
    registry = {}
    if os.path.exists(val_file):
        with open(val_file, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
    registry[session_id] = {
        "status": status,  # "valid" or "needs_correction"
        "notes": notes
    }
    
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    print(f"[Updated Registry] Session '{session_id}' status set to '{status}' in tracking_validation.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render validation video and log QA status for CHUV pipeline.")
    parser.add_argument("--session_id", required=True, help="Target session ID (e.g., 15_6)")
    parser.add_argument("--status", choices=["valid", "needs_correction"], default=None, help="Set validation status")
    parser.add_argument("--notes", default="", help="Optional QA notes or comments")
    args = parser.parse_args()
    
    render_validation_video(args.session_id)
    if args.status:
        update_validation_registry(args.session_id, args.status, args.notes)
