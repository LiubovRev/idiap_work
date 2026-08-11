#!/usr/bin/env python3
"""
inspect_masks.py

Interactive mask inspector. Play through each mask video, decide:
  - Who is this track? (child / therapist / ignore)
  - Should it be merged with another track? (yes / no / review)
  - Save decisions to a JSON config file

Usage:
    python inspect_masks.py --session_dir /path/to/7_INDIVIDUAL_14 --output_config mask_decisions.json

Keys while playing:
  - SPACE: pause/resume
  - ARROW LEFT/RIGHT: frame-by-frame
  - M: mark for merge
  - D: mark for delete/ignore
  - K: keep as-is
  - N: next track
  - Q: quit
"""

import cv2
import json
import argparse
from pathlib import Path
from collections import defaultdict


class MaskInspector:
    def __init__(self, mask_dir, output_config):
        self.mask_dir = Path(mask_dir)
        self.output_config = output_config
        self.masks = sorted([f for f in self.mask_dir.glob("*.mp4") if f.name[0].isdigit()])
        self.decisions = {}
        self.current_track_idx = 0
        self.paused = False
        self.current_frame = 0
        
    def run(self):
        """Main inspection loop."""
        print(f"\n{'='*70}")
        print(f"Found {len(self.masks)} mask tracks:")
        for i, m in enumerate(self.masks):
            print(f"  {i}: {m.name}")
        print(f"{'='*70}\n")
        
        for idx, mask_path in enumerate(self.masks):
            self.current_track_idx = idx
            self.current_frame = 0
            self.paused = False
            self.play_mask_video(mask_path)
    
    def play_mask_video(self, mask_path):
        """Play a single mask video with interactive controls."""
        cap = cv2.VideoCapture(str(mask_path))
        if not cap.isOpened():
            print(f"[error] Could not open {mask_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        track_id = mask_path.stem
        print(f"\n[Track {self.current_track_idx}] {mask_path.name}")
        print(f"  Duration: {total_frames} frames @ {fps:.1f} fps ({total_frames/fps:.1f} sec)")
        print(f"  Resolution: {width}x{height}")
        print(f"\n  CONTROLS:")
        print(f"    SPACE     - pause/resume")
        print(f"    LEFT/RIGHT - frame-by-frame (when paused)")
        print(f"    K         - Keep this track as-is")
        print(f"    M         - Mark for merge (with another track)")
        print(f"    D         - Mark for delete/ignore")
        print(f"    N         - Next track")
        print(f"    Q         - Quit & save decisions")
        print(f"\n  Status: [press a key to begin]\n")
        
        frame_idx = 0
        decision = None
        merge_target = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if decision is None:
                # Display frame with info overlay
                display = frame.copy()
                
                # Add text overlay
                time_sec = frame_idx / fps
                status_text = "PAUSED" if self.paused else "PLAYING"
                info = [
                    f"Track: {track_id}  Frame: {frame_idx}/{total_frames}  Time: {time_sec:.2f}s  [{status_text}]",
                    "Press K=Keep, M=Merge, D=Delete, N=Next, Q=Quit"
                ]
                
                for i, text in enumerate(info):
                    y = 30 + i * 25
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow(f"Mask Inspector - Track {self.current_track_idx}: {track_id}", display)
                
                # Advance frame if not paused
                if not self.paused:
                    frame_idx += 1
                
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                
                if key == ord('q'):
                    cap.release()
                    return 'quit'
                elif key == ord('k'):
                    print(f"  ✓ Decision: KEEP")
                    self.decisions[track_id] = {
                        "action": "keep",
                        "role": "unknown (manual assignment needed)",
                        "notes": "Inspect visually and update role"
                    }
                    decision = 'keep'
                elif key == ord('m'):
                    print(f"  ⚠ Decision: MERGE (specify merge target when done)")
                    self.decisions[track_id] = {
                        "action": "merge",
                        "merge_with": None,  # User fills in later
                        "notes": "Decide merge target after reviewing all tracks"
                    }
                    decision = 'merge'
                elif key == ord('d'):
                    print(f"  ✗ Decision: DELETE/IGNORE")
                    self.decisions[track_id] = {
                        "action": "delete",
                        "reason": "Track marked for deletion by user",
                        "notes": ""
                    }
                    decision = 'delete'
                elif key == ord('n'):
                    print(f"  → Skipping to next track (no decision recorded)")
                    cap.release()
                    return 'next'
                elif key == ord(' '):
                    self.paused = not self.paused
                elif key == 83:  # RIGHT arrow
                    if self.paused:
                        frame_idx = min(frame_idx + 1, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                elif key == 81:  # LEFT arrow
                    if self.paused:
                        frame_idx = max(frame_idx - 1, 0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            else:
                # Decision made, move to next track
                cap.release()
                cv2.destroyAllWindows()
                return 'next'
        
        cap.release()
    
    def save_decisions(self):
        """Save all decisions to JSON."""
        output_path = Path(self.output_config)
        with open(output_path, 'w') as f:
            json.dump(self.decisions, f, indent=2)
        print(f"\n[ok] Decisions saved to: {output_path}")
        print(json.dumps(self.decisions, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect mask videos and record manual decisions (keep/merge/delete)"
    )
    parser.add_argument(
        "--session_dir",
        required=True,
        help="Path to session directory (e.g., /path/to/7_INDIVIDUAL_14)"
    )
    parser.add_argument(
        "--mask_subdir",
        default="MaskDir",
        help="Subdirectory containing masks (default: MaskDir)"
    )
    parser.add_argument(
        "--output_config",
        default="mask_decisions.json",
        help="Output JSON file with your decisions"
    )
    args = parser.parse_args()
    
    mask_dir = Path(args.session_dir) / args.mask_subdir
    if not mask_dir.exists():
        print(f"[error] Mask directory not found: {mask_dir}")
        return
    
    inspector = MaskInspector(mask_dir, args.output_config)
    inspector.run()
    inspector.save_decisions()


if __name__ == "__main__":
    main()
