#!/usr/bin/env python3
"""
Validation & QA Registration Script (T4 Phase)

Purpose:
  - Check if a processed session passes quality thresholds
  - Register session status in tracking_validation.json
  - Prepare summary report for human review

Usage:
  python validate_session.py \\
    --session_id 15_6 \\
    --status valid \\
    --reviewer alice \\
    --tracking_validation_file GENERAL_FILES/tracking_validation.json \\
    --notes "Masks stable throughout. Ready for analysis."
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class SessionValidator:
    """Validates a processed session and registers status."""
    
    def __init__(self, session_id: str, output_base: str = "ROOT_Directory_Processed"):
        self.session_id = session_id
        self.output_base = Path(output_base)
        self.session_dir = self.output_base / "SESSIONS" / session_id
    
    def check_pipeline_stages(self) -> Dict[str, bool]:
        """
        Verify which pipeline stages have been completed for this session.
        
        Returns:
            Dict mapping stage name to completion status (True/False)
        """
        stages = {
            "annotation_parsing": (self.session_dir / "annotations" / f"{self.session_id}_annotations.json").exists(),
            "sam3_tracking": (self.session_dir / "tracking" / "masks").exists(),
            "bbox_extraction": (self.session_dir / "tracking" / "bboxes" / f"{self.session_id}_bboxes.json").exists(),
            "pose_extraction": (self.session_dir / "features" / "skeleton").exists(),
            "gaze_extraction": (self.session_dir / "features" / "heads").exists(),
            "visualization": (self.session_dir / "validation" / "validation_rendered.mp4").exists(),
            "validation_report": (self.session_dir / "validation" / "validation_report.json").exists(),
        }
        return stages
    
    def check_annotation_quality(self) -> Dict[str, any]:
        """
        Load annotation JSON and compute coverage metrics.
        
        Returns:
            Dict with coverage stats and any warnings
        """
        annot_file = self.session_dir / "annotations" / f"{self.session_id}_annotations.json"
        
        if not annot_file.exists():
            return {"status": "FILE_NOT_FOUND", "coverage_pct": 0}
        
        try:
            with open(annot_file) as f:
                annot = json.load(f)
            
            total_dur = annot.get("metadata", {}).get("total_session_duration_sec", 0)
            coverage = annot.get("metadata", {}).get("annotation_coverage_pct", 0)
            
            # Tier-level breakdown
            tier_coverage = {}
            if "tiers" in annot:
                for tier_name, tier_data in annot["tiers"].items():
                    total_tier_dur = sum(e.get("duration_sec", 0) for e in tier_data.get("events", []))
                    tier_coverage[tier_name] = {
                        "duration_sec": total_tier_dur,
                        "pct_of_session": (total_tier_dur / total_dur * 100) if total_dur > 0 else 0
                    }
            
            return {
                "status": "OK",
                "total_session_duration_sec": total_dur,
                "overall_coverage_pct": coverage,
                "tier_coverage": tier_coverage,
                "n_tiers": len(tier_coverage),
                "warnings": []
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    
    def check_mask_quality(self) -> Dict[str, any]:
        """
        Load mask files and check frame-by-frame coverage.
        
        Returns:
            Dict with mask completeness stats
        """
        import numpy as np
        
        masks_dir = self.session_dir / "tracking" / "masks"
        results = {"status": "OK", "tracks": {}}
        
        if not masks_dir.exists():
            return {"status": "MASKS_DIR_NOT_FOUND"}
        
        # Check child mask
        child_mask_file = masks_dir / f"{self.session_id}_c_mask_frames.npz"
        if child_mask_file.exists():
            try:
                masks_c = np.load(child_mask_file)
                frames_c = masks_c.get("frames", None)
                if frames_c is not None:
                    n_frames_with_mask = (frames_c.max(axis=(1, 2)) > 0).sum()
                    pct = (n_frames_with_mask / len(frames_c) * 100) if len(frames_c) > 0 else 0
                    results["tracks"][f"{self.session_id}_c"] = {
                        "n_frames_total": len(frames_c),
                        "n_frames_with_mask": int(n_frames_with_mask),
                        "coverage_pct": float(pct),
                        "status": "PASS" if pct > 95 else "WARN"
                    }
            except Exception as e:
                results["tracks"][f"{self.session_id}_c"] = {"status": "ERROR", "message": str(e)}
        
        # Check therapist mask
        therapist_mask_file = masks_dir / f"{self.session_id}_t_mask_frames.npz"
        if therapist_mask_file.exists():
            try:
                masks_t = np.load(therapist_mask_file)
                frames_t = masks_t.get("frames", None)
                if frames_t is not None:
                    n_frames_with_mask = (frames_t.max(axis=(1, 2)) > 0).sum()
                    pct = (n_frames_with_mask / len(frames_t) * 100) if len(frames_t) > 0 else 0
                    results["tracks"][f"{self.session_id}_t"] = {
                        "n_frames_total": len(frames_t),
                        "n_frames_with_mask": int(n_frames_with_mask),
                        "coverage_pct": float(pct),
                        "status": "PASS" if pct > 90 else "WARN"
                    }
            except Exception as e:
                results["tracks"][f"{self.session_id}_t"] = {"status": "ERROR", "message": str(e)}
        
        return results
    
    def check_gaze_quality(self) -> Dict[str, any]:
        """
        Load gaze JSON files and check confidence scores.
        
        Returns:
            Dict with gaze quality metrics
        """
        heads_dir = self.session_dir / "features" / "heads"
        results = {"status": "OK", "tracks": {}}
        
        if not heads_dir.exists():
            return {"status": "HEADS_DIR_NOT_FOUND"}
        
        for track_id in [f"{self.session_id}_c", f"{self.session_id}_t"]:
            gaze_file = heads_dir / f"{track_id}_gaze_3d.json"
            
            if gaze_file.exists():
                try:
                    with open(gaze_file) as f:
                        gaze_data = json.load(f)
                    
                    confidences = [
                        gaze_data["frames"][frame_str]["gaze_confidence"]
                        for frame_str in gaze_data["frames"].keys()
                    ]
                    
                    if confidences:
                        mean_conf = sum(confidences) / len(confidences)
                        min_conf = min(confidences)
                        n_high_conf = sum(1 for c in confidences if c > 0.85)
                        
                        results["tracks"][track_id] = {
                            "n_frames": len(confidences),
                            "mean_confidence": float(mean_conf),
                            "min_confidence": float(min_conf),
                            "frames_with_high_confidence_pct": (n_high_conf / len(confidences) * 100) if confidences else 0,
                            "status": "PASS" if mean_conf > 0.85 else "WARN"
                        }
                except Exception as e:
                    results["tracks"][track_id] = {"status": "ERROR", "message": str(e)}
        
        return results
    
    def load_validation_report(self) -> Optional[Dict]:
        """Load the validation report generated during visualization."""
        report_file = self.session_dir / "validation" / "validation_report.json"
        if report_file.exists():
            try:
                with open(report_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load validation report: {e}")
                return None
        return None
    
    def generate_qc_summary(self) -> Dict:
        """
        Run all QC checks and generate comprehensive summary.
        
        Returns:
            Dict with all QC results
        """
        summary = {
            "session_id": self.session_id,
            "qc_timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline_stages": self.check_pipeline_stages(),
            "annotations": self.check_annotation_quality(),
            "masks": self.check_mask_quality(),
            "gaze": self.check_gaze_quality(),
            "validation_report": self.load_validation_report(),
        }
        
        # Aggregate status
        all_pass = all(v for v in summary["pipeline_stages"].values()) and \
                   all(
                       t.get("status") in ["PASS", "OK"]
                       for t in summary["masks"].get("tracks", {}).values()
                   )
        
        summary["overall_readiness"] = "READY_FOR_ANALYSIS" if all_pass else "NEEDS_REVIEW"
        
        return summary
    
    def register_status(
        self,
        status: str,
        reviewer: str,
        tracking_validation_file: str,
        notes: str = "",
        manual_flags: Optional[List[str]] = None,
    ) -> None:
        """
        Register session validation status in the master tracking file.
        
        Args:
            status: One of 'valid', 'needs_correction', 'in_progress', 'missing_assets'
            reviewer: Name of reviewer
            tracking_validation_file: Path to GENERAL_FILES/tracking_validation.json
            notes: Optional human-readable notes
            manual_flags: List of manual flags/issues found during review
        """
        tracking_file = Path(tracking_validation_file)
        tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing records
        if tracking_file.exists():
            with open(tracking_file) as f:
                data = json.load(f)
        else:
            data = {"validation_records": []}
        
        # Create new record
        record = {
            "session_id": self.session_id,
            "status": status,
            "reviewer": reviewer,
            "review_timestamp": datetime.utcnow().isoformat() + "Z",
            "notes": notes,
            "manual_flags": manual_flags or [],
            "pipeline_stages_complete": [
                stage for stage, complete in self.check_pipeline_stages().items() if complete
            ],
            "validation_report_file": str(self.session_dir / "validation" / "validation_report.json"),
            "validation_video_file": str(self.session_dir / "validation" / "validation_rendered.mp4"),
        }
        
        # Update or append
        existing_idx = None
        for i, rec in enumerate(data["validation_records"]):
            if rec["session_id"] == self.session_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            data["validation_records"][existing_idx] = record
            print(f"Updated existing record for {self.session_id}")
        else:
            data["validation_records"].append(record)
            print(f"Added new record for {self.session_id}")
        
        # Write back
        with open(tracking_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Registered {self.session_id} as '{status}' in {tracking_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate and register session processing status")
    parser.add_argument("--session_id", required=True, help="Session ID (e.g., 15_6)")
    parser.add_argument(
        "--status",
        required=True,
        choices=["valid", "needs_correction", "in_progress", "missing_assets"],
        help="Validation status"
    )
    parser.add_argument("--reviewer", required=True, help="Reviewer name")
    parser.add_argument(
        "--tracking_validation_file",
        default="ROOT_Directory_Processed/GENERAL_FILES/tracking_validation.json",
        help="Path to tracking validation JSON"
    )
    parser.add_argument("--notes", default="", help="Optional review notes")
    parser.add_argument(
        "--output_base",
        default="ROOT_Directory_Processed",
        help="Base output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed QC summary")
    
    args = parser.parse_args()
    
    validator = SessionValidator(args.session_id, args.output_base)
    
    # Generate QC summary
    qc_summary = validator.generate_qc_summary()
    
    if args.verbose:
        print("\n" + "="*60)
        print(f"QC SUMMARY: {args.session_id}")
        print("="*60)
        print(json.dumps(qc_summary, indent=2))
        print("="*60 + "\n")
    
    # Register status
    validator.register_status(
        status=args.status,
        reviewer=args.reviewer,
        tracking_validation_file=args.tracking_validation_file,
        notes=args.notes,
    )
    
    print(f"Session {args.session_id} is ready for next phase.")


if __name__ == "__main__":
    main()
