import os
import json
import csv

def create_project_structure(processed_root="ROOT_Directory_Processed"):
    # 1. Define folder hierarchy
    folders = [
        os.path.join(processed_root, "GENERAL_FILES"),
        os.path.join(processed_root, "SESSIONS")
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"[OK] Created directory: {folder}")

    # 2. Initialize GENERAL_FILES/children_table.csv
    children_csv = os.path.join(processed_root, "GENERAL_FILES", "children_table.csv")
    if not os.path.exists(children_csv):
        with open(children_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["child_id", "age_range", "session_type", "nb_sessions", "notes"])
        print(f"[OK] Initialized: {children_csv}")

    # 3. Initialize GENERAL_FILES/sessions_list.json
    sessions_json = os.path.join(processed_root, "GENERAL_FILES", "sessions_list.json")
    if not os.path.exists(sessions_json):
        sample_sessions = {
            "15_6": {
                "child_id": "15",
                "session_number": 6,
                "session_type": "INDIVIDUAL",
                "audio_offset_ms": 0,
                "people_present": ["child", "therapist"],
                "is_annotated": True
            }
        }
        with open(sessions_json, "w", encoding="utf-8") as f:
            json.dump(sample_sessions, f, indent=2)
        print(f"[OK] Initialized: {sessions_json}")

    # 4. Initialize GENERAL_FILES/tracking_validation.json
    validation_json = os.path.join(processed_root, "GENERAL_FILES", "tracking_validation.json")
    if not os.path.exists(validation_json):
        sample_validation = {
            "15_6": {
                "status": "pending",
                "notes": "Initial setup"
            }
        }
        with open(validation_json, "w", encoding="utf-8") as f:
            json.dump(sample_validation, f, indent=2)
        print(f"[OK] Initialized: {validation_json}")

    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
