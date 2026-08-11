from ftplib import FTP, error_perm
import re
import pandas as pd

# =====================================================
# FTP SETTINGS
# =====================================================

FTP_HOST = "fileftp.intranet.chuv"
FTP_USER = "user"
FTP_PASS = "password"
FTP_ROOT = "/filearc/DRM/EEGFENL/BackupFolder/Naomi"

# =====================================================
# SESSION FOLDER PATTERN
# =====================================================

pattern = re.compile(
    r'^(\d{1,2})-(\d{1,2})-(\d{4})_#(\d+)([A-Za-z0-9_]*)_(GROUP|INDIVIDUAL)_\[([\d\-]+)\]$'
)

# =====================================================
# RECURSIVE FTP SCANNER
# =====================================================

def scan_ftp_folder(ftp, path):

    results = []

    try:
        ftp.cwd(path)
        items = ftp.nlst()

        folders = []
        files = []

        for item in items:
            full_path = f"{path.rstrip('/')}/{item}"

            try:
                ftp.cwd(full_path)
                folders.append(item)
                ftp.cwd(path)

            except error_perm:
                files.append(item)
                ftp.cwd(path)

        folder_name = path.rstrip("/").split("/")[-1]

        # -------------------------------------------------
        # Check whether this is a session folder
        # -------------------------------------------------

        if pattern.fullmatch(folder_name):

            mkv_files = [
                f for f in files
                if f.lower().endswith(".mkv")
            ]

            json_files = [
                f for f in files
                if f.lower().endswith(".json")
            ]

            audio_files = [
                f for f in files
                if f.lower().endswith((".wav", ".m4a"))
            ]

            eaf_files = [
                f for f in files
                if f.lower().endswith(".eaf")
            ]

            txt_files = [
                f for f in files
                if f.lower().endswith(".txt")
            ]

            bounding_boxes_available = any(
                folder.lower() == "bounding_boxes"
                for folder in folders
            )

            skeletons_available = any(
                folder.lower() == "skeletons"
                for folder in folders
            )

            visualizations_available = any(
                folder.lower() == "visualizations"
                for folder in folders
            )

            config_reid_available = any(
                f.lower() == "config_reid.json"
                for f in files
            )

            results.append(
                {
                    "session_id": folder_name,

                    "mkv_files_count": len(mkv_files),

                    "metadata_json_available":
                        len(json_files) > 0,

                    "audio_file_count":
                        len(audio_files),

                    "eaf_available":
                        len(eaf_files) > 0,

                    "other_folders_count":
                        len(folders),

                    "bounding_boxes_folder_available":
                        bounding_boxes_available,

                    "skeletons_folder_available":
                        skeletons_available,

                    "Visualizations_folder_available":
                        visualizations_available,

                    "config_reid.json_available":
                        config_reid_available,

                    "txt_files_count":
                        len(txt_files),

                    "folders_count":
                        len(folders),
                }
            )

        # -------------------------------------------------
        # Continue scanning recursively
        # -------------------------------------------------

        for folder in folders:

            print("Scanning:", f"{path}/{folder}")

            results.extend(
                scan_ftp_folder(
                    ftp,
                    f"{path.rstrip('/')}/{folder}"
                )
            )

    except error_perm as e:
        print(f"Cannot access {path}: {e}")

    return results


# =====================================================
# CONNECT TO FTP
# =====================================================

print("Connecting to FTP...")

ftp = FTP(FTP_HOST)
ftp.encoding = "latin-1"

ftp.login(
    FTP_USER,
    FTP_PASS
)

print("Connected.")
print("Scanning folders...")

sessions = scan_ftp_folder(
    ftp,
    FTP_ROOT
)

ftp.quit()

print(f"Found {len(sessions)} session folders")

# =====================================================
# BUILD TABLE
# =====================================================

rows = []

for session in sessions:

    session_id = session["session_id"]

    match = pattern.fullmatch(session_id)

    if not match:
        continue

    day, month, year, number, suffix, session_type, child_id = match.groups()

    rows.append(
        {
            # Session information
            "session_id":
                session_id,

            "session_date":
                f"{year}-{int(month):02d}-{int(day):02d}",

            "child_id":
                child_id,

            "session_type":
                session_type,

            "session_number":
                f"{number}{suffix}",

            # Manual fields
            "audio_available":
                "",

            "time_offset_ms":
                "",

            "coded_bei_xuan":
                "",

            "coded_emily":
                "",

            "combined_file":
                "",

            "psifx_processed":
                "",

            # Automatically detected
            "mkv_files_count":
                session["mkv_files_count"],

            "metadata_json_available":
                session["metadata_json_available"],

            "audio_file_count":
                session["audio_file_count"],

            "eaf_available":
                session["eaf_available"],

            "other_folders_count":
                session["other_folders_count"],

            "bounding_boxes_folder_available":
                session["bounding_boxes_folder_available"],

            "skeletons_folder_available":
                session["skeletons_folder_available"],

            "Visualizations_folder_available":
                session["Visualizations_folder_available"],

            "config_reid.json_available":
                session["config_reid.json_available"],

            "txt_files_count":
                session["txt_files_count"],

            "folders_count":
                session["folders_count"],
        }
    )

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(rows)

df = df[
    [
        "session_id",
        "session_date",
        "child_id",
        "session_type",
        "session_number",

        "audio_available",
        "time_offset_ms",
        "coded_bei_xuan",
        "coded_emily",
        "combined_file",
        "psifx_processed",

        "mkv_files_count",
        "metadata_json_available",
        "audio_file_count",
        "eaf_available",
        "other_folders_count",
        "bounding_boxes_folder_available",
        "skeletons_folder_available",
        "Visualizations_folder_available",
        "config_reid.json_available",
        "txt_files_count",
        "folders_count",
    ]
]

# =====================================================
# SAVE CSV
# =====================================================

output_file = "sessions_inventory2.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nFinished")
print("--------------------------------")
print(f"CSV created : {output_file}")
print(f"Sessions    : {len(df)}")

print("\nPreview:")
print(df.head(10).to_string(index=False))
