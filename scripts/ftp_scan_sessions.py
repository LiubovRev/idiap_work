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
                # Try entering item.
                # Success means it is a folder.

                ftp.cwd(full_path)

                folders.append(item)

                ftp.cwd(path)


            except error_perm:

                # Otherwise it is a file

                files.append(item)

                ftp.cwd(path)



        folder_name = path.rstrip("/").split("/")[-1]


        # ---------------------------------------------
        # Check if this is a session folder
        # ---------------------------------------------

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
                if (
                    f.lower().endswith(".wav")
                    or f.lower().endswith(".m4a")
                )
            ]


            eaf_files = [
                f for f in files
                if f.lower().endswith(".eaf")
            ]



            results.append(
                {
                    "session_id": folder_name,

                    # Video
                    "mkv_files":
                        "; ".join(mkv_files),

                    "mkv_count":
                        len(mkv_files),


                    # Metadata
                    "metadata_json_available":
                        len(json_files) > 0,


                    # Audio
                    "audio_files":
                        "; ".join(audio_files),

                    "audio_file_count":
                        len(audio_files),


                    # Annotation
                    "eaf_available":
                        len(eaf_files) > 0,


                    # Other folders inside session folder
                    "other_folders":
                        "; ".join(folders),
                }
            )



        # ---------------------------------------------
        # Continue scanning subfolders
        # ---------------------------------------------

        for folder in folders:

            print("Scanning:", f"{path}/{folder}")

            results.extend(
                scan_ftp_folder(
                    ftp,
                f"{path.rstrip('/')}/{folder}"
                )
        )



    except error_perm as e:

        print(
            f"Cannot access {path}: {e}"
        )


    return results




# =====================================================
# CONNECT TO FTP
# =====================================================

print("Connecting to FTP...")

ftp = FTP(FTP_HOST)

# Many FTP servers use latin-1 encoding for filenames
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


print(
    f"Found {len(sessions)} session folders"
)



# =====================================================
# CREATE CSV
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

            # -----------------------------
            # Original columns
            # -----------------------------

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



            # -----------------------------
            # FTP inventory columns
            # -----------------------------

            "mkv_files":
                session["mkv_files"],


            "mkv_count":
                session["mkv_count"],


            "metadata_json_available":
                session["metadata_json_available"],


            "audio_files":
                session["audio_files"],


            "audio_file_count":
                session["audio_file_count"],


            "eaf_available":
                session["eaf_available"],


            "other_folders":
                session["other_folders"],

        }
    )



# =====================================================
# SAVE CSV
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


        "mkv_files",

        "mkv_count",


        "metadata_json_available",


        "audio_files",

        "audio_file_count",


        "eaf_available",


        "other_folders",

    ]
]



output_file = "sessions_inventory.csv"


df.to_csv(
    output_file,
    index=False
)



print("\nFinished")
print("----------------------------")
print(f"CSV created : {output_file}")
print(f"Sessions    : {len(df)}")


print("\nPreview:")
print(
    df.head(10).to_string(index=False)
)
