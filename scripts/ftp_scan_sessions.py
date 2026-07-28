from ftplib import FTP, error_perm
import re
import json
import csv


# =====================================================
# LOAD FTP CONFIGURATION
# =====================================================

def load_ftp_config(filename="ftp.txt"):

    config = {}

    with open(filename, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)

            config[key.strip()] = value.strip()

    return config



config = load_ftp_config()


FTP_HOST = config["FTP_HOST"]
FTP_USER = config["FTP_USER"]
FTP_PASS = config["FTP_PASS"]
FTP_ROOT = config["FTP_ROOT"]



# =====================================================
# SESSION PATTERN
# =====================================================

pattern = re.compile(
    r'^(\d{1,2})-(\d{1,2})-(\d{4})_#(\d+)([A-Za-z0-9_]*)_(GROUP|INDIVIDUAL)_\[([\d\-]+)\]$'
)



# =====================================================
# GET FTP CONTENT
# Compatible with older FTP servers
# =====================================================

def get_directory_content(ftp, path):

    files = []
    folders = []

    ftp.cwd(path)

    items = ftp.nlst()


    for item in items:

        try:

            current = ftp.pwd()

            ftp.cwd(item)

            folders.append(item)

            ftp.cwd(current)


        except error_perm:

            files.append(item)

            ftp.cwd(path)


    return files, folders



# =====================================================
# BUILD COMPLETE TREE
# =====================================================

def build_tree(ftp, path):

    tree = {

        "files": [],

        "folders": {}

    }


    try:

        files, folders = get_directory_content(
            ftp,
            path
        )


        tree["files"] = files


        for folder in folders:

            print(
                "Scanning:",
                f"{path}/{folder}"
            )


            tree["folders"][folder] = build_tree(
                ftp,
                f"{path.rstrip('/')}/{folder}"
            )


    except error_perm as e:

        print(
            "Cannot access:",
            path,
            e
        )


    return tree



# =====================================================
# COLLECT ALL FILES
# =====================================================

def collect_files(tree):

    files = list(tree["files"])


    for folder in tree["folders"].values():

        files.extend(
            collect_files(folder)
        )


    return files



# =====================================================
# COLLECT ALL FOLDERS
# =====================================================

def collect_folders(tree):

    folders = []


    for name, content in tree["folders"].items():

        folders.append(name)

        folders.extend(
            collect_folders(content)
        )


    return folders



# =====================================================
# SCAN SESSIONS
# =====================================================

def scan_sessions(ftp, path):

    results = []


    tree = build_tree(
        ftp,
        path
    )


    folder_name = path.rstrip("/").split("/")[-1]



    # -------------------------------------------------
    # Is this a session folder?
    # -------------------------------------------------

    if pattern.fullmatch(folder_name):


        files = collect_files(tree)

        folders = collect_folders(tree)



        mkv_files = [
            f for f in files
            if f.lower().endswith(".mkv")
        ]


        audio_files = [
            f for f in files
            if f.lower().endswith(
                (".wav", ".m4a")
            )
        ]


        json_files = [
            f for f in files
            if f.lower().endswith(".json")
        ]


        eaf_files = [
            f for f in files
            if f.lower().endswith(".eaf")
        ]


        txt_files = [
            f for f in files
            if f.lower().endswith(".txt")
        ]



        match = pattern.fullmatch(
            folder_name
        )


        day, month, year, number, suffix, session_type, child_id = match.groups()



        results.append(

            {

                "session_id":
                    folder_name,


                "session_date":
                    f"{year}-{int(month):02d}-{int(day):02d}",


                "child_id":
                    child_id,


                "session_type":
                    session_type,


                "session_number":
                    f"{number}{suffix}",



                # Complete FTP structure

                "folder_structure":
                    tree,


                "all_files":
                    files,


                "all_folders":
                    folders,



                # File categories

                "mkv_files":
                    mkv_files,


                "audio_files":
                    audio_files,


                "json_files":
                    json_files,


                "eaf_files":
                    eaf_files,


                "txt_files":
                    txt_files,



                # Availability

                "bounding_boxes_folder_available":
                    any(
                        x.lower() == "bounding_boxes"
                        for x in folders
                    ),


                "skeletons_folder_available":
                    any(
                        x.lower() == "skeletons"
                        for x in folders
                    ),


                "Visualizations_folder_available":
                    any(
                        x.lower() == "visualizations"
                        for x in folders
                    ),


                "config_reid_json_available":
                    any(
                        x.lower() == "config_reid.json"
                        for x in files
                    )

            }

        )



    # Continue recursively

    for folder in tree["folders"]:

        results.extend(

            scan_sessions(
                ftp,
                f"{path.rstrip('/')}/{folder}"
            )

        )


    return results



# =====================================================
# CONNECT FTP
# =====================================================

print("Connecting FTP...")


ftp = FTP(
    FTP_HOST
)


ftp.encoding = "latin-1"


ftp.login(
    FTP_USER,
    FTP_PASS
)


print(
    "Connected"
)


print(
    "FTP location:",
    ftp.pwd()
)



# Test root

ftp.cwd(
    FTP_ROOT
)


print(
    "Scanning root:",
    ftp.pwd()
)



# =====================================================
# RUN SCAN
# =====================================================

sessions = scan_sessions(
    ftp,
    FTP_ROOT
)



ftp.quit()



print(
    f"Found {len(sessions)} sessions"
)



# =====================================================
# SAVE JSON
# =====================================================

with open(
    "sessions_inventory.json",
    "w",
    encoding="utf-8"
) as f:


    json.dump(
        sessions,
        f,
        indent=4,
        ensure_ascii=False
    )



print(
    "Saved sessions_inventory.json"
)



# =====================================================
# SAVE CSV SUMMARY
# =====================================================

columns = [

    "session_id",
    "session_date",
    "child_id",
    "session_type",
    "session_number",

    "mkv_files",
    "audio_files",
    "eaf_files",
    "txt_files",

    "bounding_boxes_folder_available",
    "skeletons_folder_available",
    "Visualizations_folder_available",
    "config_reid_json_available"

]



with open(
    "sessions_inventory.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=columns
    )


    writer.writeheader()


    for row in sessions:

        writer.writerow(

            {

                key:
                    "; ".join(row[key])
                    if isinstance(row[key], list)
                    else row[key]

                for key in columns

            }

        )



print(
    "Saved sessions_inventory.csv"
)


print("Finished")
