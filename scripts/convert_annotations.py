#!/usr/bin/env python3
"""
convert_annotations.py
=======================

Converts existing clinical annotation files (tab-delimited text exports
from ELAN, e.g. `10-1-2024__6_INDIVIDUAL__15__WAKEE_17_10_25_BL.txt`)
into a unified JSON/CSV format compatible with the Phase 0 schema
(Children / Sessions / Tracking+features).

------------------------------------------------------------------
INPUT FORMAT (verified against a real file)
------------------------------------------------------------------
Each line is one annotation, 9 tab-separated columns, no header:

    tier <TAB> "" <TAB> start_hms <TAB> start_sec <TAB> end_hms <TAB> end_sec
        <TAB> dur_hms <TAB> dur_sec <TAB> value

Example:
    c15_CP\t\t00:04:14.273\t254.273\t00:04:44.726\t284.726\t00:00:30.453\t30.453\tCST

The "tier" name encodes the track:
    - "ST"            -> session-level tier (e.g. session boundaries)
    - "c<id>_<CODE>"  -> child track with child_id=<id>, annotation tag <CODE>
    - "t<n>_<CODE>"   -> therapist/adult track #<n>, annotation tag <CODE>
    (other prefixes are kept as-is with track_type="other")

Real tag examples (CP, CAO — confirmed against Odobez's meeting/PDF notes):
    c15_CP  -> Child Position:  CST (standing), CSI (sitting), CHO (leaning over)...
    c15_CAO -> Child Attending to Objects: AO
    c15_JA  -> Joint Attention: TC ...

------------------------------------------------------------------
FILENAME -> session_id / child_id
------------------------------------------------------------------
Expected (flexible) filename pattern:
    <date>_#<session>_<INDIVIDUAL|GROUP>_[<child_id>]_..._BL.<ext>
(the characters # [ ] . may be replaced with "_" by an upload system —
 the parser tolerates both variants).

------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------
1) <out_dir>/json/<session_id>.json   — one file per session
2) <out_dir>/annotations_master.csv   — all annotations, all sessions, one row per annotation

If you pass --sessions-csv (a CSV export of the "Sessions" sheet from
Phase 0 — the same one built in OmniHead_CHUV_Data_Tables.xlsx), the
script looks up time_offset_ms by session_id and adds audio-aligned
start/end times to every annotation (video_ms - offset_ms), flagging
those that precede the start of the audio.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
    python convert_annotations.py \\
        --input-dir /path/to/annotation_exports \\
        --out-dir   /path/to/output \\
        [--sessions-csv /path/to/sessions.csv] \\
        [--tier-map /path/to/tier_code_map.json]

No external dependencies — Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Filename parsing -> session metadata
# ---------------------------------------------------------------------------

FILENAME_SESSION_RE = re.compile(r"[#_](\d{1,4})[_]+(INDIVIDUAL|GROUP)", re.IGNORECASE)
FILENAME_CHILD_RE = re.compile(
    r"(INDIVIDUAL|GROUP)[_\[\(]+([\d]+(?:-[\d]+)*)[_\]\)]+", re.IGNORECASE
)


def parse_filename(path: Path) -> dict:
    """Extracts session_number, session_type, child_id from the filename.

    Tolerant of special characters (# [ ] .) being replaced with "_"
    (typical when files are uploaded via web interfaces), as well as
    the "clean" original format with # [ ].
    """
    name = path.stem

    session_number = None
    session_type = None
    child_id = None

    m = FILENAME_SESSION_RE.search(name)
    if m:
        session_number = m.group(1)
        session_type = m.group(2).capitalize()

    m2 = FILENAME_CHILD_RE.search(name)
    if m2:
        child_id = m2.group(2)

    if session_number is None or child_id is None:
        # Fallback: warn but don't crash — process the file with
        # child_id/session="UNKNOWN" so the user sees the issue in the
        # output instead of silently losing data.
        session_number = session_number or "UNKNOWN"
        child_id = child_id or "UNKNOWN"
        session_type = session_type or "Unknown"

    session_id = f"{child_id}_{session_number}"
    return {
        "session_id": session_id,
        "child_id": child_id,
        "session_number": session_number,
        "session_type": session_type,
        "source_file": str(path),
    }


# ---------------------------------------------------------------------------
# 2. Tier name parsing -> track_id / track_type / tier_code
# ---------------------------------------------------------------------------

CHILD_TIER_RE = re.compile(r"^c(\d+(?:-\d+)*)_(.+)$", re.IGNORECASE)
THERAPIST_TIER_RE = re.compile(r"^t(\d+)_(.+)$", re.IGNORECASE)


def parse_tier(tier: str, session_child_id: str) -> dict:
    """Classifies an ELAN tier into track_type/track_id/tier_code."""
    if tier.upper() == "ST":
        return {"track_type": "session", "track_id": "session", "tier_code": "ST"}

    m = CHILD_TIER_RE.match(tier)
    if m:
        cid, code = m.group(1), m.group(2)
        return {"track_type": "child", "track_id": f"child_{cid}", "tier_code": code}

    m = THERAPIST_TIER_RE.match(tier)
    if m:
        tnum, code = m.group(1), m.group(2)
        return {"track_type": "therapist", "track_id": f"therapist_{tnum}", "tier_code": code}

    # Unknown prefix — keep it without losing information.
    return {"track_type": "other", "track_id": tier, "tier_code": tier}


# ---------------------------------------------------------------------------
# 3. Parsing the annotation file itself (tab-delimited, 9 columns, no header)
# ---------------------------------------------------------------------------

def sec_to_ms(value: str) -> int:
    return int(round(float(value) * 1000))


def parse_annotation_file(path: Path) -> list[dict]:
    """Parses one tab-delimited annotation file -> list of raw segments."""
    segments = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 9:
                print(
                    f"  [!] {path.name}: line {line_no} has {len(cols)} columns "
                    f"(expected 9) — skipped",
                    file=sys.stderr,
                )
                continue
            tier, _blank, start_hms, start_sec, end_hms, end_sec, dur_hms, dur_sec, value = cols[:9]
            try:
                start_ms = sec_to_ms(start_sec)
                end_ms = sec_to_ms(end_sec)
                dur_ms = sec_to_ms(dur_sec)
            except ValueError:
                print(f"  [!] {path.name}: line {line_no} — could not parse timing", file=sys.stderr)
                continue
            segments.append(
                {
                    "tier": tier,
                    "start_hms": start_hms,
                    "end_hms": end_hms,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": dur_ms,
                    "value": value,
                }
            )
    return segments


# ---------------------------------------------------------------------------
# 4. (Optional) Phase 0 Sessions table -> time_offset_ms by session_id
# ---------------------------------------------------------------------------

def load_time_offsets(sessions_csv: Optional[Path]) -> dict:
    if not sessions_csv:
        return {}
    if not sessions_csv.exists():
        print(
            f"[!] --sessions-csv path not found: {sessions_csv}\n"
            f"    (resolved to: {sessions_csv.resolve()})\n"
            f"    Check the path, or drop --sessions-csv to run without audio alignment.",
            file=sys.stderr,
        )
        sys.exit(1)
    offsets = {}
    with sessions_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("session_id")
            off = row.get("time_offset_ms")
            if sid and off not in (None, "", "nan"):
                try:
                    offsets[sid] = int(float(off))
                except ValueError:
                    pass
    return offsets


# ---------------------------------------------------------------------------
# 5. (Optional) tier_code -> human-readable label map (extendable by the team)
# ---------------------------------------------------------------------------

# Tier-level labels (what the tier itself represents).
DEFAULT_TIER_MAP = {
    "ST": "Session time (boundary)",
    "CP": "Child position",
    "CAO": "Child attending to objects",
    "CAT": "Child attention to therapist",
    "CG": "Child gaze",
    "CSP": "Child session pattern",
    "CTCA": "Common action during shared engagement (child-tier, within CSP:TC/PL/PRC)",
    "TSP": "Therapist session pattern",
    "JA": "Joint eye contact",
    "TP": "Therapist position",
    "CV": "Child vocalization",
    "TV": "Therapist vocalization",
    # "CI" appears in at least one real annotation file but is not documented
    # in the official ELAN schema table we have — meaning unconfirmed.
    "CI": "UNCONFIRMED — appears in real data, not in the documented schema",
}

# Per-VALUE operational definitions, keyed by (tier_code, value). Looked up
# in addition to the tier-level label above when both are available.
# Confirmed against the official "ELAN annotation schema for individual
# sessions" table (CHUV / Bei-Xuan Lin).
TIER_VALUE_MAP = {
    ("ST", "ST"): "Defines session start/end range for analysis",
    ("CP", "CST"): "Standing; two feet on ground",
    ("CP", "CHO"): "Hovering/leaning over table; one foot on chair/one on ground",
    ("CP", "CSI"): "Sitting",
    ("CP", "CGO"): "Child gone (left room / in box / not visible)",
    ("CP", "CLF"): "On floor (lying down)",
    ("CP", "CRE"): "Reaching: partial rise from seated/half-standing to grasp objects",
    ("CP", "CCR"): "Crouch: knees bent, upper body forward",
    ("CAO", "AO"): "Attending to session-related objects (e.g. clay, Legos, paintings)",
    ("CAO", "ANO"): "Attending to non-session-related objects (e.g. sink, shoes, jacket)",
    ("CAO", "AU"): "Attending to undetermined objects",
    ("CAT", "AT"): "Attending to therapist",
    ("CG", "GO"): "Gaze at session-related objects",
    ("CG", "GT"): "Gaze at therapist",
    ("CG", "GNO"): "Gaze at non-session-related objects",
    ("CG", "GU"): "Gaze undetermined",
    ("CSP", "CP"): "Child creating/playing on their own",
    ("CSP", "COB"): "Child observing therapist's creative actions without participation",
    ("CSP", "TC"): "Child and therapist building/creating/discussing together",
    ("CSP", "PL"): "Child and therapist playing together",
    ("CSP", "PRC"): "Preparing/cleaning up by either child or therapist",
    ("CTCA", "OBJ_EXCHANGE"): "Transfer of materials between child and therapist",
    ("CTCA", "ARTMAKING"): "Object-based creation or sensory exploration (drawing, sculpting, baking, arranging materials...)",
    ("CTCA", "SYMBOLIC_PLAY"): "Pretending/symbolic use of objects representing roles/events (role-play, enacting stories...)",
    ("CTCA", "COORDINATED_PLAY"): "Rule-governed or rhythmic physical interaction requiring mutual timing/turn structure (drumming, ball catch, clapping games...)",
    ("CTCA", "CONVERSATION"): "Reciprocal verbal interaction when not dominated by active play or artmaking",
    ("CTCA", "CLEAN_UP_ACTIVITY"): "Organizing/tidying/closing an activity",
    ("TSP", "T"): "Therapist creating on their own",
    ("TSP", "TOB"): "Therapist observing/supervising child without direct material engagement or verbal collaboration",
    ("TSP", "TC"): "Child and therapist building/creating/discussing together",
    ("TSP", "PL"): "Child and therapist playing together",
    ("TSP", "PRC"): "Preparing/cleaning up by either child or therapist",
    ("JA", "TC"): "Joint eye contact present between therapist and child",
    ("TP", "TST"): "Standing; two feet on ground",
    ("TP", "THO"): "Hovering/leaning over table; one foot on chair/one on ground",
    ("TP", "TSI"): "Sitting",
    ("TP", "TGO"): "Therapist gone (left room / not visible)",
    ("TP", "TLF"): "On floor (lying down)",
    ("TP", "TRE"): "Reaching: partial rise from seated/half-standing to grasp objects",
    ("TP", "TCR"): "Crouch: knees bent, upper body forward",
    ("CV", "CS"): "Child speaking",
    ("CV", "CNS"): "Child making non-speech sounds",
    ("TV", "TS"): "Therapist speaking",
    ("TV", "TNS"): "Therapist making non-speech sounds",
}


def load_tier_map(path: Optional[Path]) -> dict:
    tier_map = dict(DEFAULT_TIER_MAP)
    if path and path.exists():
        with path.open(encoding="utf-8") as f:
            tier_map.update(json.load(f))
    return tier_map


# ---------------------------------------------------------------------------
# 6. Main conversion logic
# ---------------------------------------------------------------------------

def convert_file(path: Path, tier_map: dict, offsets: dict) -> dict:
    meta = parse_filename(path)
    raw_segments = parse_annotation_file(path)

    annotations = []
    tracks_seen = set()
    offset_ms = offsets.get(meta["session_id"])

    for seg in raw_segments:
        tinfo = parse_tier(seg["tier"], meta["child_id"])
        tracks_seen.add(tinfo["track_id"])
        entry = {
            "session_id": meta["session_id"],
            "child_id": meta["child_id"],
            "track_type": tinfo["track_type"],
            "track_id": tinfo["track_id"],
            "tier_code": tinfo["tier_code"],
            "tier_label": tier_map.get(tinfo["tier_code"], ""),
            "value": seg["value"],
            "value_label": TIER_VALUE_MAP.get((tinfo["tier_code"], seg["value"]), ""),
            "start_ms_video": seg["start_ms"],
            "end_ms_video": seg["end_ms"],
            "duration_ms": seg["duration_ms"],
        }
        if offset_ms is not None:
            entry["start_ms_audio"] = seg["start_ms"] - offset_ms
            entry["end_ms_audio"] = seg["end_ms"] - offset_ms
            entry["audio_offset_ms"] = offset_ms
        else:
            entry["start_ms_audio"] = None
            entry["end_ms_audio"] = None
            entry["audio_offset_ms"] = None
        annotations.append(entry)

    return {
        "schema_version": "phase0-v1",
        "session_id": meta["session_id"],
        "child_id": meta["child_id"],
        "session_number": meta["session_number"],
        "session_type": meta["session_type"],
        "source_file": meta["source_file"],
        "tracks": sorted(tracks_seen),
        "n_annotations": len(annotations),
        "audio_offset_ms": offset_ms,
        "annotations": annotations,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder with annotation files (.txt/.tsv)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Where to write json/ and annotations_master.csv")
    ap.add_argument("--sessions-csv", type=Path, default=None, help="CSV export of the Sessions sheet (for time_offset_ms)")
    ap.add_argument("--tier-map", type=Path, default=None, help="JSON with additional tier_code -> label mappings")
    ap.add_argument("--pattern", default="*.txt", help="Glob pattern for input files (default *.txt)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.out_dir / "json"
    json_dir.mkdir(exist_ok=True)

    offsets = load_time_offsets(args.sessions_csv)
    tier_map = load_tier_map(args.tier_map)

    files = sorted(args.input_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} found in {args.input_dir}.", file=sys.stderr)
        sys.exit(1)

    master_rows = []
    for path in files:
        print(f"Processing: {path.name}")
        result = convert_file(path, tier_map, offsets)
        out_json = json_dir / f"{result['session_id']}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        master_rows.extend(result["annotations"])
        print(
            f"  -> session_id={result['session_id']} | "
            f"{result['n_annotations']} annotations | tracks: {', '.join(result['tracks'])} | "
            f"-> {out_json}"
        )

    master_csv = args.out_dir / "annotations_master.csv"
    if master_rows:
        fieldnames = list(master_rows[0].keys())
        with master_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_rows)
        print(f"\nConsolidated CSV: {master_csv} ({len(master_rows)} rows)")

    print(f"Done. Files processed: {len(files)}.")


if __name__ == "__main__":
    main()
