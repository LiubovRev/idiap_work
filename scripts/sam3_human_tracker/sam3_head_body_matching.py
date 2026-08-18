# coding=utf-8

import argparse
import os

import numpy as np
import pandas as pd


HEAD_COLS = [
    "head_bbox_xmin",
    "head_bbox_ymin",
    "head_bbox_xmax",
    "head_bbox_ymax",
    "head_confidence",
]

BODY_COLS = [
    "body_bbox_xmin",
    "body_bbox_ymin",
    "body_bbox_xmax",
    "body_bbox_ymax",
    "body_confidence",
]


# Maximum normalized distance allowed when the head center
# is outside the body bbox.
MAX_OUTSIDE_DISTANCE = 0.2


def head_center(head):
    """Return the center (x, y) of a head bbox."""
    x = (
        head["head_bbox_xmin"]
        + head["head_bbox_xmax"]
    ) / 2.0

    y = (
        head["head_bbox_ymin"]
        + head["head_bbox_ymax"]
    ) / 2.0

    return x, y


def body_center(body):
    """Return the center (x, y) of a body bbox."""
    x = (
        body["body_bbox_xmin"]
        + body["body_bbox_xmax"]
    ) / 2.0

    y = (
        body["body_bbox_ymin"]
        + body["body_bbox_ymax"]
    ) / 2.0

    return x, y


def head_inside_body(head, body):
    """Check whether the head center is inside the body bbox."""
    x, y = head_center(head)

    return (
        body["body_bbox_xmin"] <= x <= body["body_bbox_xmax"]
        and
        body["body_bbox_ymin"] <= y <= body["body_bbox_ymax"]
    )


def normalized_distance(head, body):
    """
    Distance between head and body centers normalized
    by body width and height.
    """
    hx, hy = head_center(head)
    bx, by = body_center(body)

    body_w = max(
        body["body_bbox_xmax"] - body["body_bbox_xmin"],
        1.0,
    )

    body_h = max(
        body["body_bbox_ymax"] - body["body_bbox_ymin"],
        1.0,
    )

    dx = (hx - bx) / body_w
    dy = (hy - by) / body_h

    return np.sqrt(dx * dx + dy * dy)


def match_frame(body_frame, head_frame):
    """
    Match HumanTracker heads to SAM3 bodies.

    Matching strategy:

    1. Prefer associations where the head center is inside
       the body bbox.
    2. If multiple bodies contain the same head, choose
       the closest body.
    3. Each head and body can be matched only once.
    4. Remaining heads can be matched to remaining bodies
       only if they are outside but sufficiently close.
    5. SAM3 PID is preserved.

    Returns
    -------
    matches : dict
        body_index -> head_index

    match_info : dict
        body_index -> {
            "match_type": "inside" or "outside",
            "match_distance": float,
        }
    """

    matches = {}
    match_info = {}

    used_heads = set()
    used_bodies = set()

    # ==============================================================
    # STEP 1: INSIDE-BODY MATCHING
    # ==============================================================

    inside_candidates = []

    for head_idx, head in head_frame.iterrows():

        for body_idx, body in body_frame.iterrows():

            if not head_inside_body(head, body):
                continue

            distance = normalized_distance(
                head,
                body,
            )

            inside_candidates.append(
                (
                    distance,
                    head_idx,
                    body_idx,
                )
            )

    # Closest associations first.
    inside_candidates.sort(
        key=lambda x: x[0]
    )

    for distance, head_idx, body_idx in inside_candidates:

        if head_idx in used_heads:
            continue

        if body_idx in used_bodies:
            continue

        matches[body_idx] = head_idx

        match_info[body_idx] = {
            "match_type": "inside",
            "match_distance": distance,
        }

        used_heads.add(head_idx)
        used_bodies.add(body_idx)

    # ==============================================================
    # STEP 2: OUTSIDE FALLBACK
    # ==============================================================

    outside_candidates = []

    for head_idx, head in head_frame.iterrows():

        if head_idx in used_heads:
            continue

        for body_idx, body in body_frame.iterrows():

            if body_idx in used_bodies:
                continue

            if head_inside_body(head, body):
                continue

            distance = normalized_distance(
                head,
                body,
            )

            if distance <= MAX_OUTSIDE_DISTANCE:

                outside_candidates.append(
                    (
                        distance,
                        head_idx,
                        body_idx,
                    )
                )

    outside_candidates.sort(
        key=lambda x: x[0]
    )

    for distance, head_idx, body_idx in outside_candidates:

        if head_idx in used_heads:
            continue

        if body_idx in used_bodies:
            continue

        matches[body_idx] = head_idx

        match_info[body_idx] = {
            "match_type": "outside",
            "match_distance": distance,
        }

        used_heads.add(head_idx)
        used_bodies.add(body_idx)

    return matches, match_info


def main(args):

    print("SAM3 body CSV:", args.body_csv)
    print("Head detection CSV:", args.head_csv)
    print("Output CSV:", args.output)

    df_body = pd.read_csv(args.body_csv)
    df_head = pd.read_csv(args.head_csv)

    print()
    print("Body rows:", len(df_body))
    print("Body frames:", df_body["frame_index"].nunique())
    print("SAM3 PIDs:", sorted(df_body["pid"].unique()))

    print()
    print("Head rows:", len(df_head))
    print("Head frames:", df_head["frame_index"].nunique())

    metadata_cols = [
        "video_file",
        "video_name",
        "num_frames",
        "frame_height",
        "frame_width",
        "fps",
    ]

    output_rows = []

    inside_matches = 0
    outside_matches = 0

    for frame_index in sorted(
        df_body["frame_index"].unique()
    ):

        body_frame = (
            df_body[
                df_body["frame_index"] == frame_index
            ].copy()
        )

        head_frame = (
            df_head[
                df_head["frame_index"] == frame_index
            ].copy()
        )

        matches, match_info = match_frame(
            body_frame,
            head_frame,
        )

        for body_idx, body in body_frame.iterrows():

            row = {
                "frame_index": int(frame_index),
                "pid": int(body["pid"]),
            }

            # ------------------------------------------------------
            # Body information
            # ------------------------------------------------------

            for col in BODY_COLS:

                if col in body:

                    row[col] = body[col]

                else:

                    row[col] = np.nan

            # ------------------------------------------------------
            # Default: no head
            # ------------------------------------------------------

            for col in HEAD_COLS:
                row[col] = np.nan

            row["match_type"] = "none"
            row["match_distance"] = np.nan

            # ------------------------------------------------------
            # Matched head
            # ------------------------------------------------------

            if body_idx in matches:

                head_idx = matches[body_idx]

                head = head_frame.loc[head_idx]

                for col in HEAD_COLS:
                    row[col] = head[col]

                info = match_info[body_idx]

                row["match_type"] = info["match_type"]
                row["match_distance"] = info[
                    "match_distance"
                ]

                if info["match_type"] == "inside":
                    inside_matches += 1

                elif info["match_type"] == "outside":
                    outside_matches += 1

            # ------------------------------------------------------
            # Metadata
            # ------------------------------------------------------

            for col in metadata_cols:

                if col in body:
                    row[col] = body[col]

            output_rows.append(row)

    df_output = pd.DataFrame(output_rows)

    columns = [
        "frame_index",
        "pid",

        "head_bbox_xmin",
        "head_bbox_ymin",
        "head_bbox_xmax",
        "head_bbox_ymax",
        "head_confidence",

        "body_bbox_xmin",
        "body_bbox_ymin",
        "body_bbox_xmax",
        "body_bbox_ymax",
        "body_confidence",

        "match_type",
        "match_distance",

        "video_file",
        "video_name",
        "num_frames",
        "frame_height",
        "frame_width",
        "fps",
    ]

    df_output = df_output[columns]

    os.makedirs(
        os.path.dirname(
            os.path.abspath(args.output)
        ),
        exist_ok=True,
    )

    df_output.to_csv(
        args.output,
        index=False,
        float_format="%.2f",
    )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    matched_heads = (
        df_output["head_confidence"].notna().sum()
    )

    total_heads = len(df_head)

    unmatched_heads = (
        total_heads - matched_heads
    )

    print()
    print("========================================")
    print("MATCHING STATISTICS")
    print("========================================")

    print(
        f"Matched heads: "
        f"{matched_heads}/{total_heads} "
        f"({100.0 * matched_heads / max(total_heads, 1):.1f}%)"
    )

    print(
        f"Unmatched heads: {unmatched_heads}"
    )

    print(
        f"Inside-body matches: {inside_matches}"
    )

    print(
        f"Outside fallback matches: {outside_matches}"
    )

    print()
    print("Output rows:", len(df_output))
    print(
        "Output frames:",
        df_output["frame_index"].nunique(),
    )

    print(
        "Output PIDs:",
        sorted(df_output["pid"].unique()),
    )

    print()
    print("Match types:")

    print(
        df_output["match_type"].value_counts(
            dropna=False
        )
    )

    print()
    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Associate HumanTracker head detections "
            "with SAM3 body detections while preserving "
            "SAM3 PIDs."
        )
    )

    parser.add_argument(
        "--body_csv",
        required=True,
        help="SAM3 body detection CSV",
    )

    parser.add_argument(
        "--head_csv",
        required=True,
        help="HumanTracker head detection CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output combined CSV",
    )

    args = parser.parse_args()

    main(args)
