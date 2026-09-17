#!/usr/bin/env python3
"""
Simple OpenCV editor for person tracking CSV files.

Expected CSV columns:
    frame_index, pid,
    body_bbox_xmin, body_bbox_ymin,
    body_bbox_xmax, body_bbox_ymax,
    body_confidence

Mouse:
    Left click   -> select primary track A
    Right click  -> select secondary track B

Keys:
    Space        -> play/pause
    p            -> replay from the first processed frame
    a / d        -> previous / next frame
    j / l        -> -10 / +10 frames
    m            -> merge B into A for the whole video
    f            -> merge B into A from the current frame onward
    s            -> swap A and B from the current frame onward
    r            -> remove A for the whole video
    e            -> remove A from the current frame onward
    u            -> undo last edit
    w            -> save
    c            -> clear selections
    h            -> show/hide help
    q / Esc      -> quit

Notes:
- "merge" is blocked if A and B coexist in any affected frame, because that
  would create duplicate detections with the same pid. Use "swap" for an ID
  switch involving two simultaneously visible people.
- Edits are kept in memory until saved. Pressing w saves both the edited CSV and a structured change-log CSV.
- Playback/navigation is limited to the min..max frame_index interval found in
  the CSV (after applying --frame-offset), not the complete source video.
- A bottom timeline shows every active PID with the same color as its bbox and updates after edits.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "frame_index",
    "pid",
    "body_bbox_xmin",
    "body_bbox_ymin",
    "body_bbox_xmax",
    "body_bbox_ymax",
    "body_confidence",
]


# Predefined track colors in OpenCV BGR order.
# PIDs are assigned colors once at startup and keep the same color throughout
# the video. If there are more PIDs than colors, the palette cycles.
TRACK_COLORS_BGR = [
    (230, 25, 75),    # red-ish
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # orange
    (245, 130, 48),   # blue-ish in BGR
    (145, 30, 180),   # purple
    (70, 240, 240),   # yellow/cyan-ish
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 190),  # light cyan
    (0, 128, 128),    # olive
    (230, 190, 255),  # pink
    (170, 110, 40),   # brown/blue-ish
    (255, 250, 200),  # pale cyan
    (128, 0, 0),      # dark blue
    (170, 255, 195),  # pale green
    (128, 128, 0),    # teal
    (255, 215, 180),  # peach/light blue
    (0, 0, 128),      # dark red
    (128, 128, 128),  # gray
    (255, 105, 180),  # pink-ish
    (64, 224, 208),   # turquoise
    (0, 215, 255),    # gold
    (180, 105, 255),  # salmon-ish
    (255, 144, 30),   # dodger-blue-ish
    (50, 205, 50),    # lime green
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (0, 165, 255),    # orange
    (147, 20, 255),   # deep pink-ish
    (238, 130, 238),  # violet-ish
]


class TrackEditor:
    WINDOW = "Track editor"

    def __init__(
        self,
        video_path,
        csv_path,
        output_path,
        log_path=None,
        frame_offset=0,
        max_width=1400,
        max_height=900,
        undo_limit=100,
    ):
        self.video_path = Path(video_path)
        self.csv_path = Path(csv_path)
        self.output_path = Path(output_path)
        self.log_path = (
            Path(log_path)
            if log_path is not None
            else self.output_path.with_name(self.output_path.stem + "_changes.csv")
        )
        self.frame_offset = int(frame_offset)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.undo_limit = int(undo_limit)

        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")
        if self.df.empty:
            raise ValueError("The tracking CSV is empty")

        # The CSV does not explicitly store frames with zero detections. The
        # processed interval is therefore inferred as the inclusive range from
        # the smallest to the largest frame_index present in the CSV.
        self.csv_frame_min = int(self.df["frame_index"].min())
        self.csv_frame_max = int(self.df["frame_index"].max())

        # Stable pid -> color assignment. Sorting by string representation also
        # works when pid values are strings rather than plain integers.
        unique_pids = list(pd.unique(self.df["pid"]))
        unique_pids.sort(key=lambda value: str(value))
        self.pid_colors = {
            pid: TRACK_COLORS_BGR[i % len(TRACK_COLORS_BGR)]
            for i, pid in enumerate(unique_pids)
        }
        self.palette_reused = len(unique_pids) > len(TRACK_COLORS_BGR)

        # Internal soft-delete flag makes undo inexpensive.
        self.original_columns = list(self.df.columns)
        self.df["_deleted"] = False

        # frame_index never changes, so cache row indices per frame.
        self.frame_to_rows = {
            int(frame): np.asarray(rows, dtype=np.int64)
            for frame, rows in self.df.groupby("frame_index", sort=False).groups.items()
        }

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.video_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

        # CSV frame_index = OpenCV video frame + frame_offset. Restrict the
        # editor to the interval represented by the tracking CSV instead of
        # walking through the complete source video.
        requested_start = self.csv_frame_min - self.frame_offset
        requested_end = self.csv_frame_max - self.frame_offset
        self.processed_video_start = max(0, requested_start)
        if self.video_frame_count > 0:
            self.processed_video_end = min(self.video_frame_count - 1, requested_end)
        else:
            self.processed_video_end = requested_end

        if self.processed_video_end < self.processed_video_start:
            raise ValueError(
                "The CSV frame range does not overlap the video after applying "
                f"--frame-offset={self.frame_offset}. CSV range: "
                f"{self.csv_frame_min}..{self.csv_frame_max}."
            )

        self.current_video_frame = self.processed_video_start
        self.frame_bgr = None
        self.display_scale = 1.0

        self.primary_pid = None
        self.secondary_pid = None
        self.paused = True
        self.show_help = False
        self.status = "Left-click A, right-click B. Press h for help."
        self.timeline_cache = None
        self.timeline_dirty = True
        self.video_display_height = 0
        self.timeline_plot_left = 0
        self.timeline_plot_right = 0
        self.history = []
        self.change_log = []
        self.next_change_id = 1
        self.dirty = False

        self._goto_video_frame(self.processed_video_start)

    @property
    def current_csv_frame(self):
        return self.current_video_frame + self.frame_offset

    def _set_status(self, text):
        self.status = str(text)

    def _rows_at_current_frame(self):
        rows = self.frame_to_rows.get(self.current_csv_frame)
        if rows is None or len(rows) == 0:
            return self.df.iloc[0:0]
        cur = self.df.loc[rows]
        return cur.loc[~cur["_deleted"]]

    def _goto_video_frame(self, frame_idx):
        frame_idx = max(
            self.processed_video_start,
            min(int(frame_idx), self.processed_video_end),
        )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok:
            self._set_status(f"Could not read video frame {frame_idx}")
            return False

        self.current_video_frame = frame_idx
        self.frame_bgr = frame
        return True

    def _read_next_frame(self):
        if self.current_video_frame >= self.processed_video_end:
            self.paused = True
            self._set_status("End of processed CSV frame range")
            return False

        ok, frame = self.cap.read()
        if not ok:
            self.paused = True
            self._set_status(
                f"Could not read video frame {self.current_video_frame + 1}"
            )
            return False
        self.current_video_frame += 1
        self.frame_bgr = frame
        return True

    def _record_change(
        self,
        action,
        *,
        pid_a=None,
        pid_b=None,
        scope="",
        affected_rows=0,
        description="",
        undone_change_id=None,
    ):
        change_id = self.next_change_id
        self.next_change_id += 1
        self.change_log.append(
            {
                "change_id": change_id,
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "action": action,
                "video_frame": self.current_video_frame,
                "csv_frame": self.current_csv_frame,
                "pid_a": pid_a,
                "pid_b": pid_b,
                "scope": scope,
                "affected_rows": int(affected_rows),
                "undone_change_id": undone_change_id,
                "description": description,
            }
        )
        return change_id

    def _push_undo(self, indices, description, change_id):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return
        before = self.df.loc[indices, ["pid", "_deleted"]].copy(deep=True)
        self.history.append((before, description, change_id))
        if len(self.history) > self.undo_limit:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            self._set_status("Nothing to undo")
            return

        before, description, change_id = self.history.pop()
        self.df.loc[before.index, "pid"] = before["pid"]
        self.df.loc[before.index, "_deleted"] = before["_deleted"]
        self._record_change(
            "undo",
            affected_rows=len(before),
            description=f"undo: {description}",
            undone_change_id=change_id,
        )
        self.dirty = True
        self._invalidate_timeline()
        self._set_status(f"Undid: {description}")

    def _active_mask(self):
        return ~self.df["_deleted"]

    def _active_pids(self):
        """Return all PIDs currently present anywhere in the edited dataframe."""
        pids = list(pd.unique(self.df.loc[self._active_mask(), "pid"]))
        pids.sort(key=lambda value: str(value))
        return pids

    @staticmethod
    def _format_pid(value):
        # Avoid displaying integer-valued float IDs as e.g. 3.0 when possible.
        try:
            f = float(value)
            if np.isfinite(f) and f.is_integer():
                return str(int(f))
        except (TypeError, ValueError):
            pass
        return str(value)

    def _invalidate_timeline(self):
        self.timeline_dirty = True

    def _timeline_data(self):
        """Return [(pid, [(start_frame, end_frame), ...]), ...].

        The result is cached because it only changes after an edit/undo, not
        while simply playing the video.
        """
        if self.timeline_cache is not None and not self.timeline_dirty:
            return self.timeline_cache

        start = self.processed_video_start + self.frame_offset
        end = self.processed_video_end + self.frame_offset
        active = self.df.loc[
            self._active_mask(), ["frame_index", "pid"]
        ]
        active = active[
            (active["frame_index"] >= start) & (active["frame_index"] <= end)
        ]

        result = []
        for pid, group in active.groupby("pid", sort=False):
            frames = np.sort(pd.unique(group["frame_index"].astype(int)))
            if len(frames) == 0:
                continue

            # Convert detections into contiguous visible segments. Missing
            # detections therefore appear as gaps in the timeline.
            split_at = np.where(np.diff(frames) > 1)[0] + 1
            chunks = np.split(frames, split_at)
            segments = [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if len(chunk)]
            result.append((pid, segments))

        result.sort(key=lambda item: str(item[0]))
        self.timeline_cache = result
        self.timeline_dirty = False
        return result

    def merge(self, from_current=False):
        a = self.primary_pid
        b = self.secondary_pid
        if a is None or b is None:
            self._set_status("Merge needs A and B selections")
            return
        if a == b:
            self._set_status("A and B are already the same pid")
            return

        active = self._active_mask()
        scope = active.copy()
        if from_current:
            scope &= self.df["frame_index"] >= self.current_csv_frame

        source_mask = scope & (self.df["pid"] == b)
        target_mask = scope & (self.df["pid"] == a)

        source_indices = self.df.index[source_mask].to_numpy()
        if len(source_indices) == 0:
            self._set_status(f"No active rows for pid {b} in the selected scope")
            return

        source_frames = set(self.df.loc[source_mask, "frame_index"].tolist())
        target_frames = set(self.df.loc[target_mask, "frame_index"].tolist())
        overlap = sorted(source_frames & target_frames)

        if overlap:
            preview = ", ".join(map(str, overlap[:5]))
            suffix = "..." if len(overlap) > 5 else ""
            self._set_status(
                f"Merge blocked: A and B coexist in {len(overlap)} frame(s): "
                f"{preview}{suffix}. Use swap for an ID switch."
            )
            return

        scope_text = f"from frame {self.current_csv_frame}" if from_current else "globally"
        description = f"merge pid {b} -> {a} {scope_text}"
        change_id = self._record_change(
            "merge",
            pid_a=a,
            pid_b=b,
            scope="from_current" if from_current else "global",
            affected_rows=len(source_indices),
            description=description,
        )
        self._push_undo(source_indices, description, change_id)
        self.df.loc[source_indices, "pid"] = a

        self.secondary_pid = None
        self.dirty = True
        self._invalidate_timeline()
        self._set_status(f"Merged pid {b} -> {a} {scope_text} ({len(source_indices)} rows)")

    def swap_from_current(self):
        a = self.primary_pid
        b = self.secondary_pid
        if a is None or b is None:
            self._set_status("Swap needs A and B selections")
            return
        if a == b:
            self._set_status("A and B are the same pid")
            return

        scope = (
            self._active_mask()
            & (self.df["frame_index"] >= self.current_csv_frame)
            & self.df["pid"].isin([a, b])
        )
        indices = self.df.index[scope].to_numpy()
        if len(indices) == 0:
            self._set_status("No rows to swap in this scope")
            return

        # Compute masks before modifying values.
        a_idx = self.df.index[scope & (self.df["pid"] == a)].to_numpy()
        b_idx = self.df.index[scope & (self.df["pid"] == b)].to_numpy()

        description = f"swap pid {a} <-> {b} from frame {self.current_csv_frame}"
        change_id = self._record_change(
            "swap",
            pid_a=a,
            pid_b=b,
            scope="from_current",
            affected_rows=len(indices),
            description=description,
        )
        self._push_undo(indices, description, change_id)

        self.df.loc[a_idx, "pid"] = b
        self.df.loc[b_idx, "pid"] = a

        self.dirty = True
        self._invalidate_timeline()
        self._set_status(
            f"Swapped pid {a} <-> {b} from frame {self.current_csv_frame} "
            f"({len(indices)} rows)"
        )

    def remove_primary(self, from_current=False):
        a = self.primary_pid
        if a is None:
            self._set_status("Remove needs a primary A selection")
            return

        mask = self._active_mask() & (self.df["pid"] == a)
        if from_current:
            mask &= self.df["frame_index"] >= self.current_csv_frame

        indices = self.df.index[mask].to_numpy()
        if len(indices) == 0:
            self._set_status(f"No rows to remove for pid {a}")
            return

        scope_text = f"from frame {self.current_csv_frame}" if from_current else "globally"
        description = f"remove pid {a} {scope_text}"
        change_id = self._record_change(
            "remove",
            pid_a=a,
            scope="from_current" if from_current else "global",
            affected_rows=len(indices),
            description=description,
        )
        self._push_undo(indices, description, change_id)
        self.df.loc[indices, "_deleted"] = True

        self.dirty = True
        self._invalidate_timeline()
        self._set_status(f"Removed pid {a} {scope_text} ({len(indices)} rows)")

        if not from_current:
            self.primary_pid = None

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        out = self.df.loc[~self.df["_deleted"], self.original_columns]

        tmp = self.output_path.with_name(self.output_path.name + ".tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, self.output_path)

        log_columns = [
            "change_id",
            "timestamp",
            "action",
            "video_frame",
            "csv_frame",
            "pid_a",
            "pid_b",
            "scope",
            "affected_rows",
            "undone_change_id",
            "description",
        ]
        log_df = pd.DataFrame(self.change_log, columns=log_columns)
        log_tmp = self.log_path.with_name(self.log_path.name + ".tmp")
        log_df.to_csv(log_tmp, index=False)
        os.replace(log_tmp, self.log_path)

        self.dirty = False
        self._set_status(
            f"Saved {len(out)} rows + {len(log_df)} log entries to "
            f"{self.output_path} / {self.log_path}"
        )

    def _pick_detection(self, x_video, y_video):
        cur = self._rows_at_current_frame()
        if cur.empty:
            return None

        hits = cur[
            (cur["body_bbox_xmin"] <= x_video)
            & (x_video <= cur["body_bbox_xmax"])
            & (cur["body_bbox_ymin"] <= y_video)
            & (y_video <= cur["body_bbox_ymax"])
        ].copy()

        if hits.empty:
            return None

        # If boxes overlap, select the smallest one under the cursor.
        hits["_area"] = (
            (hits["body_bbox_xmax"] - hits["body_bbox_xmin"]).clip(lower=0)
            * (hits["body_bbox_ymax"] - hits["body_bbox_ymin"]).clip(lower=0)
        )
        return hits.sort_values("_area").iloc[0]["pid"]

    def _mouse_callback(self, event, x, y, flags, userdata):
        if self.frame_bgr is None:
            return

        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return

        self.paused = True

        # Click inside the bottom timeline to seek. This keeps timeline clicks
        # from accidentally clearing the A/B selection.
        if y >= self.video_display_height:
            if self.timeline_plot_right > self.timeline_plot_left:
                x_clamped = max(
                    self.timeline_plot_left,
                    min(x, self.timeline_plot_right),
                )
                fraction = (
                    (x_clamped - self.timeline_plot_left)
                    / (self.timeline_plot_right - self.timeline_plot_left)
                )
                frame = int(round(
                    self.processed_video_start
                    + fraction * (self.processed_video_end - self.processed_video_start)
                ))
                self._goto_video_frame(frame)
                self._set_status(f"Jumped to frame {self.current_csv_frame}")
            return

        x_video = x / self.display_scale
        y_video = y / self.display_scale
        pid = self._pick_detection(x_video, y_video)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.primary_pid = pid
            self._set_status(f"Primary A = {pid}" if pid is not None else "Primary A cleared")
        else:
            self.secondary_pid = pid
            self._set_status(
                f"Secondary B = {pid}" if pid is not None else "Secondary B cleared"
            )

    @staticmethod
    def _safe_confidence(value):
        try:
            v = float(value)
            return f"{v:.2f}" if np.isfinite(v) else "nan"
        except Exception:
            return str(value)

    def _render(self):
        frame = self.frame_bgr.copy()
        h, w = frame.shape[:2]

        timeline_data = self._timeline_data()
        n_tracks = len(timeline_data)

        # Compact timeline dimensions. Reserve space for the timeline inside
        # max_height so it does not make the OpenCV window grow off-screen.
        timeline_header_h = 24
        timeline_footer_h = 22
        timeline_pad = 8
        desired_row_h = 18
        minimum_video_h = min(260, self.max_height // 2)
        available_for_rows = max(
            6 * max(n_tracks, 1),
            self.max_height - minimum_video_h - timeline_header_h - timeline_footer_h - 2 * timeline_pad,
        )
        if n_tracks:
            row_h = min(desired_row_h, max(6, available_for_rows // n_tracks))
        else:
            row_h = desired_row_h
        timeline_h = timeline_header_h + timeline_footer_h + 2 * timeline_pad + row_h * max(n_tracks, 1)

        available_video_h = max(180, self.max_height - timeline_h)
        self.display_scale = min(
            1.0,
            self.max_width / max(w, 1),
            available_video_h / max(h, 1),
        )

        if self.display_scale != 1.0:
            disp = cv2.resize(
                frame,
                (int(round(w * self.display_scale)), int(round(h * self.display_scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            disp = frame

        self.video_display_height = disp.shape[0]
        s = self.display_scale
        cur = self._rows_at_current_frame()

        for _, row in cur.iterrows():
            pid = row["pid"]

            # Each pid keeps its predefined color. Selection is shown by a
            # thicker box, so selecting a track never hides its identity color.
            color = self.pid_colors.get(pid, (220, 220, 220))
            thickness = 2
            selection_tag = ""
            if pid == self.primary_pid:
                thickness = 5
                selection_tag = "A "
            elif pid == self.secondary_pid:
                thickness = 5
                selection_tag = "B "

            x1 = int(round(float(row["body_bbox_xmin"]) * s))
            y1 = int(round(float(row["body_bbox_ymin"]) * s))
            x2 = int(round(float(row["body_bbox_xmax"]) * s))
            y2 = int(round(float(row["body_bbox_ymax"]) * s))

            cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
            label = (
                f"{selection_tag}pid={self._format_pid(pid)} "
                f"conf={self._safe_confidence(row['body_confidence'])}"
            )
            label_y = max(18, y1 - 7)
            cv2.putText(
                disp,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Minimal top overlay: no processed-range lines and no complete PID
        # list (the timeline below already shows every active track).
        state = "PAUSED" if self.paused else "PLAYING"
        save_state = "UNSAVED" if self.dirty else "saved"
        info_lines = [
            f"frame {self.current_csv_frame} | {state} | tracks {n_tracks} | "
            f"A={self._format_pid(self.primary_pid) if self.primary_pid is not None else '-'} "
            f"B={self._format_pid(self.secondary_pid) if self.secondary_pid is not None else '-'} | {save_state}",
            self.status,
        ]
        if self.show_help:
            info_lines += [
                "mouse: left=A, right=B | timeline click=seek | space play/pause | p replay | a/d +/-1 | j/l +/-10",
                "m merge B->A global | f merge B->A from here | s swap A<->B from here",
                "r remove A global | e remove A from here | u undo | w save | c clear | h help | q quit",
            ]

        line_h = 22
        panel_h = 8 + line_h * len(info_lines)
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (disp.shape[1], panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.48, disp, 0.52, 0, disp)
        for i, text in enumerate(info_lines):
            cv2.putText(
                disp,
                text,
                (9, 20 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

        # Bottom per-track timeline. Each PID has its own row; its visible
        # detections are drawn as colored segments using the exact bbox color.
        tw = disp.shape[1]
        timeline = np.full((timeline_h, tw, 3), 24, dtype=np.uint8)
        label_w = min(95, max(62, tw // 9))
        plot_left = label_w
        plot_right = max(plot_left + 1, tw - 10)
        self.timeline_plot_left = plot_left
        self.timeline_plot_right = plot_right

        cv2.putText(
            timeline,
            f"Timeline - {n_tracks} track{'s' if n_tracks != 1 else ''}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        csv_start = self.processed_video_start + self.frame_offset
        csv_end = self.processed_video_end + self.frame_offset
        frame_span = max(1, csv_end - csv_start)

        def frame_to_x(frame_index):
            fraction = (float(frame_index) - csv_start) / frame_span
            fraction = min(1.0, max(0.0, fraction))
            return int(round(plot_left + fraction * (plot_right - plot_left)))

        y0 = timeline_header_h + timeline_pad
        if timeline_data:
            for i, (pid, segments) in enumerate(timeline_data):
                top = y0 + i * row_h
                bottom = top + row_h - 2
                center_y = top + max(5, row_h // 2 + 4)
                color = self.pid_colors.get(pid, (220, 220, 220))

                # Subtle row baseline helps distinguish gaps in detections.
                cv2.line(
                    timeline,
                    (plot_left, top + row_h // 2),
                    (plot_right, top + row_h // 2),
                    (65, 65, 65),
                    1,
                )

                tag = ""
                if pid == self.primary_pid:
                    tag = "A "
                elif pid == self.secondary_pid:
                    tag = "B "
                cv2.putText(
                    timeline,
                    f"{tag}{self._format_pid(pid)}",
                    (8, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 if row_h >= 14 else 0.34,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                seg_top = top + max(1, row_h // 5)
                seg_bottom = max(seg_top + 1, bottom - max(1, row_h // 5))
                for seg_start, seg_end in segments:
                    x1 = frame_to_x(seg_start)
                    x2 = frame_to_x(seg_end)
                    if x2 <= x1:
                        x2 = min(plot_right, x1 + 1)
                    cv2.rectangle(
                        timeline,
                        (x1, seg_top),
                        (x2, seg_bottom),
                        color,
                        -1,
                    )
        else:
            cv2.putText(
                timeline,
                "No active tracks",
                (8, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # Current-frame playhead spans all track rows.
        playhead_x = frame_to_x(self.current_csv_frame)
        rows_bottom = y0 + row_h * max(n_tracks, 1)
        cv2.line(
            timeline,
            (playhead_x, timeline_header_h),
            (playhead_x, min(timeline_h - timeline_footer_h, rows_bottom)),
            (255, 255, 255),
            1,
        )

        # Only start/end labels: useful context without covering video content.
        footer_y = timeline_h - 7
        cv2.putText(
            timeline,
            str(csv_start),
            (plot_left, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        end_label = str(csv_end)
        (end_w, _), _ = cv2.getTextSize(
            end_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        cv2.putText(
            timeline,
            end_label,
            (max(plot_left, plot_right - end_w), footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        return np.vstack([disp, timeline])

    def run(self):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, self._mouse_callback)

        try:
            while True:
                display = self._render()
                cv2.imshow(self.WINDOW, display)

                delay_ms = 30 if self.paused else max(1, int(round(1000.0 / self.fps)))
                key = cv2.waitKey(delay_ms) & 0xFF

                # Window close button.
                try:
                    if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

                manual_navigation = False

                if key in (27, ord("q")):  # Esc or q
                    break
                elif key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("p"):
                    manual_navigation = self._goto_video_frame(self.processed_video_start)
                    if manual_navigation:
                        self.paused = False
                        self._set_status("Replaying from first processed frame")
                elif key == ord("a"):
                    self.paused = True
                    manual_navigation = self._goto_video_frame(self.current_video_frame - 1)
                elif key == ord("d"):
                    self.paused = True
                    manual_navigation = self._goto_video_frame(self.current_video_frame + 1)
                elif key == ord("j"):
                    self.paused = True
                    manual_navigation = self._goto_video_frame(self.current_video_frame - 10)
                elif key == ord("l"):
                    self.paused = True
                    manual_navigation = self._goto_video_frame(self.current_video_frame + 10)
                elif key == ord("m"):
                    self.paused = True
                    self.merge(from_current=False)
                elif key == ord("f"):
                    self.paused = True
                    self.merge(from_current=True)
                elif key == ord("s"):
                    self.paused = True
                    self.swap_from_current()
                elif key == ord("r"):
                    self.paused = True
                    self.remove_primary(from_current=False)
                elif key == ord("e"):
                    self.paused = True
                    self.remove_primary(from_current=True)
                elif key == ord("u"):
                    self.paused = True
                    self.undo()
                elif key == ord("w"):
                    self.paused = True
                    self.save()
                elif key == ord("c"):
                    self.primary_pid = None
                    self.secondary_pid = None
                    self._set_status("Selections cleared")
                elif key == ord("h"):
                    self.show_help = not self.show_help

                # Sequential playback avoids expensive random seeking on every frame.
                if not self.paused and not manual_navigation:
                    if self.current_video_frame >= self.processed_video_end:
                        self.paused = True
                        self._set_status("End of processed CSV frame range")
                    else:
                        self._read_next_frame()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Interactive OpenCV editor for body-tracking CSVs.")
    p.add_argument("video", help="Input video")
    p.add_argument("csv", help="Input tracking CSV")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV. Default: <input_stem>_edited.csv",
    )
    p.add_argument(
        "--log",
        default=None,
        help="Change-log CSV. Default: <output_stem>_changes.csv",
    )
    p.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help=(
            "CSV frame_index = OpenCV video frame + offset. "
            "Use 0 for zero-based CSV frames, 1 for one-based CSV frames."
        ),
    )
    p.add_argument("--max-width", type=int, default=1400)
    p.add_argument("--max-height", type=int, default=900)
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    output = (
        Path(args.output)
        if args.output
        else csv_path.with_name(csv_path.stem + "_edited.csv")
    )

    editor = TrackEditor(
        video_path=args.video,
        csv_path=args.csv,
        output_path=output,
        log_path=args.log,
        frame_offset=args.frame_offset,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    editor.run()


if __name__ == "__main__":
    main()
