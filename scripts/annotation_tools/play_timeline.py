#!/usr/bin/env python3

"""
play_timeline.py

Annotation timeline next to the video, in one matplotlib window.
No browser, no HTML - OpenCV decodes the frames, matplotlib draws.

TXT format (same as visualize_timeline.py):
    session  start_time start_sec  end_time end_sec  duration duration_sec  label

Usage:
    python play_timeline.py annotations.txt --video session.mp4
    python play_timeline.py annotations.txt --video trimmed.mp4 --video-offset 167.571

--video-offset is the second of the original recording that the video starts at
(the same number as trim_start_s). The timeline axis is always in recording
seconds, so with the wrong offset the playhead is shifted but nothing else.

Controls:
    space        play / pause
    left/right   -/+ 5 s          (shift: 1 s)
    click        seek to that point on the timeline
    q            quit
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = ["session", "start_time", "start_sec", "end_time", "end_sec",
           "duration", "duration_sec", "label"]

PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
           "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]


def read_txt(path):
    """Intervals in seconds of the original recording, one row per annotation."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")

    df["tier"] = df["session"].astype(str).str.strip().str.split("_").str[-1]
    df["code"] = df["label"].astype(str).str.strip()
    df["start"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end"] = pd.to_numeric(df["end_sec"], errors="coerce")

    df = df.dropna(subset=["start", "end"])
    df = df[df["end"] >= df["start"]].copy()
    df["row"] = df["tier"] + " : " + df["code"]
    return df


class Player:
    """One figure: video on top, timeline below, a playhead linking them."""

    def __init__(self, df, video, offset=0.0):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {video}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.offset = float(offset)
        self.playing = False

        # --- rows: alphabetical by tier, then by code -------------------------
        pairs = sorted(set(zip(df["tier"], df["code"])))
        self.order = [f"{t} : {c}" for t, c in pairs]
        ypos = {lab: i for i, lab in enumerate(self.order)}

        tiers = sorted({t for t, _ in pairs})
        tier_color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(tiers)}

        # --- layout -----------------------------------------------------------
        height = max(5.0, 0.26 * len(self.order) + 1.8)
        self.fig, (ax_v, ax_t) = plt.subplots(
            1, 2, figsize=(17, height),
            gridspec_kw=dict(width_ratios=[1.6, 2.0]),
        )
        self.ax_t = ax_t

        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"cannot read a frame from {video}")
        self.im = ax_v.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax_v.set_axis_off()
        ax_v.set_anchor("N")
        self.clock = ax_v.set_title("", fontsize=10, family="monospace")

        for _, r in df.iterrows():
            ax_t.barh(ypos[r["row"]], max(r["end"] - r["start"], 1e-6),
                      left=r["start"], height=0.62,
                      color=tier_color[r["tier"]])

        ax_t.set_yticks(range(len(self.order)))
        ax_t.set_yticklabels(self.order, fontsize=8)
        ax_t.set_ylim(-0.7, len(self.order) - 0.3)
        ax_t.invert_yaxis()
        ax_t.set_xlabel("time in the original recording (s)")
        ax_t.grid(axis="x", alpha=0.3)

        # The axis covers the video, not the whole recording: over 2000 s of
        # ELAN the bars inside the processed window collapse into one smear.
        v_end = self.offset + (self.n_frames / self.fps if self.n_frames else 0)
        if v_end > self.offset:
            pad = 0.02 * (v_end - self.offset)
            ax_t.set_xlim(self.offset - pad, v_end + pad)
            ax_t.axvline(self.offset, color="#1565c0", lw=1.0, ls="--")
            ax_t.axvline(v_end, color="#1565c0", lw=1.0, ls="--")

        self.playhead = ax_t.axvline(self.offset, color="#d62728", lw=1.6)

        # --- events -----------------------------------------------------------
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.fps))
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._show(frame, 0)
        self.fig.tight_layout()

    # --- time helpers ---------------------------------------------------------
    def _pos_frame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def _show(self, frame, index):
        """Push one decoded frame and move the playhead to match it."""
        t = self.offset + index / self.fps
        self.im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.playhead.set_xdata([t, t])
        self.clock.set_text(f"{int(t // 60):02d}:{t % 60:06.3f}"
                            f"   |   frame {index}   |   {t:.3f} s")
        self.fig.canvas.draw_idle()

    def _tick(self):
        index = self._pos_frame()
        ok, frame = self.cap.read()
        if not ok:
            self.toggle(False)
            return
        self._show(frame, index)

    def seek(self, t_recording):
        """Jump to a second of the recording; clamped to the video."""
        index = int(round((t_recording - self.offset) * self.fps))
        index = max(0, min(index, max(self.n_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if ok:
            self._show(frame, index)

    def toggle(self, playing=None):
        playing = (not self.playing) if playing is None else playing
        if playing == self.playing:
            return
        self.playing = playing
        self.timer.start() if playing else self.timer.stop()

    # --- input ----------------------------------------------------------------
    def _on_key(self, event):
        if event.key == " ":
            self.toggle()
        elif event.key in ("left", "right", "shift+left", "shift+right"):
            step = 1.0 if "shift" in event.key else 5.0
            step = -step if "left" in event.key else step
            self.seek(self.offset + self._pos_frame() / self.fps + step)
        elif event.key == "q":
            plt.close(self.fig)

    def _on_click(self, event):
        if event.inaxes is self.ax_t and event.xdata is not None:
            self.seek(event.xdata)

    def close(self):
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("input", help="TXT annotation file")
    parser.add_argument("--video", required=True, help="video file")
    parser.add_argument("--video-offset", type=float, default=0.0,
                        help="recording second the video starts at")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    df = read_txt(path)
    print(f"{len(df)} annotations, {df['tier'].nunique()} tiers, "
          f"{df['start'].min():.1f}s .. {df['end'].max():.1f}s")

    player = Player(df, args.video, args.video_offset)
    plt.show()
    player.close()


if __name__ == "__main__":
    main()
