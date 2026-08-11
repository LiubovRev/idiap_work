# Mask Tool

Unified tool for inspecting mask tracks, applying decisions, and generating annotated tracking videos.

## Install

```bash
pip install opencv-python numpy
```

## Workflow

### 1. Inspect masks

```bash
python mask_tool.py inspect \
  --session_dir /path/to/video/session
```

Controls:

| Key          | Action                    |
| ------------ | ------------------------- |
| `C`          | Assign track as child     |
| `T`          | Assign track as therapist |
| `K`          | Keep without role         |
| `M`          | Merge with another track  |
| `D`          | Delete/ignore             |
| `N`          | Skip                      |
| `SPACE`      | Pause/resume              |
| `LEFT/RIGHT` | Step through frames       |
| `Q`          | Quit and save             |

Decisions are saved automatically to:

```text
7_INDIVIDUAL_14/
└── mask_decisions.json
```

### 2. Apply decisions

```bash
python mask_tool.py apply \
  --session_dir /path/to/video/session
```

Merged masks are written to `MaskDir/`. Deleted tracks are moved to files ending in `_DELETED.mp4` rather than permanently removed.

### 3. Create annotated video

```bash
python mask_tool.py annotate \
  --session_dir /path/to/video/session \
  --time_offset 167
```

The child and therapist tracks are automatically read from `mask_decisions.json`.

Output:

```text
7_INDIVIDUAL_14/
└── visualization_annotated.mp4
```

### Run everything

To inspect, apply decisions, and create the annotated video in one workflow:

```bash
python mask_tool.py all \
  --session_dir /path/to/video/session \
  --time_offset 167
```

## Expected folder structure

```text
7_INDIVIDUAL_14/
├── MaskDir/
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
├── VisualizationVideos/
│   └── visualization_a.mp4
├── *.txt
├── mask_decisions.json
└── visualization_annotated.mp4
```

## Notes

* `--time_offset` is the raw-video timestamp corresponding to frame `0` of the tracking video.
* If there are multiple annotation `.txt` files, specify one with `--annotations_file`.
* Test temporal alignment between the mask videos and tracking video before processing the full dataset.
