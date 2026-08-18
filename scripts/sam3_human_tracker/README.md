# SAM3 Human Tracker




## Scripts

`sam3_masks_to_body_csv.py` - Extracts body bounding boxes from SAM3 binary masks and preserves SAM3 mask IDs as pid.  

`sam3_masks_to_body_csv.sh` - Bash wrapper for SAM3 mask → body CSV extraction.

`sam3_head_body_matching.py` - Associates HumanTracker head detections with SAM3 body detections while preserving SAM3 pid.  

`sam3_head_body_matching.sh` -Bash wrapper for SAM3 head/body matching.  

`visualize_sam3_boxes.py` - Visualizes SAM3 body bounding boxes and PIDs on the original video.  

`visualize_sam3_boxes.sh` - Bash wrapper for SAM3 body-box visualization.  

`visualize_sam3_head_body.py` - Visualizes combined SAM3 body boxes, PIDs, and matched head detections.  

`visualize_sam3_head_body.sh` - Bash wrapper for SAM3 head/body visualization.  

---

### `sam3_masks_to_body_csv.py`

Extracts a body bounding box from each SAM3 binary mask (`0.mp4`, `1.mp4`, `2.mp4`, etc.) for every frame and saves the detections to CSV.

Example:

```bash
python sam3_masks_to_body_csv.py \
    --mask_dir MaskDir \
    --output sam3_body_detections.csv
```

Output format:
frame_index,pid,body_bbox_xmin,body_bbox_ymin,body_bbox_xmax,body_bbox_ymax,...

The pid corresponds to the SAM3 mask filename.

---
### `visualize_sam3_boxes.py`

Draws the SAM3 body bounding boxes from the CSV onto the original video.

```bash
python visualize_sam3_boxes.py
```

