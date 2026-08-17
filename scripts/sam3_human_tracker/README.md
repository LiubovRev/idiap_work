# SAM3 Mask Visualization

Utilities for converting SAM3 binary mask tracks into body bounding-box CSV data and visualizing the boxes on the original video.

## Scripts

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

Expected files:

.  
├── video_processed_a.mp4  
├── sam3_body_detections.csv  
├── sam3_masks_to_body_csv.py  
├── visualize_sam3_boxes.py  
└── MaskDir/  
    ├── 0.mp4  
    ├── 1.mp4  
    └── 2.mp4  

Output:

sam3_boxes_visualization.mp4

