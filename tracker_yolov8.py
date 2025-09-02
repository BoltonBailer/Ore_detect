import cv2
import numpy as np
import math
from pathlib import Path
from object_detection import Yolov8ModelStuff

od = Yolov8ModelStuff()

video_path = "videos/ore_vid3.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")


frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 30.0

out_name = "track.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
result = cv2.VideoWriter(out_name, fourcc, fps, size)


save_path = Path(out_name).resolve()
print(f"Saving output to: {save_path}")


frame_idx = 0
center_points_prev_frame = []
tracking_objects = {} 
next_track_id = 0


link_thresh = 45

# Draw settings
box_color  = (0, 255, 0)
text_color = (255, 255, 255)
id_color   = (0, 0, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    detections = od.detect(frame)

    # build for centers 
    center_points_cur_frame = []

    for (x1, y1, x2, y2, conf, cls_id, cls_name, cx, cy) in detections:
        # confidence
        if conf < 0.35:
            continue

        #draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

        center_points_cur_frame.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, id_color, -1)

    
    if frame_idx <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < link_thresh:
                    tracking_objects[next_track_id] = pt
                    next_track_id += 1
    else:
        # Update existing tracks
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, prev_pt in tracking_objects_copy.items():
            found = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < link_thresh:
                    tracking_objects[object_id] = pt
                    found = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    break
            if not found:
                tracking_objects.pop(object_id)

        # Any unmatched centers become new tracks
        for pt in center_points_cur_frame:
            tracking_objects[next_track_id] = pt
            next_track_id += 1

    # Draw track IDs
    for object_id, pt in tracking_objects.items():
        cv2.putText(frame, str(object_id), (pt[0], max(15, pt[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, id_color, 2)

    # Write/Show
    result.write(frame)
    cv2.imshow("YOLOv8 Track", frame)

    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

    center_points_prev_frame = [p for p in tracking_objects.values()]

cap.release()
result.release()
cv2.destroyAllWindows()
print(f"save vid {save_path}")
