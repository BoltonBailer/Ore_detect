import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class Yolov8ModelStuff:
    def __init__(self):
        print("Loading custom YOLOv8 (gold + iron)")

        # Custome model path
        model_path = Path("add_path_here")

        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model: {model_path}")

        # Load trained model
        self.model = YOLO(str(model_path))
        self.names = self.model.names  # dict: {0: 'gold', 1: 'iron'}
        print(f"Loaded model with classes: {self.names}")

    def detect(self, frame):
        """
        Run detection on a single frame.
        Returns: list of [xmin, ymin, xmax, ymax, conf, class_id, class_name, cx, cy]
        """
        detections = self.model(frame, verbose=False)[0]
        data = detections.boxes.data.tolist() if detections.boxes is not None else []

        results = []
        for d in data:
            xmin, ymin, xmax, ymax, conf, class_id = d[:6]
            class_id = int(class_id)
            class_name = self.names.get(class_id, str(class_id))

            # Center point of the bounding box
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            results.append([
                int(xmin), int(ymin), int(xmax), int(ymax),
                float(conf), class_id, class_name, cx, cy
            ])

        return results
