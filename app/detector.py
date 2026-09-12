"""
Person detector using YOLOv8n (Ultralytics).

WHY YOLOv8n:
- Nano variant (~6MB) runs at usable speed on CPU — required for free-tier deployment.
- Pretrained on COCO; class 0 = "person", so no custom training needed.
- Single-pass CNN detector: far more robust to pose/lighting/scale than the
  classical HOG+SVM detector used in the original project.
"""

from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4):
        """
        model_path: path or name of YOLO weights (auto-downloaded by ultralytics if missing).
        conf_threshold: minimum confidence to keep a detection (filters weak/false detections).
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Run detection on a single BGR frame (numpy array, as read by OpenCV).

        Returns: list of dicts, each: {"bbox": (x1, y1, x2, y2), "confidence": float}
        bbox coordinates are integer pixel coordinates in the input frame.
        """
        results = self.model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": conf,
            })
        return detections
