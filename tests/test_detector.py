import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics.utils import ASSETS
from app.detector import PersonDetector


def test_detects_real_people_in_sample_image():
    """
    Sanity check against ultralytics' bundled zidane.jpg, which contains
    two real people. Confirms the model loads and produces plausible
    detections — not a benchmark, just a smoke test.
    """
    detector = PersonDetector(conf_threshold=0.4)
    frame = cv2.imread(str(ASSETS / "zidane.jpg"))
    detections = detector.detect(frame)
    assert len(detections) >= 1
    for d in detections:
        assert 0.0 <= d["confidence"] <= 1.0
        x1, y1, x2, y2 = d["bbox"]
        assert x2 > x1 and y2 > y1
