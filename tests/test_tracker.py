import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tracker import CentroidIoUTracker


def test_id_persists_across_small_movement():
    t = CentroidIoUTracker(iou_threshold=0.3, max_missed=2)
    out1 = t.update([{"bbox": (10, 10, 50, 100), "confidence": 0.9}])
    out2 = t.update([{"bbox": (15, 10, 55, 100), "confidence": 0.9}])
    assert out1[0]["id"] == out2[0]["id"]


def test_id_survives_brief_occlusion():
    t = CentroidIoUTracker(iou_threshold=0.3, max_missed=2)
    out1 = t.update([{"bbox": (10, 10, 50, 100), "confidence": 0.9}])
    t.update([])  # occluded frame
    out2 = t.update([{"bbox": (20, 10, 60, 100), "confidence": 0.9}])
    assert out1[0]["id"] == out2[0]["id"]


def test_distinct_people_get_distinct_ids():
    t = CentroidIoUTracker()
    out = t.update([
        {"bbox": (10, 10, 50, 100), "confidence": 0.9},
        {"bbox": (200, 200, 240, 300), "confidence": 0.85},
    ])
    ids = [d["id"] for d in out]
    assert len(set(ids)) == 2


def test_track_dropped_after_max_missed():
    t = CentroidIoUTracker(max_missed=1)
    t.update([{"bbox": (10, 10, 50, 100), "confidence": 0.9}])
    t.update([])  # missed frame 1
    t.update([])  # missed frame 2 -> should be dropped now
    # A detection appearing in a totally different spot should get a NEW id
    out = t.update([{"bbox": (10, 10, 50, 100), "confidence": 0.9}])
    assert out[0]["id"] == 2


def test_id_survives_large_frame_to_frame_jump():
    """
    Regression test: on CPU, YOLO inference latency means the effective
    frame rate seen by the tracker is much lower than real camera FPS, so a
    walking person's box can move far enough between processed frames that
    it no longer overlaps the previous box at all (IoU=0). Pure IoU matching
    would treat each jump as a new person -- this must not happen, or
    unique-person counts explode with only one real person in frame.
    """
    t = CentroidIoUTracker()
    positions = [(10, 10, 60, 150), (90, 15, 140, 155), (170, 10, 220, 150), (250, 15, 300, 155)]
    ids_seen = set()
    for pos in positions:
        out = t.update([{"bbox": pos, "confidence": 0.9}])
        ids_seen.update(d["id"] for d in out)
    assert len(ids_seen) == 1


def test_distinct_far_apart_people_stay_distinct_under_motion():
    """Companion to the jump test: the centroid fallback must not merge two
    genuinely different people just because both are moving."""
    t = CentroidIoUTracker()
    t.update([
        {"bbox": (10, 10, 60, 150), "confidence": 0.9},
        {"bbox": (400, 10, 450, 150), "confidence": 0.9},
    ])
    out = t.update([
        {"bbox": (90, 15, 140, 155), "confidence": 0.9},
        {"bbox": (480, 15, 530, 155), "confidence": 0.9},
    ])
    assert len({d["id"] for d in out}) == 2
