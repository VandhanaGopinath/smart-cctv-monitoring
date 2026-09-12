"""
Lightweight tracker: IoU matching with a centroid-distance fallback.

Detection answers: "what objects exist in this frame?"
Tracking answers: "is this the same person I saw before?"

Why the fallback exists:
Pure IoU matching only works if a person's box in the current frame still
overlaps their box in the previous *processed* frame. On CPU, YOLO inference
takes tens of milliseconds per frame, so the *effective* frame rate seen by
the tracker is much lower than the camera's real FPS. A person walking at
normal speed can easily move far enough between two processed frames that
their boxes stop overlapping at all (IoU = 0) — the tracker would then
wrongly treat them as a brand-new person, causing IDs (and "unique count")
to climb rapidly even with the same one or two people in frame.

The fix: if no track overlaps a detection well enough by IoU, fall back to
matching by centroid distance, scaled to that track's box size (a person's
box is a natural yardstick for "how far is too far to be the same person").
This is intentionally simple (no motion prediction/Kalman filter); it can
still be fooled by two people swapping positions quickly or crossing paths,
which is a known limitation of this MVP tracker (see README).
"""

import math


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _diag(box):
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def _center_distance(box_a, box_b):
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    return math.hypot(ax - bx, ay - by)


class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox
        self.missed_frames = 0


class CentroidIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 max_center_distance_factor: float = 1.2):
        """
        iou_threshold: minimum overlap to match by bounding-box IoU.
        max_missed: consecutive frames a track can go unmatched before removal
                    (handles brief occlusion).
        max_center_distance_factor: when IoU matching fails, a detection can
            still match a track if its center is within this many multiples
            of the track's box diagonal — i.e. "moved by at most ~1.2 body
            lengths since the last processed frame". Lower = stricter
            (more new IDs on fast motion), higher = more forgiving (more
            risk of merging two different nearby people).
        """
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_distance_factor = max_center_distance_factor
        self.tracks = {}  # id -> Track
        self._next_id = 1

    def update(self, detections):
        """
        detections: list of {"bbox": (x1,y1,x2,y2), "confidence": float}
        Returns: list of {"id": int, "bbox": (x1,y1,x2,y2), "confidence": float}
        """
        unmatched_detections = list(range(len(detections)))

        # Pass 1: match by IoU (best overlap wins) — precise, preferred.
        unmatched_track_ids = []
        for track_id, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for det_idx in unmatched_detections:
                iou = _iou(track.bbox, detections[det_idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx is not None and best_iou >= self.iou_threshold:
                track.bbox = detections[best_det_idx]["bbox"]
                track.missed_frames = 0
                unmatched_detections.remove(best_det_idx)
            else:
                unmatched_track_ids.append(track_id)

        # Pass 2: for tracks IoU couldn't match, fall back to nearest centroid
        # within a distance scaled to the track's own box size.
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            max_dist = _diag(track.bbox) * self.max_center_distance_factor
            best_dist = float("inf")
            best_det_idx = None
            for det_idx in unmatched_detections:
                dist = _center_distance(track.bbox, detections[det_idx]["bbox"])
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            if best_det_idx is not None and best_dist <= max_dist:
                track.bbox = detections[best_det_idx]["bbox"]
                track.missed_frames = 0
                unmatched_detections.remove(best_det_idx)
            else:
                track.missed_frames += 1

        # Remove stale tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id].missed_frames > self.max_missed:
                del self.tracks[track_id]

        # Create new tracks for leftover detections
        for det_idx in unmatched_detections:
            new_id = self._next_id
            self._next_id += 1
            self.tracks[new_id] = Track(new_id, detections[det_idx]["bbox"])

        # Build output for currently-visible tracks (missed_frames == 0 this frame)
        output = []
        for track_id, track in self.tracks.items():
            if track.missed_frames == 0:
                output.append({"id": track_id, "bbox": track.bbox})
        return output
