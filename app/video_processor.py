"""
Orchestrates: frame -> detect -> track -> rules -> annotated frame + stats.

This is the ONE pipeline shared by webcam, upload, and RTSP modes
(they only differ in how frames are produced — see video_sources.py).

Definitions (kept precise per spec):
- "detections this frame"  = raw YOLO outputs before tracking
- "current_count"          = number of people visible in the CURRENT frame
- "unique_count"           = number of distinct track IDs ever seen so far
  (i.e. total unique individuals across the whole video, not a per-frame number)
"""

from app.detector import PersonDetector
from app.tracker import CentroidIoUTracker
from app.rules import RuleEngine
from app.utils import draw_tracks, draw_zone, draw_hud


class VideoProcessor:
    def __init__(self, detector: PersonDetector, tracker: CentroidIoUTracker, rules: RuleEngine):
        self.detector = detector
        self.tracker = tracker
        self.rules = rules
        self.seen_ids = set()
        self.event_history = []  # list of alert strings, in order, deduplicated per new occurrence
        self._last_alert_set = set()

    def process_frame(self, frame):
        """
        Runs the full pipeline on one frame.
        Returns: dict with annotated_frame, current_count, unique_count, new_alerts
        """
        try:
            detections = self.detector.detect(frame)
        except Exception as e:
            # Model/inference failure on a single frame shouldn't crash the whole stream.
            detections = []
            self.event_history.append(f"DETECTION ERROR (frame skipped): {e}")

        tracks = self.tracker.update(detections)

        for t in tracks:
            self.seen_ids.add(t["id"])

        alerts = self.rules.evaluate(tracks)
        new_alerts = [a for a in alerts if a not in self._last_alert_set]
        self._last_alert_set = set(alerts)
        self.event_history.extend(new_alerts)

        annotated = frame.copy()
        annotated = draw_zone(annotated, self.rules.restricted_zone)
        annotated = draw_tracks(annotated, tracks)
        annotated = draw_hud(annotated, len(tracks), self.rules.max_occupancy)

        return {
            "annotated_frame": annotated,
            "current_count": len(tracks),
            "unique_count": len(self.seen_ids),
            "new_alerts": new_alerts,
        }


def process_video_file(input_path, output_path, detector, tracker, rules, progress_callback=None):
    """
    Runs the full pipeline over an entire video file and writes an annotated
    output video. Used by the "video upload" mode.

    progress_callback(frame_index, total_frames): optional, called per frame
    so the UI can show a progress bar.

    Returns: stats dict (frames_processed, max_current_count, unique_count, event_history)
    Raises: VideoSourceError if the input can't be opened.
    """
    import cv2
    from app.video_sources import VideoSourceError

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise VideoSourceError(f"Could not open uploaded video '{input_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    processor = VideoProcessor(detector, tracker, rules)
    max_current_count = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = processor.process_frame(frame)
            writer.write(result["annotated_frame"])
            max_current_count = max(max_current_count, result["current_count"])
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index, total_frames)
    finally:
        cap.release()
        writer.release()

    return {
        "frames_processed": frame_index,
        "max_current_count": max_current_count,
        "unique_count": processor.unique_count if hasattr(processor, "unique_count") else len(processor.seen_ids),
        "event_history": processor.event_history,
    }
