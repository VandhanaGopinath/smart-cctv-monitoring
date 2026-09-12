"""Drawing helpers, kept separate so the pipeline logic stays readable."""

import cv2


def draw_tracks(frame, tracks):
    """Draw bounding box + ID label for each tracked person."""
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"ID {t['id']}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return frame


def draw_zone(frame, polygon):
    """Draw the restricted-zone polygon outline, if configured."""
    if not polygon:
        return frame
    pts = [(int(x), int(y)) for x, y in polygon]
    for i in range(len(pts)):
        cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], (0, 0, 255), 2)
    return frame


def draw_hud(frame, count, max_occupancy):
    """Draw the current people-count / occupancy readout."""
    text = f"People: {count}"
    if max_occupancy > 0:
        text += f" / {max_occupancy}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame
