"""
Rule engine: turns tracked people into events/alerts.

Kept separate from detection/tracking so rules can be added/changed
(loitering, new zones, etc.) without touching the AI pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


def _point_in_polygon(point, polygon):
    """Standard ray-casting point-in-polygon test. polygon: list of (x,y)."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class RuleEngine:
    restricted_zone: List[Tuple[int, int]] = field(default_factory=list)  # polygon points, empty = disabled
    max_occupancy: int = 0  # 0 = disabled

    def evaluate(self, tracks):
        """
        tracks: list of {"id": int, "bbox": (x1,y1,x2,y2)}
        Returns: list of alert strings for THIS frame (not deduplicated across frames —
                 caller decides how to log/display history).
        """
        alerts = []
        current_count = len(tracks)

        if self.restricted_zone:
            for t in tracks:
                center = _bbox_center(t["bbox"])
                if _point_in_polygon(center, self.restricted_zone):
                    alerts.append(f"RESTRICTED AREA INTRUSION: Person ID {t['id']}")

        if self.max_occupancy > 0 and current_count > self.max_occupancy:
            alerts.append(
                f"OCCUPANCY ALERT: {current_count} people present (limit {self.max_occupancy})"
            )

        return alerts
