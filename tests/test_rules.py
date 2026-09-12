import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rules import RuleEngine


def test_zone_intrusion_detected():
    engine = RuleEngine(restricted_zone=[(0, 0), (50, 0), (50, 50), (0, 50)])
    alerts = engine.evaluate([{"id": 1, "bbox": (10, 10, 20, 20)}])
    assert any("INTRUSION" in a for a in alerts)


def test_no_alert_outside_zone():
    engine = RuleEngine(restricted_zone=[(0, 0), (50, 0), (50, 50), (0, 50)])
    alerts = engine.evaluate([{"id": 1, "bbox": (80, 80, 90, 90)}])
    assert alerts == []


def test_occupancy_alert_triggers_over_limit():
    engine = RuleEngine(max_occupancy=2)
    tracks = [{"id": i, "bbox": (80, 80, 90, 90)} for i in range(3)]
    alerts = engine.evaluate(tracks)
    assert any("OCCUPANCY" in a for a in alerts)


def test_no_occupancy_alert_within_limit():
    engine = RuleEngine(max_occupancy=3)
    tracks = [{"id": i, "bbox": (80, 80, 90, 90)} for i in range(2)]
    alerts = engine.evaluate(tracks)
    assert alerts == []


def test_disabled_rules_produce_no_alerts():
    engine = RuleEngine()  # zone and occupancy both disabled
    tracks = [{"id": i, "bbox": (0, 0, 10, 10)} for i in range(10)]
    assert engine.evaluate(tracks) == []
