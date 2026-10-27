import numpy as np
import pytest

from firedet.alerts import AlertManager
from firedet.detection import ColorMotionDetector, Detection
from firedet.fusion import Assessment, FusionConfig, Level, RiskEngine
from firedet.sensors import SensorReading, SimulatedSensor

cv2 = pytest.importorskip("cv2")


def scene(flame_shift=0, with_flame=True):
    frame = np.full((240, 320, 3), (70, 60, 55), np.uint8)
    if with_flame:
        cv2.ellipse(frame, (120, 150 + flame_shift), (30, 55 + flame_shift), 0, 0, 360, (0, 150, 255), -1)
        cv2.ellipse(frame, (120, 160), (15, 30), 0, 0, 360, (120, 230, 255), -1)
    return frame


# ------------------------------------------------------------- vision
def test_color_detector_finds_flickering_flame():
    det = ColorMotionDetector(min_area=200)
    det.detect(scene(0))
    found = det.detect(scene(8))  # shape changed -> flicker
    assert found and found[0].label == "fire"
    x1, y1, x2, y2 = found[0].box
    assert x1 < 120 < x2


def test_color_detector_ignores_static_orange_object():
    det = ColorMotionDetector(min_area=200)
    det.detect(scene(0))
    assert det.detect(scene(0)) == []


def test_color_detector_ignores_empty_scene():
    det = ColorMotionDetector()
    assert det.detect(scene(with_flame=False)) == []


# ------------------------------------------------------------- fusion
FIRE = [Detection("fire", 0.9, (0, 0, 10, 10))]


def test_single_frame_false_positive_does_not_alarm():
    eng = RiskEngine(FusionConfig(window=15))
    assert eng.update(FIRE).level == Level.NORMAL
    for _ in range(5):
        a = eng.update([])
    assert a.level == Level.NORMAL


def test_persistent_fire_alarms():
    eng = RiskEngine(FusionConfig(window=10))
    for _ in range(10):
        a = eng.update(FIRE)
    assert a.level == Level.ALARM


def test_sensors_alone_can_alarm():
    eng = RiskEngine()
    a = eng.update([], SensorReading(temp_c=70, smoke_ppm=900, flame=True))
    assert a.level == Level.ALARM and "IR flame sensor triggered" in a.reasons


def test_normal_sensors_keep_normal():
    eng = RiskEngine()
    assert eng.update([], SensorReading(temp_c=23, smoke_ppm=110)).level == Level.NORMAL


def test_sensor_parse():
    r = SensorReading.parse('{"temp_c": 30.5, "smoke_ppm": 250, "flame": 1}')
    assert r.temp_c == 30.5 and r.flame is True
    assert SensorReading.parse("garbage") is None
    assert SensorReading.parse('{"temp_c": null, "smoke_ppm": 10}').temp_c is None


def test_simulated_sensor_fire_event():
    assert SimulatedSensor(fire_after_s=0).read().flame
    assert not SimulatedSensor(fire_after_s=None).read().flame


# ------------------------------------------------------------- alerts
class Recorder:
    def __init__(self):
        self.msgs = []

    def send(self, subject, body, image_path=None):
        self.msgs.append(subject)


def test_alert_cooldown_and_escalation(tmp_path):
    now = [0.0]
    rec = Recorder()
    mgr = AlertManager([rec], cooldown_s=60, snapshot_dir=str(tmp_path), clock=lambda: now[0])
    warn = Assessment(Level.WARNING, 0.4, 0.4, 0, [])
    alarm = Assessment(Level.ALARM, 0.8, 0.8, 0, [])

    assert mgr.handle(warn)            # first warning
    now[0] = 10
    assert not mgr.handle(warn)        # suppressed by cooldown
    assert mgr.handle(alarm)           # escalation bypasses cooldown
    now[0] = 30
    assert not mgr.handle(alarm)
    now[0] = 100
    assert mgr.handle(alarm)           # cooldown expired
    assert len(rec.msgs) == 3


def test_failing_notifier_does_not_block_others(tmp_path):
    class Broken:
        def send(self, *a, **k):
            raise RuntimeError("smtp down")
    rec = Recorder()
    mgr = AlertManager([Broken(), rec], snapshot_dir=str(tmp_path))
    mgr.handle(Assessment(Level.ALARM, 0.9, 0.9, 0, []), np.zeros((10, 10, 3), np.uint8))
    assert len(rec.msgs) == 1 and list(tmp_path.glob("*.jpg"))
