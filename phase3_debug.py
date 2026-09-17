"""Debug the regrab reclassification path (refined geometry)."""
from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    LitteringEventDetector,
)


def _person(pid, cx, cy, height=160.0, conf=0.9):
    hw = 40.0
    hh = height / 2.0
    return DetectorPerson(
        track_id=pid,
        bbox=(cx - hw, cy - hh, cx + hw, cy + hh),
        confidence=conf,
        keypoints=DetectorKeypoints(
            left_wrist=(cx + 10, cy + 30),
            right_wrist=(cx - 10, cy + 30),
            torso_center=(cx, cy - 20),
        ),
    )


def _bag(bid, cx, cy, conf=0.5):
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=conf,
        class_name="trash_bag",
        source="yolo",
        yolo_confirmed=True,
    )


cfg = EventDetectorConfig(
    analysis_fps=10.0,
    min_carried_frames=3,
    min_stationary_frames=4,
    min_departed_frames=2,
    confirmation_grace_frames=2,
    smoothing_window=3,
    stationary_window_frames=3,
    max_pair_age_frames=8,
    min_event_confidence=0.65,
)
det = LitteringEventDetector(cfg)
t = 0.0
seq = []
seq += [("carry", 220)] * 8
seq += [("rel", y) for y in (240, 260, 280)]
seq += [("ground", 300)] * 6
seq += [("regrab", y) for y in (280, 260, 240, 220, 220)]
seq += [("carry", 220)] * 6

for label, y in seq:
    det.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t)
    t += 0.1
    for mem in det._pairs.values():
        print(f"{label:7s} y={y:3d} state={mem.state.value:18s} carried={mem.carried_frames} stat={mem.stationary_frames} dep={mem.departed_frames} rel={mem.release_frame} reclaim={mem.reclaimed}")

print("--- finalize ---")
fin = det.finalize()
for e in fin:
    print("FINAL", e.state.value, e.reason, "confirmed=", e.confirmed)
print("summary:", det.summary())
