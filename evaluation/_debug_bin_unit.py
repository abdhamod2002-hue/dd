from littering_event_detector import LitteringEventDetector, EventState
from tests.test_littering_event_detector import _fast_config, _person, _bag

detector = LitteringEventDetector(_fast_config())
events = []
t = 0.0
snapshots = []


def snap(label):
    for key, mem in list(detector._pairs.items()):
        snapshots.append(
            {
                "label": label,
                "key": key,
                "state": str(mem.state),
                "aidm_separated": mem.aidm_separated,
                "release_via_aidm_only": mem.release_via_aidm_only,
                "ground": mem.ground_evidence_frames,
                "syn": getattr(mem, "aidm_synthetic_ground", False),
                "max_wrist": mem.max_wrist_d_norm,
                "ever_attach": mem.ever_aidm_attached,
                "ever_wrist": mem.ever_wrist_near,
                "carried": mem.carried_frames,
                "release_frame": mem.release_frame,
            }
        )


for _ in range(6):
    events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
    t += 0.1
snap("after_carry")
for y in (280, 340, 400):
    events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
    t += 0.1
    snap(f"release_y={y}")
for _ in range(6):
    events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
    t += 0.1
snap("after_ground")
for x in (360, 520, 680, 840):
    events.extend(detector.update([_person(1, x, 180)], [_bag(10001, 140, 420)], t))
    t += 0.1
    snap(f"depart_x={x}")
events.extend(detector.finalize())

for s in snapshots:
    print(s)

print("--- events ---")
for e in events:
    print(e.state, e.confirmed, e.reason)
    print(" evidence", e.evidence)
