from littering_event_detector import LitteringEventDetector
from tests.test_littering_event_detector import _person, _bag, _fast_config

det = LitteringEventDetector(_fast_config())
t = 0.0
for i in range(6):
    det.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t)
    t += 0.1
    for mem in list(getattr(det, "_pairs", {}).values()) or list(getattr(det, "pairs", {}).values()):
        pass
# introspect internal pair dict name
pairs = None
for name in ("_pairs", "pairs", "_pair_memory", "_memories"):
    if hasattr(det, name):
        pairs = getattr(det, name)
        print("attr", name, type(pairs), len(pairs) if hasattr(pairs, "__len__") else pairs)
        break
if isinstance(pairs, dict):
    for k, mem in pairs.items():
        print(
            "key", k,
            "state", mem.state,
            "carried_f", mem.carried_frames,
            "rel", mem.release_frame,
            "off", mem.ever_off_ground_while_carried,
            "feet", mem.feet_release_streak,
            "stat", mem.stationary_frames,
        )
fin = det.finalize()
for e in fin:
    print("FINAL", e.reason, e.state, e.frames, "conf", e.confidence)
