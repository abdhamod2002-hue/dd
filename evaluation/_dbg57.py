import sys
sys.path.insert(0, '.')
from tests.test_job57_stale_tracking import _person, _bag, _fast_config
from littering_event_detector import LitteringEventDetector, EventState

det = LitteringEventDetector(_fast_config())
t = 0.0
def upd(bags):
    global t
    ev = det.update([_person(1, 140, 180)], bags, t)
    t += 0.1
    return ev

for i in range(6):
    upd([_bag(10001, 140, 220)])
key = next(iter(det._pairs)); mem = det._pairs[key]
print('after carry:', mem.state, 'carried', mem.carried_frames)
for y in (280, 340, 400):
    upd([_bag(10001, 140, y)])
print('after release ticks:', mem.state, 'rel', mem.release_frame, 'live_ground', mem.live_ground_ticks, 'gap', mem.release_object_gap_ticks, 'ground_ev', mem.ground_evidence_frames)
for i in range(6):
    upd([])
    print('missing', i+1, mem.state, 'gap', mem.release_object_gap_ticks, 'inv', mem.release_invalid, 'live_g', mem.live_ground_ticks, 'aband', mem.abandonment_frames)
