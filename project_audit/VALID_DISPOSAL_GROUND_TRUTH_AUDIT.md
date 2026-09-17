# VALID-DISPOSAL GROUND-TRUTH AUDIT — M.MOV + IMG_5613 (Jobs 57, 50)

**Document role:** Read-only ground-truth investigation. **NO code, FSM, threshold, or frozen-set changes were made** (per instruction §11).
**Date:** 2026-09-16. Evidence tags used inline: PROVEN BY REAL VIDEO / PROVEN BY FROZEN REFERENCE / PROVEN BY RUNTIME / INFERRED / NOT VERIFIED.

---

## 1. M.MOV file identity — PROVEN BY RUNTIME

| Item | Value |
|------|-------|
| Source path | `D:/22/M.MOV` |
| SHA-256 | `653962973f158da09910ac8e9d354f26eee17b1877a67c26d2de01fd2a92830a` |
| Uploaded copy | `backend/uploaded_videos/M.MOV` — SHA-256 **IDENTICAL** |
| Duration / frames / size | 8.411667 s / 504 frames / 25,380,106 B / 1920x1080 |
| Job 57 CFR | `evidence_store/analysis/57/source_CFR.mp4` — 252 frames, 30/1 fps, 8.4 s |
| Frozen-set reference | `frozen_test_set.v1.json` -> `M.MOV`, label LITTER, event 5.34 s — same file name; hash match proves no duplicate/stale version |
| Verdict | **The evaluated file IS the recorded file.** |

## 2. M.MOV actual visual ground truth — PROVEN BY REAL VIDEO

Frames: `project_audit/forensics_m/M_t*.jpg` (original file, no overlays):

| t (s) | What the frame shows |
|-------|----------------------|
| 1.0 | Person walking toward camera, small **yellow bag in right hand**, large **green dumpster** at right |
| 3.1 | Still carrying, approaching the dumpster |
| **5.34** (GT "event" time) | Person stands AT the dumpster, **arm raised, yellow bag lifted ABOVE THE BIN RIM** — mid-disposal INTO the container |
| 6.0 | Bag no longer visible at hand (into/over the bin) |
| 6.83 | Person walking away, **hands EMPTY**, nothing new on the ground |
| 7.23 (system confirm) | Person walking away empty-handed; **no yellow bag on the ground anywhere** |
| 8.3 | Person handles a small WHITE object (phone/wipe — not the bag, not litter) |
| — | Ground trash visible in EVERY frame is **PRE-EXISTING** (present at t=1.0 before any action) |

**Visual verdict:** PERSON -> CARRIES YELLOW BAG -> LIFTS IT INTO THE DUMPSTER -> WALKS AWAY EMPTY-HANDED. **VALID DISPOSAL, not littering.**

## 3. M.MOV frozen-label comparison

- Frozen entry: `M.MOV, label=LITTER, event_time=5.34, notes="Drop near parked car then depart"`.
- Reality at 5.34 s: the bag is at its HIGHEST point — being lifted INTO the bin. **No drop to the ground occurs at any time.**
- **Answer: (B) The label is INCORRECT** (option D excluded by the hash match; option C excluded by the 10-frame sequence).

## 4. Second video identity — IMG_5613 (Job 50)

- DB path: `backend/uploaded_videos/20260915_172815_IMG_5613.MOV` (original file later deleted by the operator via DELETE SOURCE VIDEO; **hash NOT VERIFIED**).
- Frames verified from `evidence_store/analysis/50/source_CFR.mp4`; duration 31.1 s matches DB.
- Not in the frozen set. This is the most recent short upload matching "another short video" (**INFERRED** — the user did not name it).

## 5. IMG_5613 actual ground truth — PROVEN BY REAL VIDEO

Frames: `project_audit/forensics_m/IMG5613_t*.jpg`:

| t (s) | What the frame shows |
|-------|----------------------|
| 6.0 | **Empty street** — overflowing dumpster ("F3 29") with heavy PRE-EXISTING litter and white/black sacks around it; **no person** |
| 12.0 | Only a person's ARM at the frame edge (walking past) |
| 18.0 | Person walks past holding a **clear water bottle** — no waste, no disposal, does not stop |
| 24.0 | Empty street + dumpster + pre-existing bags (a cat walks by) |
| 30.0 | Same — no change |

**Visual verdict:** a passer-by with a water bottle walks past an ALREADY-OVERFLOWING dumpster. **No littering action occurs.** The sacks beside the dumpster and the spilled load are pre-existing.

## 6. System result for both — PROVEN BY RUNTIME

| | M.MOV (Job 57) | IMG_5613 (Job 50) |
|---|---|---|
| Result | **VIOLATION_CONFIRMED** | **VIOLATION_CONFIRMED** |
| Confirmed object | `Garbage Bag`, conf 0.9242 | `Garbage Bag`, conf 0.94 |
| Object identity | uid 100001 (track 60009) AND a SECOND simultaneous confirm uid 100002 (track 10006) at the SAME instant (frame 218) | single confirm |
| FSM arc | carry_start f94 (3.1s) -> release f206 (6.83s) -> ground f218 (7.23s) -> departure f242 | same arc shape on a static object |
| detector_source | `color_fallback` | `color_fallback` |

### THE SMOKING GUN (M.MOV `evidence_store/analysis/57/frames.jsonl`):

```
f181-f213: color_candidate_yellow bbox y 594->526->522  (the REAL bag — RISING into the bin; never semantic)
f185-f245: "Garbage Bag" tracks 10006/10011/10014 bbox ~ [474,758,593,1100] / [429,738,571,968] / [551,818,758,1163]
           = THE GREEN DUMPSTER REGION (static; bottom at/below the frame ground line)
f205: BAG_RELEASED   (on the DUMPSTER pair)
f217: BAG_ON_GROUND  (dumpster bbox [469,765,603,984])
f241: VIOLATION_CONFIRMED (conf 0.9242) — ON THE DUMPSTER
```

**The confirmed violation's object IS THE GREEN DUMPSTER.** The actually disposed item (yellow bag) existed only as a `color_candidate`; proposals never enter the FSM by design (Phase B strict-semantic split), so the REAL action was invisible while the CONTAINER was tracked as a carried-then-discarded bag.

**Wrong-object confirmation: PROVEN BY RUNTIME.** A violation attached to the container is a critical false positive regardless of any label question.

## 7. Common failure pattern (both videos) — PROVEN BY CODE + RUNTIME

1. YOLO classifies the large static **dumpster** / large white sacks as semantic `Garbage Bag` (high confidence).
2. **Anti-furniture gate evaded:** the person walks PARALLEL to the static container -> the container bbox appears to "move with the person" (`moves_with_person`), containment passes -> `ever_moved_with_person=True` -> CARRY-ORIGIN satisfied -> BAG_CARRIED.
3. Person keeps walking -> association breaks -> "release" fires on the STATIC container.
4. Container is static and its bottom sits at the ground line -> `stationary` + `near_ground_plane` immediately true -> BAG_ON_GROUND with live ground ticks.
5. Person leaves -> abandonment/departure -> **VIOLATION_CONFIRMED on a container that never moved.**
6. The REAL disposed object (yellow bag) stayed a non-semantic proposal — the FSM never saw the true action.

**General valid-disposal / static-container false-positive class: PROVEN BY RUNTIME on two independent videos** (Question B = YES, independent of Question A).
## 8. Bin/container architecture analysis — PROVEN BY CODE

- **No bin detector exists.** Container semantics = operator-drawn `bin_zones` polygons ONLY (`config/events.yaml`; empty by default). Neither job had zones configured -> `in_bin_zone` was ALWAYS False.
- **No VALID_DISPOSAL state exists** in `EventState` / `RejectionReason`. The FSM can only reject with `BIN_DISPOSAL` (negative evidence: object never observed on the ground plane); it can never POSITIVELY recognize disposal-into-container.
- No motion/occlusion-based container reasoning exists anywhere in `littering_event_detector.py`.
- Bin zones are optional and not required for confirmation.
- **Confirmation timing:** no grace window exists for disposal evidence — and nothing could supply it anyway (no container representation). Confirmation can occur while disposal is still in progress (M.MOV: confirm 1.2 s after the hand released, i.e. while the bag was entering the bin).
- Can a false release happen while the person is still carrying? **YES** — M.MOV: `color_fallback` "Garbage Bag" (dumpster) was released at f206 although the person was still holding/raising the real bag; the two objects were conflated.

## 9. Object/person identity analysis — PROVEN BY RUNTIME

- Raw tracker -> stable UID mapping behaved as coded; the failure is UPSTREAM, not in identity arithmetic.
- M.MOV: violation objects (uid 100001/100002) = DUMPSTER detections; the true yellow bag never became semantic -> **wrong-object violation**.
- IMG_5613: violation object = container/pre-existing sack in a scene where the only moving thing is a passer-by with a bottle -> **wrong-object violation**.
- M.MOV additionally produced a **duplicate confirmation of the same physical container** at the same instant (two uids), i.e. churn created two events for one object.

## 10. First divergence per video

- **M.MOV:** CARRY-ORIGIN accepted for a static container at f94 (3.1 s) because parallel walking fakes motion-sync. Everything after compounds it.
- **IMG_5613:** same — CARRY established on a static object while a passer-by walks past.

## 11. Premature confirmation timing

- M.MOV: bag enters the bin ~5.3-6.0 s; system confirmed 7.23 s — but the confirm was on the wrong object (container), so "premature" understates it.
- IMG_5613: confirm with no disposal action ever occurring.

## 12. Answers to the two questions

- **Question A — frozen M.MOV label wrong?** **YES: INCORRECT** (PROVEN BY REAL VIDEO). The 5.34 s "event" is the bag entering the bin; no ground drop exists. Recommended follow-up (NOT executed): bump `frozen_test_set` version and change M.MOV label to NO_EVENT *after* the architectural repair is validated.
- **Question B — general valid-disposal FP?** **YES** (PROVEN BY RUNTIME, two independent videos): static-container-as-bag + parallel-walk carry origin + no container semantics + no VALID_DISPOSAL state.

## 13. Future repair — DESCRIPTION ONLY (nothing changed)

1. **Static-object veto for CARRY-ORIGIN:** require the OBJECT's OWN displacement (self-motion), not motion-sync with a person walking parallel, before CARRY.
2. **Container suppression:** semantic `Garbage Bag` detections that are static from first sighting and container-scale must never become violation objects (route to proposals/diagnostics).
3. **VALID_DISPOSAL state:** explicit FSM state/evidence able to CANCEL a candidate before persistence when the object path terminates at a container region and never appears on the ground.
4. **Dedup by stable object UID** before emit (M.MOV duplicate confirm at one instant).
5. **Frozen set correction** for M.MOV after (4)+(2) are validated by re-run (version bump + documented).
6. **Re-audit IMG_5613-style "walk-past + pre-existing bins" scenes** as an explicit frozen negative scenario.

## 14. Evidence index

- M.MOV frames: `project_audit/forensics_m/M_t{1,2,3.1,4.2,5.34,6,6.83,7.23,7.8,8.3}s.jpg`
- IMG_5613 frames: `project_audit/forensics_m/IMG5613_t{6,12,18,24,30}s.jpg`
- Runtime: `evidence_store/analysis/57/frames.jsonl`, `evaluation/reports/_job57.json`, DB jobs 24/53/57 (M.MOV) and 50/51 (IMG_5613/5616)
- Hashes: section 1 of this document

_All statements above are limited to what the frames and runtime telemetry show. Nothing about the system's behaviour on other clips is claimed here._
