# Phase 1 — Temporal Persistence Thresholds: before/after report

Date: 2026-08-30 · CPU-only · 5 real videos in `D:\W`
Scope: `MIN_STATIONARY_FRAMES` sweep + absolute-pixel stationarity gate (item #4).
**No other phase was touched.**

---

## 0. Headline

> **The safe ceiling for `MIN_STATIONARY_FRAMES` is 8 frames = 1.00 s — the value
> already in production. There is zero headroom: raising it by a single frame
> (to 9 = 1.125 s) breaks two of the four currently-working videos.**
>
> The 5-second persistence rule from the DumpWatch AI paper is **not reachable**
> on this test set. At 40 frames (5.00 s) **all five videos fail**, including the
> ones that work today.

---

## 1. Current value (as measured, not assumed)

From `config/events.yaml` + `littering_event_detector.py::load_event_config()`:

| Parameter | Value |
|---|---|
| `MIN_STATIONARY_FRAMES` | **8** |
| `ANALYSIS_FPS` | **8.0** |
| → effective threshold | **8 / 8.0 = 1.00 second** |
| `STATIONARY_DISTANCE_RATIO` | 0.15 (of the bag's own bbox size) |
| `STATIONARY_WINDOW_FRAMES` | 3 (displacement measured over 3 ticks) |
| `MIN_ABANDONMENT_FRAMES` | 8 (the alternative path to confirmation) |

So the premise that the value is "set low" is correct: it is **1.0 s**, versus the
5 s the paper used.

### Verified baseline ("before")

The user's brief states IMG_5117 is the only historically confirmed video.
**That is out of date.** The current production configuration confirms 4 of 5:

| Video | Length | Baseline verdict |
|---|---|---|
| IMG_5115 | 5.12 s | REJECTED — `NO_RELEASE_TRANSITION` |
| IMG_5117 | 4.56 s | **CONFIRMED** conf 0.8581 |
| IMG_5118 | 8.06 s | **CONFIRMED** conf 0.8285 (+1 rejected `NOT_ENOUGH_CARRIED_FRAMES`) |
| IMG_5119 | 9.43 s | **CONFIRMED** conf 0.8280 |
| IMG_5120 | 8.41 s | **CONFIRMED** conf 0.8280 |

Source: `.audit/runs/<stem>/report.json` (written 15:18–15:41 today, i.e. after the
current `config/events.yaml` was saved at 13:37).

---

## 2. Method — and proof the harness is trustworthy

Running YOLO+ByteTrack+MoveNet once per candidate value would take hours on CPU
and would mix run-to-run noise into the threshold effect. So the test is split:

* **Stage A (once, 5 min):** run the exact production chain
  (`VideoFileSource → YoloDetector.track → BytetrackTracker → MovenetPose →
  build_tracks_real`) and cache every *analysis tick* to
  `.audit/phase1/tracks/*.jsonl`. Tick counts: 39 / 35 / 61 / 71 / 63 —
  consistent with 8 fps over the clip durations.
* **Stage B (many, milliseconds):** replay the cached ticks into a fresh
  `LitteringEventDetector` with a chosen config. Every candidate value sees
  **byte-identical inputs**, so any difference is caused by the threshold alone.

**Validation:** replaying at the *current* config reproduces the production
result on all 5 videos, confidence matched to 4 decimal places, including the
extra rejected event on IMG_5118:

```
IMG_5115  REJ:NO_RELEASE_TRANSITION(0.2572)        == production 0.2572
IMG_5117  CONFIRMED(0.8581)                        == production 0.8581
IMG_5118  CONFIRMED(0.8285) | REJ:NOT_ENOUGH...(0.2203)  == production, both
IMG_5119  CONFIRMED(0.8280)                        == production 0.8280
IMG_5120  CONFIRMED(0.8280)                        == production 0.8280
```

Scripts: `scripts/phase1_sweep.py`, `phase1_probe_runs.py`, `phase1_pixel_probe.py`,
`phase1_pixel_sweep.py`, `phase1_tolerance_probe.py`.

---

## 3. Sweep result — `MIN_STATIONARY_FRAMES`

Green = confirmed. 8 ticks = 1.00 s.

| frames | sec | IMG_5115 | IMG_5117 | IMG_5118 | IMG_5119 | IMG_5120 | confirmed |
|---:|---:|---|---|---|---|---|---:|
| **8** | **1.00** | REJ `NO_RELEASE_TRANSITION` | **CONFIRMED** | **CONFIRMED** | **CONFIRMED** | **CONFIRMED** | **4 / 5** |
| 9 | 1.12 | REJ `NO_RELEASE_TRANSITION` | CONFIRMED | REJ `EVENT_CONF_TOO_LOW` | CONFIRMED | REJ `PERSON_DID_NOT_DEPART` | 2 / 5 |
| 10 | 1.25 | REJ `NO_RELEASE_TRANSITION` | CONFIRMED | REJ | CONFIRMED | REJ `PERSON_DID_NOT_DEPART` | 2 / 5 |
| 11 | 1.38 | REJ | CONFIRMED | REJ | CONFIRMED | REJ | 2 / 5 |
| 12 | 1.50 | REJ | CONFIRMED | REJ | CONFIRMED | REJ | 2 / 5 |
| 13 | 1.62 | REJ | CONFIRMED | REJ | CONFIRMED | REJ | 2 / 5 |
| 14 | 1.75 | REJ | CONFIRMED | REJ | CONFIRMED | REJ | 2 / 5 |
| 16 | 2.00 | REJ | CONFIRMED | REJ `PERSON_DID_NOT_DEPART` | CONFIRMED | REJ `PERSON_DID_NOT_DEPART` | 2 / 5 |
| 20 | 2.50 | REJ | REJ `PERSON_DID_NOT_DEPART` | REJ | REJ | REJ | 0 / 5 |
| 24 | 3.00 | REJ | REJ | REJ | REJ | REJ | 0 / 5 |
| 32 | 4.00 | REJ | REJ | REJ | REJ | REJ | 0 / 5 |
| 40 | 5.00 | REJ | REJ | REJ | REJ | REJ | 0 / 5 |

**Stopping point, per instruction #3: the first value that breaks a previously
working video is 9 frames (1.125 s)** — it loses IMG_5118 and IMG_5120.

Note the confusing label at N=9 on IMG_5118: `EVENT_CONFIDENCE_TOO_LOW` with
confidence 0.8285 (above the 0.80 gate). That string is a **fallback label**
emitted by `_finalize_pair()` (line 1044) when the pair is force-finalised at
end of clip; it is not a real confidence failure. The real cause is that the
event ran out of clip.

### Why such a sharp cliff? The counter is a *consecutive-run* counter

`littering_event_detector.py:806` — `mem.stationary_frames = 0` on any
non-stationary tick. So the rule is "**N consecutive** stationary ticks", which
is far stricter than "N stationary ticks within a window". Measured longest
consecutive run actually achievable per video (probe with the threshold set
impossibly high so the counter free-runs):

| Video | longest consecutive stationary run | seconds |
|---|---:|---:|
| IMG_5115 | 0 (never reaches RELEASED) | – |
| IMG_5117 | 18 | 2.25 |
| IMG_5118 | 15 | 1.88 |
| IMG_5119 | 18 | 2.25 |
| **IMG_5120** | **8** | **1.00** |

**IMG_5120 is the binding constraint and it has exactly 8.** It reaches
`BAG_ON_GROUND` at f=408 (t=6.81 s) with `stationary_frames = 8` — the very last
tick it can afford. Hence: current value is the ceiling, no headroom.

Two separate limits are in play:
* **Run length** — the stationarity signal itself (5120 tops out at 8).
* **Remaining clip time** — even where a longer run exists, `BAG_ON_GROUND` is
  reached later, leaving too little clip for the 8-tick abandonment window or
  the departure check. That is what kills IMG_5118 at N=9 despite it having 15
  ticks of run available.

Tolerance does **not** help: allowing the counter to survive 1–5 consecutive
misses changes the achievable run by **0 ticks** on every video. The runs are
not being broken by jitter — they end because the clip ends or the track ends.

---

## 4. Item #4 — absolute-pixel stationarity gate

Implemented as a **complementary** gate (`stationary_max_pixel_step`) in
`_is_stationary()`. It is `None` by default, so **production behaviour is
unchanged** until a value is chosen. 27 unit tests pass
(`test_littering_event_detector`, `test_phase2_regressions`, `test_state_machine`).

Measured drift first, before picking a number. Bag centroid displacement per
3-tick window:

| Video | step_px p50 | p90 | max | bag size p50 | ratio gate ≈ |
|---|---:|---:|---:|---:|---:|
| IMG_5115 | 7.9 | 18.1 | 102.0 | 84 px | ~12.6 px |
| IMG_5117 | 3.7 | 32.5 | 107.4 | 80 px | ~12.0 px |
| IMG_5118 | 4.3 | 64.9 | 218.9 | 70 px | ~10.5 px |
| IMG_5119 | 10.4 | 38.8 | 85.6 | 84 px | ~12.6 px |
| IMG_5120 | 10.1 | 58.9 | 119.2 | 67 px | ~10.1 px |

End-to-end effect at `MIN_STATIONARY_FRAMES = 8`:

| cap | IMG_5117 | IMG_5118 | IMG_5119 | IMG_5120 | verdict |
|---|---|---|---|---|---|
| None (current) | 0.8581 | 0.8285 | 0.8280 | 0.8280 | 4/5 |
| **10 px** | 0.8581 | 0.8285 | 0.8280 | **REJECTED** | too strict |
| **12 px** | 0.8581 | 0.8285 | 0.8280 | 0.8280 | 4/5 — safe |
| **15 px** | 0.8581 | 0.8285 | 0.8280 | 0.8280 | 4/5 — safe |
| 20–40 px | 0.8581 | 0.8285 | 0.8280 | 0.8280 | 4/5 — no-op |

* **10 px breaks IMG_5120** → rejected.
* **15 px (the value suggested in the brief) is safe** — identical outcomes and
  identical confidences on all 5 videos.

**Honest caveat:** on these 5 clips the 15 px gate changes *nothing*, so it
cannot be claimed as an improvement. The bags are only 67–84 px across, so the
existing ratio gate already imposes a ~10–13 px budget — stricter than 15 px.
The gate only starts to matter for a **large** bag (e.g. a 400 px bag currently
gets a 60 px budget from the ratio gate, which 15 px would cut). It is
**insurance against a failure mode not present in this test set**, not a
measured gain.

---

## 5. Two defects found while verifying (not fixed — reporting first)

**(a) The stationarity test has no time component.** `DetectorBag.timestamp` is
never populated by `InferencePipeline._detector_bag()`, so it is always `0.0`
(verified: the set of distinct values over a full replay is `{0.0}`). In
`_is_stationary()` this makes `dt = 1e-6`, so
`max(1.0, dt / expected_dt)` is **always exactly 1.0**. The intended
"normalise by the expected analysis window" term is dead code, and the test
collapses to a purely spatial rule: *displacement over 3 ticks < 15 % of the
bag's own size*. Any frame-rate change silently alters its meaning, because the
window is measured in ticks with no seconds attached.

**(b) `MoveNet` exposes no knees.** `inference/pose/movenet_pose.py` provides
only `left_wrist`, `right_wrist`, `left_shoulder`, `right_shoulder`,
`torso_center`, `nose`. Phase 2's carrying-zone spec asks for the vertical band
"between shoulders and knees" — **knees are not available** from the current pose
backend. Flagging now so Phase 2 can be specified against what actually exists
(likely shoulders → bbox bottom, or adding a knee keypoint).

---

## 6. Recommendation

| Item | Recommendation |
|---|---|
| `MIN_STATIONARY_FRAMES` | **Keep 8 (1.00 s). Do not raise it.** The measured ceiling is the current value; +1 frame loses 2 videos. |
| 5 s persistence (paper) | **Reject for this system.** At 5 s all 5 videos fail. The paper's 120 videos were presumably long CCTV takes; ours are 4.5–9.4 s phone clips. Not transferable. |
| `stationary_max_pixel_step` | **Optional, 12 px.** Safe (0 regressions), but inert on this test set. 15 px is also safe; 10 px breaks IMG_5120. I have **left it disabled (`None`)** — say the word and I will set it. |
| Consecutive-run semantics | Worth a separate change: switch to "N stationary ticks within M" so a single jitter tick does not wipe the count. Measured headroom gain on these clips is **0**, so it is not an automatic win — it would need its own phase. |
| `DetectorBag.timestamp` | Worth fixing so the time normalisation actually works. Behaviour-changing; needs its own tested phase. |

### Rollback status
The only production file touched is `littering_event_detector.py`, and only
additively: one new `Optional[float] = None` config field and a guarded `if`
that never executes unless the field is set. **Verified no behavioural change**
(pixel sweep `None` row identical to production; 27 unit tests pass).
`config/events.yaml` was **not** modified.

**Nothing has been changed in production. Awaiting your approval before
proceeding to Phase 2.**
