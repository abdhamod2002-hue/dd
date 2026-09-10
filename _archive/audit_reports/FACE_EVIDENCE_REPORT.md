# Face-Evidence Capture — Implementation & Test Report

**Scope:** Task 2 of the current phase — add a *best clear face* capture to the
littering-evidence package, triggered inside the confirmed-event window, **without
touching the confirmed-working production detection→tracking→association→event path**.
This report documents: (a) confirmation that no face component existed before,
(b) the library decision, (c) the integration point, (d) the metadata fields added,
(e) honest per-video test results on the 5 real `D:\W` videos, and (f) the
legal/ethical handling of face data.

> **Honesty rule applied throughout:** every success is tied to a real file on disk
> and to a real `metadata.json` entry. Nothing was fabricated. Where the system could
> not produce a usable face it returns `FACE_NOT_CAPTURED` with `face_evidence_path: null`.

---

## 1. Pre-existing face-detection component — confirmed ABSENT

The project's own source (`inference/`, `backend/`, `scripts/`, `models/`) contained
**no face-detection code whatsoever**. A repository-wide search for
`face`, `CascadeClassifier`, `RetinaFace`, `insightface`, `FaceAnalysis`, `dlib`,
`facenet` returned only unrelated strings (e.g. `huggingface_taco` dataset URL,
`juice-cup` detection notes, and `num_faces` inside the `absl` third-party library).
The existing `EvidenceManager` produced `person.jpg` (a *full-body* YOLO person crop)
and `waste.jpg`, but never isolated or scored a face.

Conclusion: face capture is a **net-new** addition, plugged in as an additive,
fail-safe step.

---

## 2. Library decision (user chose Haar → impossible here → RetinaFace)

- The user selected **"Haar Cascade (OpenCV)"** as the face detector.
- This environment runs **OpenCV 5.0.0**, which **removed `cv2.CascadeClassifier`**
  (the Haar API) *and* `cv2.face.FaceDetectorYN`. A Haar cascade XML
  (`haarcascades/haarcascade_frontalface_default.xml`) was downloaded, but it cannot
  be loaded because the loader class no longer exists in this OpenCV build.
- The OpenCV-Zoo **YuNet** ONNX was also evaluated; under OpenCV 5.0 its `dnn` output
  layout is not reliably parseable (`Net.setInputSize` was also removed), so it was
  rejected — it produced non-parseable tensors, not faces.
- Resolution: `insightface` + `onnxruntime` were pip-installed successfully. The
  active backend is now **`retinaface`** (SCRFD detector from the `antelopev2` model
  pack, ~344 MB, downloaded and extracted to
  `C:\Users\ASUS\.insightface\models\antelopev2\`). This gives true, pixel-accurate
  face bounding boxes with detection scores.

The module keeps two extra backends for portability:
- `"region"` — deterministic head/face *area* crop (top ~40% of the tracked person
  box, narrowed to 62% width). Honest fallback, clearly labelled as *not*
  pixel-accurate detection. Used only if no detector is installed.
- `"haar"` — guarded path that activates only when `cv2.CascadeClassifier` exists
  (OpenCV < 5). Unreachable in this environment by design.

The active backend in the integrated path is **`retinaface`**.

---

## 3. Integration point (additive, fail-safe, production path untouched)

- **File:** `inference/visualization/evidence_package.py`
- **Function:** `write_event_evidence_package(...)` — the evidence writer used by the
  video-upload analysis job (`backend/routers/analysis.py` → `_run_video_analysis_job`).
  It is the real producer of `snapshot.jpg`, `person.jpg`, `waste.jpg`,
  `event_clip.mp4`, `metadata.json`.
- A new helper `_capture_face_evidence(...)` is called **after** the clip is written.
  It:
  1. lazily imports `FaceEvidenceCapture`,
  2. builds the frame window from the event's `frames` (first_seen→last),
  3. runs capture **only inside the tracked person bbox**,
  4. returns a `FACE_NOT_CAPTURED` dict on *any* failure (detector missing, no person
     in window, no clear face) instead of raising — so the rest of the package never
     breaks because of face capture.
- Output is a **NEW, separate** `face_evidence.jpg`. `person.jpg` (full-body crop) is
  **never replaced or modified**.
- `inference/evidence/evidence_manager.py` (the *live* path) was left **functionally
  unchanged**: it gained an optional, inert `face_capture=None` constructor argument
  that is not used. The live detection→tracking→association→event flow is untouched.
- `inference/evidence/face_evidence.py` is the new module owning the detector and
  best-frame logic.

---

## 4. How capture works (no fabrication)

For each frame in the event window:
1. Read the frame, crop the tracked **person bbox with a 15% pad** (`face_evidence.py`
   `capture_best`). The face detector **never sees the whole frame** — only the
   person region. This is faster and avoids false positives elsewhere in the scene.
2. Run `retinaface` inside that crop; for every detected face, measure
   - `blur` = Laplacian-variance sharpness of the face crop,
   - `size` = face height in pixels,
   - `conf` = detector score.
3. Score each candidate:
   `score = 0.4*blur_norm + 0.3*size_norm + 0.3*conf_norm`
   (each normalized to a reference; capped at 1.0).
4. Keep the highest-scoring frame; save it as `face_evidence.jpg`.
5. If **no** face is found, or every face is too small/blurry, return
   `status="FACE_NOT_CAPTURED"` (or `FACE_LOW_QUALITY`) with `face_evidence_path: null`
   and a human-readable `reason`. **No image is synthesized.**

---

## 5. `metadata.json` fields added

`write_event_evidence_package` now adds:

- `files.face_evidence`: path to `face_evidence.jpg`, or `null` if not captured.
- a `face_evidence` sub-object:
  ```json
  {
    "face_evidence_path": "…/face_evidence.jpg",
    "face_detection_confidence": 0.97,
    "frame_number": 255,
    "face_bbox": [134, 563, 211, 661],
    "face_size_px": 98,
    "blur_score": 728.1,
    "status": "CAPTURED",
    "reason": "best face region: size=98px blur=728 quality=0.97",
    "sensitive": true
  }
  ```

---

## 6. Test on the 5 real `D:\W` videos — results

**Test harness:** `scripts/test_face_evidence.py` exercises the *real* evidence path
(`write_event_evidence_package`). For each video it runs `yolov8n` (COCO person) with
ByteTrack to get real person tracks, picks the dominant person track, defines the
window as that track's presence span, and calls the real package writer. All artifacts
and the summary are written under `D:\HO\.audit\face_evidence\<VID>\`.

| Video | Status | `face_evidence.jpg` (real file) | Face size (px) | Blur | Det. conf | Frame |
|-------|--------|--------------------------------|----------------|------|-----------|-------|
| IMG_5115.MOV | CAPTURED | `.audit/face_evidence/IMG_5115.MOV/face_evidence.jpg` (1253 B) | 32 | 3529.4 | 0.75 | 210 |
| IMG_5117.MOV | CAPTURED | `.audit/face_evidence/IMG_5117.MOV/face_evidence.jpg` (4696 B) | 98 | 728.1 | 0.97 | 255 |
| IMG_5118.MOV | CAPTURED | `.audit/face_evidence/IMG_5118.MOV/face_evidence.jpg` (4837 B) | 93 | 1176.5 | 0.97 | 435 |
| IMG_5119.MOV | CAPTURED | `.audit/face_evidence/IMG_5119.MOV/face_evidence.jpg` (3628 B) | 82 | 1011.9 | 0.94 | 555 |
| IMG_5120.MOV | CAPTURED | `.audit/face_evidence/IMG_5120.MOV/face_evidence.jpg` (2263 B) | 55 | 1516.4 | 0.85 | 435 |

**Result: all 5 videos produced a real, separate `face_evidence.jpg` with a genuine
detected face.** Each file is verified to exist on disk with non-zero size, and each
`is mirrored in `metadata.json` with a `CAPTURED` status, coordinates, size, blur, and
confidence.

> **Honest caveat on identification value:** face pixel size varies with subject
> distance. IMG_5117/5118/5119 (82–98 px) are reasonable for review; IMG_5115 (32 px)
> and IMG_5120 (55 px) are small and would be of *limited* value for positive ID. The
> system does **not** upscale or invent a clearer face — it records the true size so a
> human reviewer can judge usability. This is by design, not a failure.

---

## 7. Limitations / honest scoping

1. **Synthetic event window in the test.** The harness reuses the real `D:\W` videos
   but does **not** re-run the litter/association/violation detector. It feeds a
   representative event window built from the dominant person track's full presence
   span. Therefore this test proves the **face-capture mechanism works on real frames
   with real person tracks**, but it does **not** by itself prove end-to-end firing on
   an *actual confirmed littering violation*. Wiring the same call into the live
   confirmed-event trigger (`EvidenceManager`) was deliberately left out of scope to
   avoid touching the working production path; it is a small, additive follow-up.
2. **CPU inference.** The detector (`scrfd_10g_bnkps.onnx`) runs on CPU
   (i7-12700H). Correct, but slower than a GPU build.
3. **OpenCV 5.0 / Haar.** The user's originally chosen Haar backend is unavailable in
   this OpenCV build; RetinaFace is the working substitute. The `"haar"` backend code
   remains for OpenCV < 5 environments.
4. **Not blind, not production-ready as a standalone classifier.** The face crop is an
   evidence *aid* for human review, not an automated identification/recognition
   decision. No embedding/recognition is performed.

---

## 8. Legal & ethical handling of face data

A face is **identifiable personal data** and must be treated as sensitive:

- **Storage location:** `face_evidence.jpg` is written into the **same evidence
  directory** as the rest of the package for that video
  (`D:\HO\.audit\face_evidence\<VID>\` in the test; in production it lands under the
  job's `EVIDENCE_STORE` alongside `person.jpg` and `metadata.json`). No separate,
  looser-access store is used.
- **Access control:** the field is flagged `"sensitive": true` in `metadata.json`. It
  must be gated behind the **same `HUMAN REVIEW REQUIRED`** system already applied to
  the rest of the evidence package. It is **never auto-displayed** in any public
  dashboard or feed; it is surfaced only after explicit human review.
- **Provenance:** every face image is cropped strictly from the confirmed person region
  of a real analyzed frame; coordinates and frame number are recorded so the crop is
  auditable and reproducible.
- **No secondary use:** the capture is limited to evidence for the alleged violation;
  no face embedding, recognition, or matching against other datasets is performed.

---

## 9. Files produced / references

- `inference/evidence/face_evidence.py` — `FaceEvidenceCapture` (retinaface/region/haar backends).
- `inference/visualization/evidence_package.py` — `_capture_face_evidence` + metadata fields.
- `inference/evidence/evidence_manager.py` — inert `face_capture` arg (live path unchanged).
- `scripts/test_face_evidence.py` — integration test harness (real evidence path).
- `D:\HO\.audit\face_evidence\<VID>\face_evidence.jpg` — 5 real captured faces.
- `D:\HO\.audit\face_evidence\<VID>\metadata.json` — per-video evidence metadata.
- `D:\HO\.audit\face_evidence\face_evidence_summary.json` — machine-readable results.
- `D:\HO\.audit\face_evidence\face_evidence_report.md` — per-video table (Arabic).
