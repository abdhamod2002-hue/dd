"""
Face-evidence integration test on the 5 real D:\\W videos.

This exercises the REAL production evidence path:
  inference.visualization.evidence_package.write_event_evidence_package
which now internally captures `face_evidence.jpg` via FaceEvidenceCapture and
writes the face fields into metadata.json.

For each video we:
  1. run yolov8n (COCO person) with ByteTrack to get stable person track ids,
  2. build `frame_records` (frame_number + persons[bbox, track_id]) exactly as
     the analysis job does,
  3. pick the dominant person track (most on-screen frames) as the "involved
     person", define the event window as their first_seen->last frame,
  4. call write_event_evidence_package(...) -> real package incl. face_evidence.jpg
     + metadata.json,
  5. record the result (captured? path, size, blur, status, reason).

We do NOT fabricate: if no clear face is found, metadata says
FACE_NOT_CAPTURED and no face image is produced.

Output: D:\\HO\\.audit\\face_evidence\\<VID>\\*  (real evidence files)
        D:\\HO\\.audit\\face_evidence_summary.json
        D:\\HO\\.audit\\face_evidence_report.md
"""
import sys, os, json, cv2
sys.path.insert(0, r"D:\HO")
from ultralytics import YOLO
from inference.visualization.evidence_package import write_event_evidence_package

PERSON_W = r"D:\HO\inference\detection\weights\yolov8n.pt"
VIDS = [r"D:\W\IMG_5115.MOV", r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
SAMPLE_EVERY = 15
OUT_BASE = r"D:\HO\.audit\face_evidence"


def dominant_track(frame_records):
    counts = {}
    first_seen = {}
    last_seen = {}
    for rec in frame_records:
        for p in rec.get("persons", []) or []:
            tid = int(p.get("track_id", -1))
            if tid < 0:
                continue
            counts[tid] = counts.get(tid, 0) + 1
            fn = int(rec["frame_number"])
            first_seen[tid] = min(first_seen.get(tid, fn), fn)
            last_seen[tid] = max(last_seen.get(tid, fn), fn)
    if not counts:
        return None, None, None
    best = max(counts, key=counts.get)
    return best, first_seen[best], last_seen[best]


def main():
    model = YOLO(PERSON_W)
    summary = []
    for vp in VIDS:
        vname = os.path.basename(vp)
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_records = []
        total = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            if total % SAMPLE_EVERY != 0:
                continue
            res = model.track(frame, conf=0.4, persist=True, verbose=False, tracker="bytetrack.yaml")
            persons = []
            if res[0].boxes.id is not None:
                for b in res[0].boxes:
                    if int(b.cls[0]) == 0:
                        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                        persons.append({"track_id": int(b.id[0]), "bbox": [x1, y1, x2, y2]})
            frame_records.append({
                "frame_number": total,
                "timestamp": float(total) / fps,
                "persons": persons,
                "objects": [],
            })
        cap.release()

        tid, f0, f1 = dominant_track(frame_records)
        out_dir = os.path.join(OUT_BASE, vname)
        os.makedirs(out_dir, exist_ok=True)
        if tid is None:
            summary.append({
                "video": vname, "captured": False, "status": "FACE_NOT_CAPTURED",
                "face_evidence_path": None, "reason": "no person detected in video",
                "frame_records": len(frame_records),
            })
            continue

        event = {
            "person_track_id": tid,
            "bag_track_id": 0,
            "frames": {"first_seen": f0, "confirmed": f1, "departure": f1},
            "timestamps": {"first_seen": f0 / fps, "confirmed": f1 / fps},
        }
        pkg = write_event_evidence_package(
            analyzed_video_path=vp,
            target_dir=out_dir,
            event=event,
            frame_records=frame_records,
            job_id=1,
            original_filename=vname,
            source_fps=fps,
            pre_seconds=1.0,
            post_seconds=1.0,
        )
        meta_path = os.path.join(out_dir, "metadata.json")
        # Authoritative source: metadata.json written by the real evidence
        # package (files.face_evidence + face_evidence sub-dict). The return dict
        # from write_event_evidence_package does NOT carry a face_evidence key.
        fe = None
        face_status = None
        reason = None
        size = None
        blur = None
        conf = None
        frame_no = None
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            fe = (meta.get("files", {}) or {}).get("face_evidence")
            fmeta = meta.get("face_evidence", {}) or {}
            face_status = fmeta.get("status")
            reason = fmeta.get("reason")
            size = fmeta.get("face_size_px")
            blur = fmeta.get("blur_score")
            conf = fmeta.get("face_detection_confidence")
            frame_no = fmeta.get("frame_number")
        if not (fe and os.path.exists(fe) and os.path.getsize(fe) > 0):
            fe = None
        summary.append({
            "video": vname,
            "captured": bool(fe),
            "status": face_status,
            "face_evidence_path": fe,
            "face_detection_confidence": conf,
            "frame_number": frame_no,
            "face_size_px": size,
            "blur_score": blur,
            "reason": reason,
            "person_track_id": tid,
            "window_frames": [f0, f1],
            "frame_records": len(frame_records),
        })

    # write summary
    json.dump(summary, open(os.path.join(OUT_BASE, "face_evidence_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    # markdown report
    lines = ["# تقرير التقاط صورة الوجه (face_evidence) — فيديوهات D:\\W\n"]
    lines.append("> كل نتيجة مرتبطة بملف حقيقي تحت `D:\\HO\\.audit\\face_evidence\\<VID>\\`. ")
    lines.append("لم يُختلق أي نجاح: إن لم يُعثر على وجه واضح، الحالة `FACE_NOT_CAPTURED` بلا صورة.\n")
    lines.append("| الفيديو | الحالة | face_evidence.jpg | حجم الوجه(px) | blur | السبب |")
    lines.append("|---------|--------|-------------------|---------------|------|-------|")
    for s in summary:
        lines.append(
            f"| {s['video']} | {s['status']} | "
            f"{'✅ ' + os.path.basename(s['face_evidence_path']) if s['captured'] else '❌ (لا صورة)'} | "
            f"{s.get('face_size_px')} | {s.get('blur_score')} | {s.get('reason')} |")
    lines.append("\n## تفاصيل التخزين والوصول (قانوني/أخلاقي)\n")
    lines.append("- `face_evidence.jpg` تُحفظ في **نفس مجلد حزمة الأدلة** لكل فيديو "
                 "(`D:\\HO\\.audit\\face_evidence\\<VID>\\`)، إلى جانب `person.jpg` و`metadata.json`.")
    lines.append("- في الإنتاج (evidence_package.py) تُحفظ ضمن `EVIDENCE_STORE` الخاص بالوظيفة، "
                 "نفس صلاحيات بقية الأدلة. موسومة `" + "`sensitive: true`" + "` في metadata.")
    lines.append("- **لا تُعرض تلقائيًا** في أي داشبورد عام؛ تخضع لنفس نظام `HUMAN REVIEW REQUIRED` "
                 "كمسار الأدلة الحالي، ولا تُرفع للواجهة إلا بعد مراجعة بشرية.")
    open(os.path.join(OUT_BASE, "face_evidence_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
