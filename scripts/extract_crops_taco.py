"""
Visual-verification crop extractor for `taco_transfer_v1.pt` on the D:\\W videos.

This script ONLY produces evidence for HUMAN review. It does NOT make any
correctness judgement, does NOT touch the production pipeline, and does NOT
activate the new model anywhere. It re-runs inference with the exact same
sampling (every 15th frame) and threshold (conf=0.25) used by
`eval_taco_model.py` so the extracted detections match the reported 39 exactly.

For each detection it saves the FULL frame with a drawn bounding box + a label
(class, confidence, video, frame) into crops/<class>/ and records a row in a
summary table whose "review_result" column is LEFT EMPTY for the human to fill.

Output:
  D:\\HO\\crops\\<class>\\<VID>_frame<NNN>_<class>_conf<XX>.jpg
  D:\\HO\\crops\\_summary.csv   (filename,class,confidence,video,frame,review_result)
  D:\\HO\\crops\\_summary.md    (same table + confidence distribution)
"""
import sys, os, csv, json, cv2
sys.path.insert(0, r"D:\HO")
from ultralytics import YOLO

MODEL = r"D:\HO\inference\detection\weights\taco_transfer_v1.pt"
VIDS = [r"D:\W\IMG_5115.MOV", r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
CONF = 0.25
SAMPLE_EVERY = 15
CROPS_ROOT = r"D:\HO\crops"

# Distinct BGR colors per class for clear boxes.
COLOR = {
    "bag":    (0, 255, 255),   # yellow
    "bottle": (0, 255, 0),     # green
    "cup":    (255, 0, 0),     # blue
    "paper":  (255, 0, 255),   # magenta
}


def draw(frame, box, cls, conf, video, frame_no):
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLOR.get(cls, (0, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    # Header (top-left): video + frame
    header = f"{os.path.basename(video)} frame{frame_no}"
    cv2.rectangle(frame, (0, 0), (460, 30), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Box label (above box): class + conf
    label = f"{cls} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 8, ty + 4), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def main():
    model = YOLO(MODEL)
    records = []
    os.makedirs(CROPS_ROOT, exist_ok=True)
    for cls in COLOR:
        os.makedirs(os.path.join(CROPS_ROOT, cls), exist_ok=True)

    for vp in VIDS:
        vname = os.path.basename(vp)
        cap = cv2.VideoCapture(vp)
        total = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            if total % SAMPLE_EVERY != 0:
                continue
            res = model(frame, conf=CONF, verbose=False)[0]
            for b in res.boxes:
                cls = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                box = b.xyxy[0].tolist()
                if cls not in COLOR:
                    continue
                vis = draw(frame.copy(), box, cls, conf, vp, total)
                fname = f"{vname}_frame{total}_{cls}_conf{conf:.2f}.jpg"
                outp = os.path.join(CROPS_ROOT, cls, fname)
                cv2.imwrite(outp, vis)
                records.append({
                    "filename": fname,
                    "class": cls,
                    "confidence": round(conf, 3),
                    "video": vname,
                    "frame": total,
                    "review_result": "",   # LEFT EMPTY for human review
                })
        cap.release()

    # Order: paper first (most suspected), then the rest by class.
    order = {"paper": 0, "bag": 1, "bottle": 2, "cup": 3}
    records.sort(key=lambda r: (order.get(r["class"], 9), r["video"], r["frame"]))

    # Summary CSV
    csv_path = os.path.join(CROPS_ROOT, "_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "class", "confidence", "video", "frame", "review_result"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    # Confidence distribution
    hi = sum(1 for r in records if r["confidence"] > 0.5)
    lo = sum(1 for r in records if 0.25 <= r["confidence"] <= 0.5)
    by_class = {}
    for r in records:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1

    # Summary Markdown
    md = []
    md.append("# ملخّص الكشوفات للمراجعة البشرية (taco_transfer_v1)\n")
    md.append("> هذا الجدول **دليل بصري فقط**. عمود `review_result` فاضي عمدًا — ")
    md.append("المستخدم يملأه بعد المراجعة. لا حكم مسبق على صحة النموذج.\n")
    md.append(f"\n**إجمالي الكشوفات:** {len(records)}\n")
    md.append("\n## توزيع الثقة (Confidence Distribution)\n")
    md.append(f"- ثقة > 0.5: **{hi}** كشف")
    md.append(f"- ثقة 0.25–0.5: **{lo}** كشف\n")
    md.append("\nحسب الفئة:\n")
    for c in ["paper", "bag", "bottle", "cup"]:
        if c in by_class:
            md.append(f"- {c}: {by_class[c]}")
    md.append("\n## جدول الكشوفات\n")
    md.append("| filename | class | confidence | video | frame | review_result |")
    md.append("|---------|-------|------------|-------|-------|---------------|")
    for r in records:
        md.append(f"| {r['filename']} | {r['class']} | {r['confidence']} | {r['video']} | {r['frame']} | {r['review_result']} |")
    md_path = os.path.join(CROPS_ROOT, "_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"TOTAL DETECTIONS SAVED: {len(records)}")
    print(f"  by class: {by_class}")
    print(f"  conf>0.5: {hi}   conf 0.25-0.5: {lo}")
    print(f"  images in: {CROPS_ROOT}/<class>/")
    print(f"  summary csv: {csv_path}")
    print(f"  summary md:  {md_path}")


if __name__ == "__main__":
    main()
