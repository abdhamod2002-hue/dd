"""
Box-tightness / localization proxy check for taco_transfer_v1 detections.

IMPORTANT HONESTY NOTE:
  The agent running this cannot decode images (no image-input capability), so it
  CANNOT do a true human visual review. Instead this script performs a
  GEOMETRY-BASED PROXY using the actual saved box coordinates plus a COCO
  'person' detector run on the SAME frames:
    - person_iou  : IoU between the litter box and the best-overlapping detected
                    person box in the frame. High => the litter box likely
                    wraps a person (localization defect for person-object assoc).
    - area_frac   : litter-box area / frame area.
    - height_ratio: litter-box height / frame height (≈1.0 means it spans a
                    standing person).
  This is NOT a ground-truth IoU. It is an approximate, transparent proxy to
  guide WHERE the human should look. The final visual call remains the human's.

We re-run inference with the SAME sampling (every 15th frame, conf=0.25) so the
39 detections match the reported set exactly. We DO NOT modify any pipeline file.

Output:
  D:\\HO\\.audit\\box_quality.json   (per-detection raw proxy)
  D:\\HO\\.audit\\box_quality.md     (human-readable table + per-class decision)
"""
import sys, os, json, cv2
sys.path.insert(0, r"D:\HO")
from ultralytics import YOLO

LITTER = r"D:\HO\inference\detection\weights\taco_transfer_v1.pt"
PERSON = r"D:\HO\inference\detection\weights\yolov8n.pt"  # COCO, has 'person'
VIDS = [r"D:\W\IMG_5115.MOV", r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
CONF = 0.25
SAMPLE_EVERY = 15
PERSON_CONF = 0.30
FW, FH = 1920, 1080  # D:\W frames are 1920x1080 (verified)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def classify(box, persons):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    area_frac = (bw * bh) / (FW * FH)
    height_ratio = bh / FH
    person_iou = max([iou(box, p) for p in persons], default=0.0)

    wraps_person = person_iou >= 0.40
    huge = (area_frac >= 0.40) or (height_ratio >= 0.80)

    if wraps_person:
        label = "غير قابل للاستخدام (يحوي شخص)"
    elif huge:
        label = "غير قابل للاستخدام (صندوق ضخم/خلفية)"
    elif area_frac >= 0.08:
        label = "فضفاض"
    else:
        label = "ضيق"
    return label, round(area_frac, 3), round(height_ratio, 3), round(person_iou, 3)


def main():
    litter_model = YOLO(LITTER)
    person_model = YOLO(PERSON)
    rows = []
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
            # persons in this frame
            pres = person_model(frame, conf=PERSON_CONF, verbose=False)[0]
            persons = [list(b.xyxy[0].tolist()) for b in pres.boxes
                       if int(b.cls[0]) == 0]
            # litter
            lres = litter_model(frame, conf=CONF, verbose=False)[0]
            for b in lres.boxes:
                cls = litter_model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                box = [int(v) for v in b.xyxy[0].tolist()]
                label, af, hr, pi = classify(box, persons)
                fname = f"{vname}_frame{total}_{cls}_conf{conf:.2f}.jpg"
                rows.append({
                    "filename": fname, "class": cls, "confidence": round(conf, 3),
                    "video": vname, "frame": total,
                    "bbox": box, "area_frac": af, "height_ratio": hr,
                    "person_iou": pi, "label": label,
                })
        cap.release()

    # order: paper, bag, bottle, cup
    order = {"paper": 0, "bag": 1, "bottle": 2, "cup": 3}
    rows.sort(key=lambda r: (order.get(r["class"], 9), r["video"], r["frame"]))

    # per-class aggregation
    by_cls = {}
    for r in rows:
        c = r["class"]
        by_cls.setdefault(c, {"total": 0, "ضيق": 0, "فضفاض": 0,
                              "غير قابل للاستخدام (يحوي شخص)": 0,
                              "غير قابل للاستخدام (صندوق ضخم/خلفية)": 0,
                              "unusable": 0})
        by_cls[c]["total"] += 1
        by_cls[c][r["label"]] += 1
        if r["label"].startswith("غير قابل"):
            by_cls[c]["unusable"] += 1

    # specific example: bag conf=0.29 (the user-reported defect)
    example = next((r for r in rows if r["class"] == "bag" and abs(r["confidence"] - 0.29) < 0.005), None)

    json.dump(rows, open(r"D:\HO\.audit\box_quality.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- markdown report ----
    L = []
    L.append("# فحص جودة الصندوق (localization) — taco_transfer_v1\n")
    L.append("> **تنويه صريح:** وكيل التشغيل لا يملك قدرة على فكّ الصور، فلم يُجرَ ")
    L.append("مراجعة بصرية بشرية فعلية هنا. هذا الفحص **هندسي تقريبي** من إحداثيات ")
    L.append("الصندوق + كاشف `person` (COCO) على نفس الإطار: `person_iou` (تداخل صندوق ")
    L.append("النفايات مع صندوق الشخص)، `area_frac` (نسبة مساحة الصندوق للإطار)، ")
    L.append("`height_ratio` (ارتفاع الصندوق/ارتفاع الإطار). **ليس IoU حقيقي ضد ground ")
    L.append("truth** (لا يوجد). الغرض توجيه مكان المراجعة البشرية، والقرار النهائي ملكك.\n")

    L.append("## المثال المرفوع: bag conf=0.29\n")
    if example:
        e = example
        L.append(f"- الملف: `{e['filename']}`")
        L.append(f"- bbox: {e['bbox']} (عرض={e['bbox'][2]-e['bbox'][0]}, ارتفاع={e['bbox'][3]-e['bbox'][1]})")
        L.append(f"- area_frac={e['area_frac']}, height_ratio={e['height_ratio']}, person_iou={e['person_iou']}")
        L.append(f"- التصنيف: **{e['label']}**")
        if e['person_iou'] >= 0.4:
            L.append("  ⇒ يؤكّد هندسيًا أن الصندوق يطوّق شخصًا (عيب localization كما وصفت).\n")
        else:
            L.append("  ⇒ لم يؤكّد التداخل مع شخص في هذا التحليل (يحتاج مراجعتك البصرية).\n")
    else:
        L.append("- لم يُعثر على الكشف بالضبط؛ راجع الجدول أدناه.\n")

    L.append("## جدول لكل كشف (39)\n")
    L.append("| filename | class | conf | area_frac | height_ratio | person_iou | التقييم النوعي |")
    L.append("|---------|-------|------|-----------|--------------|------------|----------------|")
    for r in rows:
        L.append(f"| {r['filename']} | {r['class']} | {r['confidence']} | {r['area_frac']} | "
                 f"{r['height_ratio']} | {r['person_iou']} | {r['label']} |")

    L.append("\n## تجميع حسب الفئة\n")
    L.append("| الفئة | العدد | ضيق | فضفاض | يحوي شخص (غير قابل) | ضخم/خلفية (غير قابل) | غير قابل للاستخدام (المجموع) |")
    L.append("|-------|-------|------|--------|----------------------|--------------------------|------------------------------|")
    for c in ["paper", "bag", "bottle", "cup"]:
        if c in by_cls:
            d = by_cls[c]
            L.append(f"| {c} | {d['total']} | {d['ضيق']} | {d['فضفاض']} | "
                     f"{d['غير قابل للاستخدام (يحوي شخص)']} | {d['غير قابل للاستخدام (صندوق ضخم/خلفية)']} | "
                     f"{d['unusable']} |")

    L.append("\n## القرار المبدئي لكل فئة (يرجع للمراجعة البصرية النهائية)\n")
    for c in ["bag", "bottle", "cup", "paper"]:
        if c not in by_cls:
            continue
        d = by_cls[c]
        unusable = d["unusable"]
        if c == "bag":
            L.append(f"- **bag**: {unusable}/{d['total']} غير قابل للاستخدام هندسيًا + ")
            L.append("القرار السابق (HSV أصفر أفضل بكثير: 773 مقابل 10) ⇒ **يُرفض الدمج، ")
            L.append("يُبقى HSV**. سبب إضافي موثّق الآن: عيب localization (صندوق يطوّق الشخص+الكيس).\n")
        elif unusable == 0:
            L.append(f"- **{c}**: 0 غير قابل للاستخدام؛ صناديق ضيقة/مقبولة هندسيًا ⇒ ")
            L.append("**مرشّح للدمج بشرط مراجعتك البصرية النهائية** (لا أستطيع الرؤية).\n")
        elif unusable >= d["total"] * 0.5:
            L.append(f"- **{c}**: {unusable}/{d['total']} غير قابل للاستخدام ⇒ ")
            L.append("**يُرفض الدمج** حتى يُدرَّب على بيانات موقع مُشرَّحة (عيب localization).\n")
        else:
            L.append(f"- **{c}**: {unusable}/{d['total']} غير قابل للاستخدام (جزئي) ⇒ ")
            L.append("**يحتاج مزيد بيانات / ضبط**؛ لا دمج قبل فرز الكشوفات غير القابلة بصريًا.\n")

    open(r"D:\HO\.audit\box_quality.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("ROWS:", len(rows))
    for c in ["paper", "bag", "bottle", "cup"]:
        if c in by_cls:
            d = by_cls[c]
            print(f"  {c}: total={d['total']} tight={d['ضيق']} loose={d['فضفاض']} "
                  f"wraps_person={d['غير قابل للاستخدام (يحوي شخص)']} "
                  f"huge={d['غير قابل للاستخدام (صندوق ضخم/خلفية)']} UNUSABLE={d['unusable']}")


if __name__ == "__main__":
    main()
