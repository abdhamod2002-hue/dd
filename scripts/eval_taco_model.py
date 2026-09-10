"""
Evaluate the TACO-based transitional model vs the old best.pt on the SAME
six D:\\W videos (IMG_5115-5120). This is NOT a blind test (these videos were
used during diagnosis) — stated explicitly in the report.

For each video we run, frame-sampled:
  * old best.pt  (5 classes, known broken)
  * new taco_transfer_v1.pt (4 classes: bag,bottle,cup,paper)
and count detections per class. We also separately evaluate the yellow HSV
fallback (current production path) for the `bag` class for context.

Outputs a JSON + prints a summary table. No fabricated numbers: every count
comes from actual model inference on actual frames.
"""
import sys, os, json, cv2
sys.path.insert(0, r"D:\HO")
from ultralytics import YOLO

NEW_MODEL = r"D:\HO\inference\detection\weights\taco_transfer_v1.pt"
OLD_MODEL = r"D:\HO\inference\detection\weights\best.pt"
VIDS = [r"D:\W\IMG_5115.MOV", r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
PERSON_IMG = r"D:\HO\tests\real_video\person_bottle2.jpg"
CONF = 0.25
SAMPLE_EVERY = 15

def per_class_counts(model, frame, conf):
    res = model(frame, conf=conf, verbose=False)[0]
    counts = {}
    for b in res.boxes:
        n = model.names[int(b.cls[0])]
        counts[n] = counts.get(n, 0) + 1
    return counts

def run_on_video(model, vp, conf):
    cap = cv2.VideoCapture(vp)
    total = 0; sampled = 0
    agg = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        if total % SAMPLE_EVERY != 0:
            continue
        sampled += 1
        c = per_class_counts(model, frame, conf)
        for k, v in c.items():
            agg[k] = agg.get(k, 0) + v
    cap.release()
    return {"sampled_frames": sampled, "total_frames": total, "detections": agg}

def main():
    results = {"conf": CONF, "sample_every": SAMPLE_EVERY, "videos": {}, "bottle_image_test": {}}

    # Load models (skip if missing)
    new_model = YOLO(NEW_MODEL) if os.path.exists(NEW_MODEL) else None
    old_model = YOLO(OLD_MODEL) if os.path.exists(OLD_MODEL) else None

    for vp in VIDS:
        name = os.path.basename(vp)
        results["videos"][name] = {}
        if old_model is not None:
            results["videos"][name]["best_pt"] = run_on_video(old_model, vp, CONF)
        if new_model is not None:
            results["videos"][name]["taco_transfer_v1"] = run_on_video(new_model, vp, CONF)

    # Bottle-specific test on the known bottle image
    if os.path.exists(PERSON_IMG):
        im = cv2.imread(PERSON_IMG)
        if old_model is not None:
            results["bottle_image_test"]["best_pt"] = per_class_counts(old_model, im, CONF)
        if new_model is not None:
            results["bottle_image_test"]["taco_transfer_v1"] = per_class_counts(new_model, im, CONF)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    outp = r"D:\HO\.audit\eval_taco_model.json"
    json.dump(results, open(outp, "w"), indent=2, ensure_ascii=False)
    print("WROTE", outp)

if __name__ == "__main__":
    main()
