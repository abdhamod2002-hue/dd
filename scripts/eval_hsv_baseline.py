"""
HSV yellow-only color baseline on the SAME 5 D:\\W videos, sampled every 15
frames, to compare against the new YOLO `bag` counts.

We use the RAW color detection (ColorBagTracker._detect_raw) with the PRODUCTION
config (yellow only, verified). This counts color-evidence candidates BEFORE
tracking and before the person-association gate, so it is a conservative
"how much yellow-bag-like color evidence exists" baseline. It is NOT the full
production pipeline (which additionally requires a nearby person track), so it
should be read as an upper-bound on yellow color evidence, not as operational
alert counts.
"""
import sys, os, json, cv2
sys.path.insert(0, r"D:\HO")
from inference.detection.color_bag_detector import ColorBagConfig, ColorBagTracker

VIDS = [r"D:\W\IMG_5115.MOV", r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
SAMPLE_EVERY = 15
OUT = r"D:\HO\.audit\eval_hsv_baseline.json"

def main():
    tracker = ColorBagConfig and ColorBagTracker(config=ColorBagConfig())
    results = {"sample_every": SAMPLE_EVERY, "videos": {}}
    for vp in VIDS:
        name = os.path.basename(vp)
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
            raw = tracker._detect_raw(frame)
            for c in raw:
                cls = c[3]
                agg[cls] = agg.get(cls, 0) + 1
        cap.release()
        results["videos"][name] = {"sampled_frames": sampled, "total_frames": total, "hsv_yellow_raw": agg}
    print(json.dumps(results, indent=2, ensure_ascii=False))
    json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("WROTE", OUT)

if __name__ == "__main__":
    main()
