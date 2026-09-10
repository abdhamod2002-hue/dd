"""
Prepare a TACO-based transitional waste-detection dataset.

- Maps TACO's 60 fine categories -> 4 project classes (bag/bottle/cup/paper).
- Downloads only images that contain >=1 target object (from Flickr 640px URLs).
- Converts COCO polygons/bboxes -> YOLO txt.
- Applies documented quality heuristics (drops extreme macro / tiny-far boxes).
- Splits 80/20 train/val.

Does NOT overwrite best.pt. Output: datasets/taco_yolo/{images,labels}/{train,val}
plus datasets/taco_yolo/data.yaml.
"""
import sys, os, json, csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ROOT = r"D:\HO\datasets\taco_yolo"
RAW = r"D:\HO\datasets\taco_raw"
os.makedirs(ROOT, exist_ok=True)

mapping = json.load(open(os.path.join(RAW, "mapping.json")))
TACO_TO_PROJECT = mapping["taco_to_project"]
PROJ_ID = mapping["project_id"]  # bag,bottle,cup,paper

ann = json.load(open(os.path.join(RAW, "annotations.json")))
cat_by_id = {c["id"]: c["name"] for c in ann["categories"]}
img_by_id = {im["id"]: im for im in ann["images"]}

# Gather target annotations per image
img_targets = defaultdict(list)  # image_id -> list of (proj_id, x,y,w,h)
for a in ann["annotations"]:
    cname = cat_by_id.get(a["category_id"], "?")
    if cname not in TACO_TO_PROJECT:
        continue
    proj = TACO_TO_PROJECT[cname]
    pid = PROJ_ID[proj]
    x, y, w, h = a["bbox"]
    img_targets[a["image_id"]].append((pid, x, y, w, h))

# Build list of (image_id, file_name, url)
todo = []
for imid, targets in img_targets.items():
    im = img_by_id[imid]
    url = im.get("flickr_640_url") or im.get("flickr_url")
    if not url:
        continue
    todo.append((imid, im["file_name"], url, targets, im["width"], im["height"]))
print(f"Images to download: {len(todo)}")

# Prioritize bag-containing images so the capped subset keeps the key class.
todo.sort(key=lambda it: 0 if any(t[0] == 0 for t in it[3]) else 1)

# ---- Download (threaded, resumable, capped, fast timeout) ----
MAX_DOWNLOADS = 500  # cap for a feasible transitional dataset
TIMEOUT = 12

def download(item):
    imid, fname, url, targets, W, H = item
    out_dir = os.path.join(ROOT, "_download")
    os.makedirs(out_dir, exist_ok=True)
    safe = fname.replace("/", "_")
    out_path = os.path.join(out_dir, safe)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 1000:
        return (safe, W, H, targets)  # resume
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
        if len(data) < 1000:
            return None
        open(out_path, "wb").write(data)
        return (safe, W, H, targets)
    except Exception:
        return None

print(f"Downloading (40 threads, cap={MAX_DOWNLOADS}, timeout={TIMEOUT}s)...")
ok = []
done = 0
from concurrent.futures import as_completed
with ThreadPoolExecutor(max_workers=40) as ex:
    futures = [ex.submit(download, it) for it in todo]
    for fut in as_completed(futures):
        res = fut.result()
        if res:
            ok.append(res)
            done += 1
            if done >= MAX_DOWNLOADS:
                break
print(f"Downloaded successfully: {len(ok)} (capped at {MAX_DOWNLOADS})")

# ---- Quality filter + write labels ----
# Heuristic: keep an image only if it has at least one target box with
# area-fraction in [1%, 70%]. <1% = tiny/far (unlikely CCTV-relevant);
# >70% = extreme macro close-up. Documented as a proxy for the requested
# manual context review (full manual review of 50-100 recommended but not
# performed here).
def frac_ok(w, h, W, H):
    f = (w * h) / (W * H)
    return 0.01 <= f <= 0.70

kept = []
dropped = 0
for safe, W, H, targets in ok:
    valid = []
    for (pid, x, y, w, h) in targets:
        if 0.01 <= (w * h) / (W * H) <= 0.70:
            valid.append((pid, x, y, w, h))
    if not valid:
        dropped += 1
        continue
    kept.append((safe, W, H, valid))

print(f"After quality filter: kept={len(kept)}, dropped={dropped}")

# ---- Split 80/20 ----
import random
random.seed(42)
random.shuffle(kept)
split = int(0.8 * len(kept))
splits = {"train": kept[:split], "val": kept[split:]}

for sp, items in splits.items():
    idir = os.path.join(ROOT, "images", sp)
    ldir = os.path.join(ROOT, "labels", sp)
    os.makedirs(idir, exist_ok=True)
    os.makedirs(ldir, exist_ok=True)
    for safe, W, H, valid in items:
        src = os.path.join(ROOT, "_download", safe)
        dst = os.path.join(idir, safe)
        if not os.path.exists(dst):
            try:
                import shutil
                shutil.copy(src, dst)
            except Exception:
                continue
        with open(os.path.join(ldir, os.path.splitext(safe)[0] + ".txt"), "w") as f:
            for (pid, x, y, w, h) in valid:
                cx = (x + w / 2.0) / W
                cy = (y + h / 2.0) / H
                nw = w / W
                nh = h / H
                f.write(f"{pid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

print(f"train images={len(splits['train'])}, val images={len(splits['val'])}")

# ---- data.yaml ----
names = ["bag", "bottle", "cup", "paper"]
yaml_path = os.path.join(ROOT, "data.yaml")
content = f"""# TACO-based transitional waste dataset (4 classes)
path: {ROOT}
train: {os.path.join(ROOT, 'images', 'train')}
val: {os.path.join(ROOT, 'images', 'val')}

nc: 4
names:
  0: bag
  1: bottle
  2: cup
  3: paper
"""
open(yaml_path, "w").write(content)
print("data.yaml written:", yaml_path)
