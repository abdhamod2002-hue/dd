"""
Part-2 deliverable: build review contact sheets for the TACO classes `bottle`
and `cup` so the user can make the FINAL merge/reject decision visually.

Inputs:
  * D:\\HO\\.audit\\box_quality.json   (39 detections: filename, class, person_iou, label)
  * D:\\HO\\crops\\<class>\\<filename> (full-frame crops with the litter box drawn)

Output (under D:\\HO\\.audit\\contact_sheets\\):
  * bottle_contact_sheet.jpg   (15 tiles, sorted by person_iou DESC = worst first)
  * cup_contact_sheet.jpg      (2 tiles,  sorted by person_iou DESC)
  * contact_sheets_index.json   (machine-readable order + scores, for reference)

Each tile: the real crop + a caption band showing
   <filename>  |  person_iou=<x.xx>  |  <TIGHT/LOOSE/UNUSABLE>

NO merge decision is taken here — this script only prepares the visual material.
"""
import sys, os, json, cv2  # type: ignore
import numpy as np  # type: ignore

ROOT = r"D:\HO"
BQ = os.path.join(ROOT, ".audit", "box_quality.json")
CROPS = os.path.join(ROOT, "crops")
OUT = os.path.join(ROOT, ".audit", "contact_sheets")
os.makedirs(OUT, exist_ok=True)

LABEL_EN = {"ضيق": "TIGHT", "فضفاض": "LOOSE", "غير": "UNUSABLE"}


def caption_for(rec):
    lab = rec.get("label", "")
    en = next((v for k, v in LABEL_EN.items() if lab.startswith(k)), lab)
    return f"{rec['filename']}  |  person_iou={rec['person_iou']:.3f}  |  {en}"


def make_sheet(records, title, ncols):
    # sort worst-first (highest person_iou = box wraps person most)
    records = sorted(records, key=lambda r: float(r.get("person_iou", 0.0)), reverse=True)
    tiles = []
    for rec in records:
        path = os.path.join(CROPS, rec["class"], rec["filename"])
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        # resize to a fixed width, keep aspect
        tw = 300
        h, w = img.shape[:2]
        th = max(60, int(h * tw / w))
        tile = cv2.resize(img, (tw, th))
        # caption band
        band = 46
        canvas = 255 * np.ones((th + band, tw, 3), dtype=np.uint8)
        canvas[:th, :] = tile
        # title line on the band
        cv2.rectangle(canvas, (0, th), (tw, th + band), (20, 20, 20), -1)
        text = caption_for(rec)
        # cv2 can't render Arabic; the caption uses ASCII filename + score + EN label
        cv2.putText(canvas, text[:46], (6, th + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"rank {len(tiles)+1}/{len(records)}  (worst person_iou first)",
                    (6, th + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 200, 255), 1, cv2.LINE_AA)
        tiles.append((canvas, rec))

    if not tiles:
        return None, []

    # grid
    tile_h = tiles[0][0].shape[0]
    tile_w = tiles[0][0].shape[1]
    nrows = (len(tiles) + ncols - 1) // ncols
    # global title band
    title_band = 34
    sheet = 255 * np.ones((title_band + nrows * tile_h + 8, ncols * tile_w + 8, 3), dtype=np.uint8)
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], title_band), (30, 60, 120), -1)
    cv2.putText(sheet, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (tile, rec) in enumerate(tiles):
        r, c = divmod(i, ncols)
        y = title_band + 4 + r * tile_h
        x = 4 + c * tile_w
        sheet[y:y + tile_h, x:x + tile_w] = tile
    return sheet, [rec for _, rec in tiles]


def main():
    data = json.load(open(BQ, encoding="utf-8"))
    by_class = {}
    for rec in data:
        by_class.setdefault(rec["class"], []).append(rec)

    index = {}
    for cls, ncols in (("bottle", 5), ("cup", 2)):
        recs = by_class.get(cls, [])
        if not recs:
            continue
        sheet, ordered = make_sheet(recs, f"TACO class = {cls}  ({len(recs)} detections, worst person_iou first)", ncols)
        if sheet is None:
            continue
        outp = os.path.join(OUT, f"{cls}_contact_sheet.jpg")
        cv2.imwrite(outp, sheet)
        index[cls] = [
            {"rank": i + 1, "filename": r["filename"], "person_iou": round(float(r["person_iou"]), 3),
             "label": r.get("label"), "confidence": r.get("confidence")}
            for i, r in enumerate(ordered)
        ]
        print(f"wrote {outp}  ({len(ordered)} tiles)")

    json.dump(index, open(os.path.join(OUT, "contact_sheets_index.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("wrote contact_sheets_index.json")


if __name__ == "__main__":
    main()
