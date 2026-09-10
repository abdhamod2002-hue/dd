# Roboflow Colored-Bags Dataset — Initial Inspection & Blocker

**Status:** Investigation paused. Training and per-color evaluation **cannot start** — the
dataset export at `datasets/roboflow_colored_bags/Trash.v1i.yolov8` is single-class
(`Garbage Bag`), not the 8-color structure the task assumes. Awaiting your decision
before any further action.

---

## 1. data.yaml — what was actually on disk

```yaml
train: ../train/images
val:   ../valid/images
test:  ../test/images
nc: 1
names: ['Garbage Bag']
roboflow:
  workspace: school-wzmrm
  project:   trash-hvabf
  version:   1
  license:   CC BY 4.0
```

`nc = 1`, single class `Garbage Bag`. **No** `Black-bags`, `Blue-bags`,
`GarbageBag`, `PlasticBag`, `Purple-bags`, `Red-bags`, `White-bags`, `Yellow-bags`
entries anywhere.

## 2. README confirms the export configuration

`README.roboflow.txt` (relevant excerpt):
> The dataset includes 847 images. Trash are annotated in YOLOv8 format.
> The following pre-processing was applied to each image: auto-orientation,
> resize to 640×640 (Stretch). **No image augmentation techniques were applied.**

So the "847 images / no augmentation / 640×640" details in the brief are accurate,
but the "8 color classes, 0 dropped" detail is **not** — this v1 export
intentionally uses a single class.

## 4. Hard evidence: every annotation is class 0

I parsed every label `.txt` file across all three splits:

| split | image files | label files | annotation lines | class IDs found |
|-------|-------------|-------------|------------------|------------------|
| train | 780 | 780 | 3,837 | `[0]` |
| valid |  37 |  37 |    87 | `[0]` |
| test  |  30 |  30 |    61 | `[0]` |
| **TOTAL** | **847** | **847** | **3,985** | **`[0]`** |

Distinct class IDs across the entire dataset: **`[0]`** (only).
Distribution by the 8 expected colors: **undefined — the classes do not exist**.

## 5. Image-context check (Step 2 of the brief)

Quantitative sample (`random.seed(42)`, 50 train images, fraction of near-white
pixels as a studio-background proxy):

- mean brightness 110.6 / 255 (moderate, natural-leaning)
- mean near-white fraction **0.084**, **median 0.002**, max 0.668
- **5 / 50 (10%)** are clearly studio/artificial background (>50% near-white pixels)

Qualitative sample — 8 images viewed directly:

| image | white-bg fraction | actual context |
|-------|-------------------|----------------|
| `Bag40_jpg.*.jpg`   | 0.668 | yellow bag on pure white — **studio product shot** |
| `Bag85_jpg.*.jpg`   | 0.570 | yellow bag, white bg + shadow — **studio** |
| `Bag75_jpg.*.jpg`   | 0.547 | yellow bag, white bg — **studio** |
| `trash9_jpg.*.jpg`  | 0.537 | four black bags, white bg + noise — **indoor, artificial** |
| `00000255_jpg.*.jpg` | 0.000 | collage of close-ups of a black bag against grass — **close-up handheld** |
| `Bag29_jpeg_jpg.*.jpg` | 0.000 | yellow biohazard bag on a tiled indoor floor — **indoor close-up** |
| `00000419_jpg.*.jpg` | 0.000 | collage of black bags against trees / fence — **close-up handheld** |
| `00002155_jpg.*.jpg` | 0.000 | collage of black / green bags against grass — **close-up handheld** |

**Context conclusion (independent of the class-count problem):**

- ~10% of the dataset is **studio product shots** (pure white background).
- The remaining ~90% are **close-up handheld photos** of bags on indoor
  floors, grass, or against vegetation/fences.
- **None** of the 8 samples resembles our deployment domain (street CCTV:
  wide angle, people walking, mid-distance, daylight or low-light camera
  footage of public roads — the kind captured in `D:\W/IMG_5115..5120.MOV`).

So even if the color classes existed, this dataset's *viewpoint* would already
trigger a strong domain-mismatch warning for transfer to our street camera
context — exactly the warning the earlier TACO comparison raised for its own
data source.

## 6. Why steps 3–6 of the brief cannot be executed as written

| brief step | what's required | status with this dataset |
|------------|-----------------|--------------------------|
| 1. confirm 8 classes + per-color distribution | 8 distinct class IDs in `names:` and labels | ❌ only 1 class exists; per-color distribution is undefined |
| 3. train per-color transfer learning | nc=8 model, per-color mAP50 | ❌ impossible — labels carry no color signal |
| 4. box-quality test for **Yellow-bags** on `D:\W` vs HSV yellow (773 dets) | `Yellow-bags` class in the trained model | ❌ no such class; cannot be evaluated |
| 5. transfer conclusion "yellow weak ⇒ all colors weak" | per-color evidence chain | ❌ no per-color evidence possible |
| 6. eight contact sheets, one per color, sorted by person_iou | per-color detection lists on `D:\W` | ❌ no per-color detections possible |

Doing any of these would either be impossible or would require fabricating
per-color output from a single-class model. Both violate the strict constraint
"no fabrication — every number tied to real evidence".

## 7. Comparison with the previous TACO evaluation

The TACO evaluation (`WASTE_DETECTOR_TACO_REPORT.md`) used a dataset whose
labels *did* contain the relevant classes (filtered down to ~329 garbage
annotations). The per-color metrics, box-quality `box_quality.py` pass, and
the `D:\W` HSV comparison there were all directly measurable. With this
Roboflow export, **none** of those measurements are equivalent: the
class taxonomy itself is missing, so a "this dataset vs TACO" comparison
on mAP50 / box-quality / HSV-yellow is **not meaningful** in its current form.

## 8. Status & awaiting decision

- **Not started:** training, mAP50 evaluation, box-quality pass on `D:\W`,
  contact sheets, comparison-vs-TACO.
- **Not touched:** production HSV yellow pipeline (`inference/detection/...`,
  `trash_bag_tracker.py` left as-is).
- **Next step is yours, not mine.** Please choose how to proceed (suggested
  options):
  1. **Re-export** the Roboflow `school-wzmrm/trash-hvabf` project preserving
     the 8 color classes (likely a newer export version or a different
     preprocessing step), drop it next to this folder, and I'll restart
     Step 1 against the new `data.yaml`.
  2. **Treat as single-class** — proceed with the same rigor but on the
     single "Garbage Bag" class: train, mAP50, box quality on `D:\W`,
     compare to the single-class results from the TACO report and to the
     HSV yellow (as a single-class reference). Per-color metrics,
     per-color contact sheets, and the "yellow-weak ⇒ all-colors-weak"
     transfer argument would then be **out of scope**.
  3. **Different dataset entirely** — the `8 color classes` premise matches
     a different export or a different project; clarify which one and I'll
     re-run the inspection.

No training has been started. No files outside `ROBOFLOW_COLORED_BAGS_DATASET_REPORT.md`
have been written. Production code and the existing HSV yellow path are untouched.

---

## 9. Local-filesystem search for the intended 8-color dataset

After the user's "intended dataset is different" answer, I performed an
exhaustive search of the local filesystem for any dataset that actually
contains the 8 color classes (Black-bags, Blue-bags, GarbageBag,
PlasticBag, Purple-bags, Red-bags, White-bags, Yellow-bags).

Search executed:

| location searched | method | result |
|-------------------|--------|--------|
| `D:\HO\datasets\` (recursive) | full directory listing | only `roboflow_colored_bags/` (single-class), `taco_raw/`, `taco_yolo/` |
| `C:\Users\ASUS\Downloads\*.zip` | enumerate all 11 zips | **only one dataset zip** — `Trash.v1i.yolov8.zip` (44 MB, 2023-07-16) |
| Contents of `Trash.v1i.yolov8.zip` | `unzip -p data.yaml`, README.* | **identical** to the extracted folder: `nc:1`, `names:['Garbage Bag']`, same Roboflow project `school-wzmrm/trash-hvabf` v1, same 847 images, same filenames. Not an alternative. |
| `C:\Users\ASUS\Downloads\` (subdirs) | `grep` for trash/bag/roboflow/dataset/color | only the zip above |
| `C:\Users\ASUS\Desktop` | listing | no dataset folders |
| `D:\` root | listing | no dataset folders |
| Entire `D:` drive | `find -name data.yaml` | **only two** — the single-class Roboflow one, and `taco_yolo/data.yaml` |
| `taco_yolo/data.yaml` | cat | `nc:4`, classes = `bag, bottle, cup, paper` (no colors) |
| `D:\HO` for the color class names | `grep -r "Black-bags\|Yellow-bags\|Red-bags\|White-bags"` over yaml/json/txt | **zero matches** |

**Conclusion:** the 8-color dataset described in the brief does **not**
exist anywhere on this machine — neither as an extracted folder, nor as a
zip, nor as a yaml/json/txt reference. The Roboflow URL embedded in
`data.yaml` (`https://universe.roboflow.com/school-wzmrm/trash-hvabf/dataset/1`)
is the v1 export and is the same single-class archive I already inspected.

To proceed, the 8-class export must be obtained from the Roboflow project.
Possible paths:

1. The Roboflow project `school-wzmrm/trash-hvabf` may have **newer
   generations** (v2/v3/…) or a non-merged preprocessing option on
   Roboflow Universe. Download the right export preserving the 8 color
   classes, place it under `D:\HO\datasets\` (e.g. as
   `roboflow_colored_bags/Trash.vNi.yolov8/`), and I'll re-run Step 1.
2. Provide an alternate URL/path for a different Roboflow project that
   actually contains the 8 color classes (Black-bags, Blue-bags,
   GarbageBag, PlasticBag, Purple-bags, Red-bags, White-bags, Yellow-bags).

Until then, the brief remains blocked for the reasons in §6 above.