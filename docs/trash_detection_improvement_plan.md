# Trash Detection Improvement Plan

This plan follows the reference project's main lesson: a generic COCO-style
model is often not enough for real trash bags, especially small, low-contrast,
or partially occluded bags.

## Problem

Real videos can show:

- `YOLO Person = PASS`
- `YOLO Object = FAIL`

This happens because trash bags are visually variable:

- black, white, colored, translucent
- crumpled, flat, hanging, rolling
- small in the frame
- low contrast against pavement or shadows
- occluded by hands, legs, vehicles, or foliage

## Data collection

Collect 200–500 real images from the actual deployment scene or similar scenes.

Cover:

- day and night lighting
- black, white, and colored bags
- close, medium, and far distances
- frontal, side, and overhead camera angles
- carried bags and grounded bags
- windy movement and partial occlusion

## Annotation

Use one of these class schemes:

### Simple scheme

```text
trash_bag
```

### Preferred scheme if data allows

```text
trash_bag_carried
trash_bag_ground
```

The split helps the detector distinguish a bag being carried from a bag already
abandoned on the ground.

## Training

Retrain YOLO with:

- Mosaic augmentation enabled
- small-object-friendly input size where compute allows
- WIoU loss if supported by the training stack
- optional P2 high-resolution detection head for very small objects

For YOLOv8/v11, a P2 head can improve small-object recall but increases compute.
Treat it as an experiment, not a default requirement.

## Validation

Do not report invented metrics. Use real clips and record:

- confirmed violations
- true positives
- false positives
- false negatives
- rejection reason counts

Then tune detector thresholds using `docs/littering_event_detector.md`.
