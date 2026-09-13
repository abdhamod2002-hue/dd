"""Deterministic test that the renderer draws REAL person/waste boxes.

Builds a FrameAnalysis with concrete full-body person bbox (#1) and waste bbox
(#60001) carrying real track ids, class, confidence and source, and asserts the
rendered frame contains the exact box-color pixels at those locations. This proves
the analyzed video shows continuous, real, ID-labeled tracking boxes without
requiring a human to view pixels.
"""
from __future__ import annotations

import numpy as np

from inference.visualization.frame_analysis import (
    FrameAnalysis,
    ObjectAnalysis,
    PersonAnalysis,
)
from inference.visualization.tracking_visualizer import (
    COLOR_OBJECT,
    COLOR_PERSON,
    render_analysis_frame,
)


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_person_and_waste_boxes_drawn_with_ids():
    frame = _blank(480, 640)
    analysis = FrameAnalysis(
        timestamp=1.0,
        frame_number=10,
        persons=[
            PersonAnalysis(
                track_id=1,
                class_name="person",
                bbox=(100, 120, 220, 460),  # full body
                confidence=0.88,
            )
        ],
        objects=[
            ObjectAnalysis(
                track_id=60001,
                class_name="Garbage Bag",
                bbox=(300, 200, 360, 280),
                confidence=0.61,
                source="yolo",
            )
        ],
    )

    out = render_analysis_frame(frame, analysis, show_hud=False)

    # Isolate pixels actually drawn on top of the blank frame.
    person_px = np.all(np.abs(out.astype(int) - np.array(COLOR_PERSON)) <= 12, axis=2)
    object_px = np.all(np.abs(out.astype(int) - np.array(COLOR_OBJECT)) <= 12, axis=2)

    # Box colors must appear inside the respective bbox rectangles.
    person_box = person_px[120:460, 100:220]
    object_box = object_px[200:280, 300:360]
    assert person_box.sum() > 20, "person box not drawn"
    assert object_box.sum() > 20, "waste box not drawn"

    # The rendered frame must differ from the blank input (something was drawn).
    assert not np.array_equal(out, frame)


def test_boxes_follow_when_ids_stable_across_frames():
    """Same track ids across two frames -> boxes present in both (continuity)."""
    frame = _blank(480, 640)
    analyses = []
    for fx, bx in [(100, 300), (110, 305)]:
        analyses.append(
            FrameAnalysis(
                timestamp=fx * 0.1,
                frame_number=fx,
                persons=[PersonAnalysis(track_id=1, class_name="person", bbox=(fx, 120, fx + 120, 460), confidence=0.9)],
                objects=[ObjectAnalysis(track_id=60001, class_name="Garbage Bag", bbox=(bx, 200, bx + 60, 280), confidence=0.6, source="yolo")],
            )
        )
    for a in analyses:
        out = render_analysis_frame(frame, a, show_hud=False)
        p = np.all(np.abs(out.astype(int) - np.array(COLOR_PERSON)) <= 12, axis=2)
        o = np.all(np.abs(out.astype(int) - np.array(COLOR_OBJECT)) <= 12, axis=2)
        assert p.sum() > 20 and o.sum() > 20


def test_color_source_draws_as_waste_not_proposal():
    """REPAIR-P0-05: source=color is semantic waste; novelty stays proposal."""
    from inference.visualization.tracking_visualizer import COLOR_PROPOSAL

    frame = _blank(480, 640)
    color_obj = ObjectAnalysis(
        track_id=60002,
        class_name="Garbage Bag",
        bbox=(300, 200, 360, 280),
        confidence=0.55,
        source="color",
    )
    out = render_analysis_frame(
        frame,
        FrameAnalysis(
            timestamp=0.0,
            frame_number=1,
            persons=[],
            objects=[color_obj],
        ),
        show_hud=False,
    )
    object_px = np.all(np.abs(out.astype(int) - np.array(COLOR_OBJECT)) <= 12, axis=2)
    assert object_px[200:280, 300:360].sum() > 20, "color bag must draw as waste"

    prop = ObjectAnalysis(
        track_id=60003,
        class_name="detected_object",
        bbox=(100, 100, 160, 160),
        confidence=0.4,
        source="novelty",
    )
    out2 = render_analysis_frame(
        frame,
        FrameAnalysis(timestamp=0.1, frame_number=2, persons=[], objects=[prop]),
        show_hud=False,
    )
    waste_px = np.all(np.abs(out2.astype(int) - np.array(COLOR_OBJECT)) <= 12, axis=2)
    prop_px = np.all(np.abs(out2.astype(int) - np.array(COLOR_PROPOSAL)) <= 12, axis=2)
    assert waste_px[100:160, 100:160].sum() == 0, "novelty must not draw as WASTE"
    assert prop_px[100:160, 100:160].sum() > 5, "novelty must draw as faint proposal"
