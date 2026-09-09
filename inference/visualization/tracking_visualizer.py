"""Professional tracking visualization renderer.

This renderer consumes FrameAnalysis objects built from the production
pipeline. It does not run inference and does not invent boxes, IDs,
trails, states, or events.

Phase-1: box/label rendering is delegated to the proven open-source
Supervision annotators (BoxAnnotator / LabelAnnotator) via
supervision_annotator.annotate_with_supervision — guaranteeing
ONE ACTIVE TRACK -> ONE CURRENT BBOX -> ONE LABEL and deterministic
track-color mapping. Historical boxes are never rendered; trails are
lines only. Proposals remain faint and never WASTE.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from inference.visualization.frame_analysis import FrameAnalysis

# BGR colors chosen for dark surveillance UI readability.
COLOR_PERSON = (255, 220, 80)
COLOR_OBJECT = (80, 220, 255)
COLOR_TRAIL_PERSON = (180, 120, 40)
COLOR_TRAIL_OBJECT = (40, 160, 200)
COLOR_ASSOC = (80, 255, 120)
COLOR_HEAD = (255, 255, 255)
COLOR_EVENT = (80, 80, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (30, 25, 20)
COLOR_PROPOSAL = (90, 90, 90)  # faint gray for non-semantic proposals (never WASTE)


def _scale_for(frame) -> Dict[str, Any]:
    h = int(frame.shape[0])
    w = int(frame.shape[1])
    base = max(0.45, min(w, h) / 1400.0)
    return {
        "font_scale": base,
        "small_scale": max(0.35, base * 0.72),
        "thickness": max(1, int(round(base * 2))),
        "box_thickness": max(3, int(round(base * 3))),
        "pad": max(8, int(round(base * 18))),
        "w": w,
        "h": h,
    }


def _put_text(cv2, img, text: str, org, scale: float, color, thickness: int = 1, bg: bool = True):
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + baseline + 2), COLOR_PANEL, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_trail(cv2, img, points, color, thickness: int):
    if len(points) < 2:
        return
    pts = [(int(x), int(y)) for x, y in points]
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, max(1, thickness - 1), cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 2 if i < len(pts) - 1 else 4
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


def _draw_label(cv2, img, text: str, x: int, y: int, scale: float, color, thickness: int):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(2, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, y)
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 2), COLOR_PANEL, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_analysis_frame(
    frame: Any,
    analysis: FrameAnalysis,
    *,
    event: Optional[Dict[str, Any]] = None,
    show_hud: bool = True,
) -> Any:
    """Render a professional tracking overlay from real FrameAnalysis data."""
    import cv2  # lazy import

    out = frame.copy()
    s = _scale_for(out)
    event = event or analysis.event

    # Trails first, so boxes remain readable.
    for p in analysis.persons:
        _draw_trail(cv2, out, p.trail, COLOR_TRAIL_PERSON, s["thickness"])
    for o in analysis.objects:
        _draw_trail(cv2, out, o.trail, COLOR_TRAIL_OBJECT, s["thickness"])

    # Association lines only when the production detector reports a real pair.
    for a in analysis.associations:
        x1, y1 = int(a.start[0]), int(a.start[1])
        x2, y2 = int(a.end[0]), int(a.end[1])
        cv2.line(out, (x1, y1), (x2, y2), COLOR_ASSOC, max(2, s["thickness"]), cv2.LINE_AA)
        cv2.circle(out, (x1, y1), max(3, s["thickness"] + 1), COLOR_ASSOC, -1, cv2.LINE_AA)
        cv2.circle(out, (x2, y2), max(3, s["thickness"] + 1), COLOR_ASSOC, -1, cv2.LINE_AA)
        _draw_label(
            cv2,
            out,
            f"ASSOCIATED P{a.person_track_id}->B{a.object_track_id}",
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            s["small_scale"],
            COLOR_ASSOC,
            s["thickness"],
        )

    # --- Supervision box/label pass: ONE ACTIVE TRACK -> ONE CURRENT BBOX ---
    # Delegates to proven open-source annotators. Falls back to legacy cv2
    # if supervision is not installed (tests on host without deps).
    try:
        from inference.visualization.supervision_annotator import annotate_with_supervision

        out = annotate_with_supervision(out, analysis, s)
    except Exception:
        # Legacy fallback — keep exact previous behavior if supervision fails.
        # Persons.
        for p in analysis.persons:
            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_PERSON, s["box_thickness"], cv2.LINE_AA)
            state = f" \u00b7 {p.state.replace('_', ' ')}" if p.state else ""
            _draw_label(cv2, out, f"PERSON #{p.track_id}{state}", x1, y1 - 4, s["font_scale"], COLOR_PERSON, s["thickness"])
            _draw_label(
                cv2,
                out,
                f"CONF {p.confidence:.2f}",
                x1,
                y1 + int(22 * s["font_scale"] / 0.5),
                s["small_scale"],
                COLOR_TEXT,
                s["thickness"],
            )
        # Objects
        for o in analysis.objects:
            x1, y1, x2, y2 = [int(v) for v in o.bbox]
            cls_low = str(o.class_name or "").lower()
            is_proposal = (
                str(o.source).lower() != "yolo"
                or cls_low.startswith("color_candidate")
                or cls_low == "detected_object"
            )
            if is_proposal:
                cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_PROPOSAL, 1, cv2.LINE_AA)
                _draw_label(cv2, out, f"PROPOSAL {o.class_name} #{o.track_id}", x1, y1 - 4, s["small_scale"] * 0.85, COLOR_PROPOSAL, 1)
                continue
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_OBJECT, s["box_thickness"], cv2.LINE_AA)
            source = "YOLO"
            state = f" \u00b7 {o.state.replace('_', ' ')}" if o.state else ""
            _draw_label(cv2, out, f"{o.class_name.upper()} #{o.track_id}{state}", x1, y1 - 4, s["font_scale"], COLOR_OBJECT, s["thickness"])
            _draw_label(
                cv2,
                out,
                f"CONF {o.confidence:.2f} \u00b7 {source}",
                x1,
                y1 + int(22 * s["small_scale"] / 0.35),
                s["small_scale"],
                COLOR_TEXT,
                s["thickness"],
            )

    # Head/face visibility (no recognition) — per-person, not boxes.
    for p in analysis.persons:
        if p.head_bbox is not None:
            hx1, hy1, hx2, hy2 = [int(v) for v in p.head_bbox]
            cv2.rectangle(out, (hx1, hy1), (hx2, hy2), COLOR_HEAD, max(1, s["thickness"]), cv2.LINE_AA)
            _draw_label(cv2, out, "HEAD", hx1, hy1 - 3, s["small_scale"], COLOR_HEAD, s["thickness"])
        else:
            # Keep original keypoint-stage label placement for consistency.
            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            _draw_label(cv2, out, "FACE NOT VISIBLE", x1, y2 + int(16 * s["small_scale"] / 0.35), s["small_scale"], (170, 170, 170), s["thickness"])

        # Keypoints.
        kp = p.keypoints or {}
        for key, color in (
            ("left_wrist", COLOR_PERSON),
            ("right_wrist", COLOR_PERSON),
            ("torso_center", COLOR_HEAD),
            ("nose", COLOR_HEAD),
        ):
            pt = kp.get(key)
            if pt is None:
                continue
            cv2.circle(out, (int(pt[0]), int(pt[1])), max(3, s["thickness"] + 1), color, -1, cv2.LINE_AA)

    # HUD.
    if show_hud:
        hud = analysis.hud or {}
        lines = [
            "AI ENGINE: RUNNING",
            f"VIDEO: {hud.get('video', 'video')}",
            f"SOURCE FPS: {hud.get('source_fps') or '—'}   ANALYSIS FPS: {hud.get('analysis_fps') or '—'}",
            f"PERSONS: {hud.get('persons', 0)}   OBJECTS: {hud.get('objects', 0)}   PAIRS: {hud.get('active_pairs', 0)}",
        ]
        y = s["pad"] + 12
        for line in lines:
            _put_text(cv2, out, line, (s["pad"], y), s["small_scale"], COLOR_TEXT, s["thickness"])
            y += int(18 * s["small_scale"] / 0.35)

    # Current behavior state.
    if analysis.states:
        state_text = " / ".join(analysis.states[:3])
        _draw_label(cv2, out, f"STATE: {state_text}", s["w"] // 2 - 120, s["pad"] + 18, s["font_scale"], COLOR_TEXT, s["thickness"])

    # Event banner.
    if event:
        banner = event.get("banner") or "LITTERING EVENT CANDIDATE DETECTED"
        ts = float(event.get("timestamp") or analysis.timestamp)
        fr = int(event.get("frame") or analysis.frame_number)
        pid = event.get("person_track_id", "—")
        oid = event.get("object_track_id", "—")
        conf = float(event.get("confidence") or 0.0)
        panel_h = int(92 * s["font_scale"] / 0.45)
        y0 = max(0, s["h"] - panel_h - 34)
        cv2.rectangle(out, (0, y0), (s["w"], s["h"] - 28), (0, 0, 0), -1)
        cv2.rectangle(out, (0, y0), (s["w"], y0 + 4), COLOR_EVENT, -1)
        _put_text(cv2, out, banner, (s["pad"], y0 + int(28 * s["font_scale"] / 0.45)), s["font_scale"], COLOR_EVENT, s["thickness"])
        _put_text(
            cv2,
            out,
            f"EVENT  t={ts:.2f}s  f={fr}  PERSON #{pid}  BAG #{oid}  CONF {conf:.2f}",
            (s["pad"], y0 + int(58 * s["font_scale"] / 0.45)),
            s["small_scale"],
            COLOR_TEXT,
            s["thickness"],
        )

    # Timestamp/frame.
    _put_text(
        cv2,
        out,
        f"f{analysis.frame_number}  t{analysis.timestamp:.2f}s",
        (s["pad"], s["h"] - s["pad"]),
        s["small_scale"],
        COLOR_TEXT,
        s["thickness"],
    )
    return out
