"""Root-cause regression tests for the undefined-local failure class.

Production failure (IMG_5291.MOV, analysis job 43)::

    NameError: name 'sep_floor' is not defined
      File "littering_event_detector.py", line 2904, in _rejection_reason
          and mem.max_post_release_norm_distance >= max(sep_floor, 0.25)

Root cause (PROVEN BY CODE — the container log held the real traceback):

``sep_floor`` is a *local* of ``_rejection_reason``. A refactor moved the
threshold assignment BELOW its first use inside the same function. CPython
compiles such a name as ``LOAD_FAST`` (no compile-time error) and only raises
``NameError`` at runtime, on the single data path that actually evaluates the
guarded expression — ``mem.ground_evidence_frames > 0`` short-circuits the
``and``. That is why two other runs of the *same video* (jobs 42 and 44)
completed cleanly while job 43 died at frame 870.

The class is therefore: *a local variable read on an execution path where it
was not guaranteed to have been assigned*. These tests pin three things:

1. the exact original mis-ordering is detectable (the checker works);
2. no production module currently contains the class (the guard holds);
3. the threshold lifecycle (source -> init -> calibration -> per-pair -> use
   -> reset) can never leave a threshold undefined or stale.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from littering_event_detector import (
    DetectorBag,
    DetectorPerson,
    EventDetectorConfig,
    LitteringEventDetector,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _stmt_reads(stmt: ast.AST):
    """(name, line, col) for every Load read in a statement, excluding names
    bound inside the same statement (comprehension / walrus / lambda targets).
    """
    from phase_rootcause.check_definite_assignment import _read_names

    return _read_names(stmt)

# Modules whose runtime state drives the temporal FSM / pipeline. A
# use-before-assignment in any of them can reproduce the job-43 failure.
GUARDED_MODULES = [
    "littering_event_detector.py",
    "adaptive_tuner.py",
    "inference/pipeline.py",
    "inference/detection/yolo_detector.py",
    "inference/tracking/bytetrack_tracker.py",
    "inference/tracking/object_identity.py",
    "inference/tracking/person_identity.py",
    "inference/pose/movenet_pose.py",
    "backend/routers/analysis.py",
]


# --------------------------------------------------------------------------- #
# 1. The checker detects the ORIGINAL mis-ordering (proof the tool works)
# --------------------------------------------------------------------------- #

def _detector_source() -> str:
    return (REPO_ROOT / "littering_event_detector.py").read_text(encoding="utf-8-sig")


def _reconstruct_job43_buggy_ordering() -> str:
    """Rebuild the deployed code that crashed: sep_floor used before assigned.

    The shipped fix derives the separation bundle through _separation_gate();
    this helper undoes that and restores the historic inline ordering with the
    floor assigned *after* the expression that first reads it.
    """
    src = _detector_source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_rejection_reason")
    lines = src.splitlines(keepends=True)
    call_line = next(
        i for i, ln in enumerate(lines) if "self._separation_gate(mem)" in ln and i >= fn.lineno
    )
    # Replace the single helper call with the historic inline block, placing
    # the floor assignment AFTER the if-block that first reads it.
    inline = (
        "        sep_ticks_needed = max(2, int(cfg.feet_release_frames))\n"
        "        if (\n"
        "            mem.ground_evidence_frames > 0\n"
        "            and mem.max_post_release_norm_distance >= max(sep_floor, 0.25)\n"
        "        ):\n"
        "            sep_ticks_needed = 1\n"
        "        sep_floor = max(cfg.release_distance_floor, 0.08)\n"
        "        physically_separated = (\n"
        "            mem.max_post_release_norm_distance >= sep_floor\n"
        "            and mem.separated_frames >= sep_ticks_needed\n"
        "        )\n"
    )
    return "".join(lines[:call_line] + [inline] + lines[call_line + 1 :])


def test_checker_detects_original_job43_misordering():
    """The static checker must flag the exact code that crashed job 43."""
    from phase_rootcause.check_definite_assignment import analyze_source

    buggy = _reconstruct_job43_buggy_ordering()
    violations = analyze_source(buggy, filename="littering_event_detector.py")
    sep_floor_hits = [v for v in violations if v.name == "sep_floor"]
    assert sep_floor_hits, "checker must detect the sep_floor use-before-assignment"
    hit = sep_floor_hits[0]
    assert hit.function == "_rejection_reason"
    assert "sep_floor" in hit.detail


def test_checker_passes_on_current_detector():
    """The repaired detector must be free of the class."""
    from phase_rootcause.check_definite_assignment import analyze_source

    violations = analyze_source(_detector_source(), filename="littering_event_detector.py")
    assert violations == [], [v.as_dict() for v in violations]


# --------------------------------------------------------------------------- #
# 2. The class guard over every FSM-driving module
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rel", GUARDED_MODULES)
def test_no_use_before_assignment_in_pipeline_module(rel):
    from phase_rootcause.check_definite_assignment import analyze_file

    path = REPO_ROOT / rel
    if not path.is_file():
        pytest.skip(f"{rel} not present")
    violations = analyze_file(str(path))
    assert violations == [], [v.as_dict() for v in violations]


def test_sep_floor_ordering_is_structural():
    """The floor must be derived before anything reads it — by construction.

    This is the structural half of the fix: instead of trusting that the inline
    ordering stays correct, the confirmation gate reads the floor from
    ``_separation_gate()``, which computes it first.
    """
    tree = ast.parse(_detector_source())

    def _fn(name):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    gate = _fn("_separation_gate")
    # The gate must be free of the class AND must bind sep_floor before any
    # statement that reads it — in SOURCE ORDER, not just in some branch.
    from phase_rootcause.check_definite_assignment import _FunctionAnalyzer

    assert _FunctionAnalyzer(gate, "littering_event_detector.py").analyze() == []
    # Only LOCALS can be the failure class; builtins and free names (globals)
    # are excluded, exactly as the real checker does.
    import builtins

    analyzer = _FunctionAnalyzer(gate, "littering_event_detector.py")
    local_names = analyzer._all_assigned
    bound: set = {a.arg for a in gate.args.args} | set(dir(builtins))
    read_before_bind = None
    for stmt in gate.body:
        for nm, _ln, _col in _stmt_reads(stmt):
            if nm not in local_names:  # builtin / global / module name
                continue
            if nm not in bound:
                read_before_bind = nm
                break
        if read_before_bind:
            break
        bound |= {t.id for t in ast.walk(stmt) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
    assert read_before_bind is None, f"gate reads {read_before_bind!r} before binding it"
    assert "sep_floor" in bound, "the separation gate must bind sep_floor"

    # And the confirmation gate must route through the helper, not re-derive.
    rejection = _fn("_rejection_reason")
    calls: set = set()
    for n in ast.walk(rejection):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute):
                calls.add(n.func.attr)
            elif isinstance(n.func, ast.Name):
                calls.add(n.func.id)
    assert "_separation_gate" in calls
    # No inline max(cfg.release_distance_floor, ...) left in the gate.
    src_lines = _detector_source().splitlines()
    body = "\n".join(src_lines[rejection.lineno : rejection.end_lineno])
    assert "max(cfg.release_distance_floor" not in body, (
        "rejection gate must not re-derive the floor inline"
    )


# --------------------------------------------------------------------------- #
# 3. Threshold lifecycle: never undefined, never stale
# --------------------------------------------------------------------------- #

def _cfg(**kw) -> EventDetectorConfig:
    base = dict(
        auto_calibrate=False,
        pose_association_enabled=False,
        min_carried_frames=3,
        min_stationary_frames=3,
        min_departed_frames=2,
        min_abandonment_frames=3,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=20,
        min_event_confidence=0.0,
        carry_motion_norm_px=2.0,
    )
    base.update(kw)
    return EventDetectorConfig(**base)


def test_separation_floor_respects_config_and_hard_minimum():
    cfg = _cfg(release_distance_floor=0.30)
    # Config wins when it is above the hard minimum.
    assert LitteringEventDetector._separation_floor(cfg, 0.08) == pytest.approx(0.30)
    # Hard minimum wins when the config is below it (never silently lowered).
    cfg2 = _cfg(release_distance_floor=0.01)
    assert LitteringEventDetector._separation_floor(cfg2, 0.08) == pytest.approx(0.08)
    # The hard minimum itself is honoured per call site.
    assert LitteringEventDetector._separation_floor(cfg2, 0.12) == pytest.approx(0.12)


def test_separation_gate_always_returns_defined_triple():
    from littering_event_detector import _PairMemory

    det = LitteringEventDetector(_cfg())
    mem = _PairMemory(person_id=1, bag_id=60001, person_uid=1, bag_uid=100001)
    # A fresh pair with no release at all must still yield defined values.
    floor, ticks, separated = det._separation_gate(mem)
    assert floor is not None and ticks is not None and separated is not None
    assert floor > 0.0
    assert isinstance(ticks, int) and ticks >= 1
    assert isinstance(separated, bool)
    # A pair WITH ground evidence must also be defined (the job-43 path).
    mem.ground_evidence_frames = 5
    mem.max_post_release_norm_distance = 0.5
    floor2, ticks2, separated2 = det._separation_gate(mem)
    assert floor2 > 0.0 and ticks2 >= 1 and isinstance(separated2, bool)


def test_reset_restores_calibrated_thresholds():
    """reset() must not leak one video's calibrated thresholds into the next.

    _calibrate() writes carry_motion_norm_px / departure_motion_ratio into the
    shared config object; a reused detector would otherwise start from a
    stale, other-video scene scale (same family as an undefined threshold).
    """
    det = LitteringEventDetector(_cfg(auto_calibrate=True, carry_motion_norm_px=4.0))
    base = det.config.carry_motion_norm_px
    det.config.carry_motion_norm_px = 41.0
    det.config.departure_motion_ratio = 0.77
    det.reset()
    assert det.config.carry_motion_norm_px == base
    assert det.config.departure_motion_ratio == det._calibration_mutables["departure_motion_ratio"]
    assert det._calibrated is False


# --------------------------------------------------------------------------- #
# 4. Fault injection: an undefined-local crash must surface, not be hidden
# --------------------------------------------------------------------------- #

def _person(pid=1, x=200, y=200):
    return DetectorPerson(track_id=pid, bbox=(x, y, x + 60, y + 160), confidence=1.0)


def _bag(bid=60001, x=210, y=280):
    return DetectorBag(
        track_id=bid, bbox=(x, y, x + 30, y + 30), confidence=0.9,
        class_name="trash_bag", source="yolo", yolo_confirmed=True,
    )


def test_undefined_local_in_fsm_propagates_and_reports_true_stage(monkeypatch):
    """Nothing swallows the crash; the stage tracker attributes the TRUE stage.

    This is deliberately NOT a try/except that hides the failure (spec section
    3): the NameError is allowed to propagate, and the *reporting* layer is
    what turns it into an accurate, forensic-backed failure.
    """
    from backend.routers.analysis import PipelineStageTracker

    # 4a. The detector does not hide the error: drive a pair into the
    #     confirmation gate, which is exactly where job 43 raised.
    from littering_event_detector import _PairMemory

    det = LitteringEventDetector(_cfg())

    def _boom(self, mem, evidence):
        raise NameError("name 'sep_floor' is not defined")

    monkeypatch.setattr(LitteringEventDetector, "_rejection_reason", _boom)
    mem = _PairMemory(person_id=1, bag_id=60001, person_uid=1, bag_uid=100001)
    mem.last_frame, mem.last_timestamp = 870, 29.0
    with pytest.raises(NameError):
        det._evaluate_confirmation(mem)

    # 4b. The stage tracker reports the real stage, keeps completed stages,
    #     and skips later ones.
    tracker = PipelineStageTracker()
    tracker.complete_stage("video_input", "Decoded 942 frames")
    for sid in (
        "person_detection", "waste_detection", "tracking", "person_identity",
        "object_identity", "pose_estimation", "ownership_association",
        "temporal_event_detection",
    ):
        tracker.start_stage(sid, "active")
        tracker.complete_stage(sid, "ok")
    # The crash lands mid temporal_event_detection (it re-runs per frame).
    tracker.start_stage("temporal_event_detection", "FSM tick")
    tracker.fail_stage("temporal_event_detection", "name 'sep_floor' is not defined", frame_idx=870)

    by_id = {s["id"]: s for s in tracker.to_list()}
    assert by_id["video_input"]["status"] == "COMPLETED"
    assert by_id["temporal_event_detection"]["status"] == "FAILED"
    assert tracker.failed_stage_id == "temporal_event_detection"
    assert by_id["evidence_assembly"]["status"] == "SKIPPED"
    assert by_id["api_dashboard"]["status"] == "SKIPPED"
    assert "TEMPORAL EVENT DETECTION" in tracker.current_stage_display()
    assert tracker.current_stage_display().startswith("Stage 9/12")

    tel = tracker.build_telemetry(
        job_id=43, status="failed", processed_frames=870, total_frames=942,
        error_message="name 'sep_floor' is not defined",
    )
    assert tel["pipeline_telemetry"]["current_stage_id"] == "temporal_event_detection"
    assert tel["pipeline_telemetry"]["current_step"] == 9
