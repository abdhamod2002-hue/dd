#!/usr/bin/env python3
"""PHASE 4 — evidence identity verification for a real analysis job.

Reads the REAL artifacts produced by the production upload path
(backend API + DB + evidence store) and verifies event-to-evidence
identity consistency (spec Parts 8-16). Writes phase4_runs/ package.

Usage:
    python scripts/phase4_verify_evidence.py --job-id 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "phase4.db"
EVIDENCE_STORE = REPO / "backend" / "evidence_store"
OUT_DIR = REPO / "phase4_runs"


def db_rows(query, params=()):
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(query, params).fetchall()]
    finally:
        con.close()


def media_info(path: Path):
    st = path.stat()
    info = {"size_bytes": st.st_size, "mime": None}
    if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        img = cv2.imread(str(path))
        info["mime"] = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        if img is None:
            info["error"] = "unreadable image"
        else:
            h, w = img.shape[:2]
            info["width"], info["height"] = w, h
    elif path.suffix.lower() == ".mp4":
        info["mime"] = "video/mp4"
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 3)
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if info["fps"]:
                info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
        cap.release()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", type=int, required=True)
    args = ap.parse_args()

    job = db_rows("SELECT * FROM video_analysis_jobs WHERE id=?", (args.job_id,))
    assert job, f"job {args.job_id} not found"
    job = job[0]
    events = db_rows("SELECT * FROM events WHERE analysis_job_id=?", (args.job_id,))
    evidences = db_rows(
        "SELECT e.* FROM evidence e JOIN events ev ON e.event_id=ev.id WHERE ev.analysis_job_id=?",
        (args.job_id,),
    )
    print(f"job {args.job_id}: status={job['status']} events={len(events)} evidence_rows={len(evidences)}")
    print(json.dumps({k: job[k] for k in (
        "status", "processed_frames", "processing_fps", "events_count",
        "persons_detected", "objects_detected", "analyzed_video_path")}, indent=1))

    manifest = json.loads(job["manifest_json"]) if job.get("manifest_json") else {}
    confirmed = manifest.get("confirmed_events") or {}

    OUT_DIR.mkdir(exist_ok=True)
    report = {"job_id": args.job_id, "events": []}
    checks_failed = []

    for ev in events:
        eid = ev["id"]
        ev_rec = {
            "event_id": eid,
            "event_actor_person_track_id": ev.get("event_actor_person_track_id"),
            "event_actor_person_uid": ev.get("event_actor_person_uid"),
            "event_object_track_id": ev.get("event_object_track_id"),
            "event_object_uid": ev.get("event_object_uid"),
            "person_track_id": ev.get("person_track_id"),
            "object_track_id": ev.get("object_track_id"),
            "confidence": ev.get("confidence"),
        }
        det = confirmed.get(str(eid)) or confirmed.get(eid) or {}
        # the manifest keys events by detector event id; find by any value match
        if not det:
            for k, v in confirmed.items():
                if str(v.get("person_track_id")) == str(ev.get("person_track_id")):
                    det = v
                    ev_rec["detector_event_id"] = k
                    break
        ev_rec["detector_event"] = {
            k: det.get(k) for k in (
                "event_id", "person_track_id", "event_actor_person_track_id",
                "event_actor_person_uid", "event_object_track_id", "event_object_uid",
                "bag_track_id", "bag_uid", "confidence", "frames", "timestamps",
                "details") if k in det
        }
        # identity consistency DB vs detector
        for db_f, det_f in (
            ("event_actor_person_track_id", "event_actor_person_track_id"),
            ("event_actor_person_uid", "event_actor_person_uid"),
            ("event_object_track_id", "event_object_track_id"),
            ("event_object_uid", "event_object_uid"),
        ):
            dv, tv = ev.get(db_f), det.get(det_f)
            if dv is not None and tv is not None and int(dv) != int(tv):
                checks_failed.append(f"event {eid}: DB {db_f}={dv} != detector {det_f}={tv}")
        ev_rec["db_matches_detector"] = not any(
            f"event {eid}:" in c for c in checks_failed)

        # evidence row + artifacts
        erow = next((e for e in evidences if e["event_id"] == eid), None)
        artifacts = {}
        if erow:
            for label, col in (
                ("snapshot", "image_path"), ("person", "person_image_path"),
                ("waste", "waste_image_path"), ("event_clip", "clip_path"),
                ("face", "face_image_path"), ("full_evidence_video", "video_path"),
            ):
                rel = erow.get(col)
                if not rel:
                    artifacts[label] = {"present": False}
                    continue
                p = EVIDENCE_STORE / rel
                if not p.exists() or p.stat().st_size == 0:
                    artifacts[label] = {"present": False, "path": rel, "error": "missing/empty"}
                    checks_failed.append(f"event {eid}: {label} missing/empty ({rel})")
                    continue
                artifacts[label] = {"present": True, "path": str(p), **media_info(p)}
        ev_rec["evidence"] = artifacts

        # metadata.json inside the evidence dir
        meta_p = (EVIDENCE_STORE / str(eid) / "metadata.json")
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            ev_rec["evidence_metadata"] = {
                "snapshot_frame_number": meta.get("snapshot_frame_number"),
                "clip_window": meta.get("clip_window"),
                "face_evidence": meta.get("face_evidence"),
                "files": meta.get("files"),
            }
            d_ev = meta.get("event") or {}
            # evidence metadata event identity vs DB
            for db_f, meta_f in (
                ("event_actor_person_uid", "event_actor_person_uid"),
                ("event_object_uid", "event_object_uid"),
            ):
                dv, mv = ev.get(db_f), d_ev.get(meta_f)
                if dv is not None and mv is not None and int(dv) != int(mv):
                    checks_failed.append(
                        f"event {eid}: DB {db_f}={dv} != evidence metadata {meta_f}={mv}")
        report["events"].append(ev_rec)

    report["checks_failed"] = checks_failed
    out = OUT_DIR / "evidence_verification.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str))
    print(f"\nwritten: {out}")
    if checks_failed:
        print(f"CHECKS FAILED: {len(checks_failed)}")
        for c in checks_failed:
            print(" -", c)
    else:
        print("ALL IDENTITY CHECKS PASSED")


if __name__ == "__main__":
    main()
