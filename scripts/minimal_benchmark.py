#!/usr/bin/env python3
"""Minimal tracker benchmark — correct API, no frame=, includes current production ByteTrack."""
import os, time, json, sys
from pathlib import Path
import cv2, numpy as np
import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker, OCSORTTracker, BoTSORTTracker

DETECTOR = "/app/models/yolov8n.pt"
VIDEOS = ["/tmp/IMG_5302.MOV", "/tmp/IMG_5287.MOV"]
MAX_FRAMES = 50
OUT = "/tmp/bench_out"
os.makedirs(OUT, exist_ok=True)

det = YOLO(DETECTOR)

def get_dets(frame):
    r = det.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.3)
    if not r or not r[0].boxes:
        return np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    b = r[0].boxes
    xyxy = b.xyxy.cpu().numpy().astype(np.float32)
    tid = b.id.cpu().numpy().astype(int) if b.id is not None else np.arange(len(xyxy))
    cls = b.cls.cpu().numpy().astype(int)
    return xyxy, tid, cls

def tk_stats(td):
    p = td["p"]; o = td["o"]
    # count track switches: new id appearing after 0 or different id
    def switches(arr):
        s=0; prev=0
        for v in arr:
            if v!=prev: s+=1; prev=v
        return s
    return {
        "frames": td["frames"],
        "p_cpu": round(td["p_t"],3), "o_cpu": round(td["o_t"],3),
        "total_cpu": round(td["p_t"]+td["o_t"],3),
        "avg_p": round(np.mean(p),2) if p else 0, "avg_o": round(np.mean(o),2) if o else 0,
        "p_switches": switches(p), "o_switches": switches(o),
        "max_p": max(p) if p else 0, "max_o": max(o) if o else 0,
    }

results = {}
for vpath in VIDEOS:
    name = Path(vpath).stem
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"SKIP {name}: cannot open"); continue
    fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"=== {name} {total}f {fps}fps {w}x{h} ===")

    trackers = {n: {"t": ByteTrackTracker() if n=="rf_bytetrack" else (OCSORTTracker() if n=="ocsort" else BoTSORTTracker()),
                    "p": [], "o": [], "p_t":0.0, "o_t":0.0, "frames":0} for n in ["rf_bytetrack","ocsort","botsort"]}

    frame_idx = 0; t0 = time.time()
    while frame_idx < min(MAX_FRAMES, total):
        ok, frame = cap.read()
        if not ok: break
        if max(w,h) > 720:
            frame = cv2.resize(frame, (640, int(640*h/w)))
        xyxy, tid, cls = get_dets(frame)
        if len(xyxy)==0:
            p_sv = sv.Detections(xyxy=np.zeros((0,4)))
            o_sv = sv.Detections(xyxy=np.zeros((0,4)))
        else:
            is_p = (cls==0)
            p_xy, o_xy = xyxy[is_p], xyxy[~is_p]
            p_t, o_t = tid[is_p], tid[~is_p]
            p_sv = sv.Detections(xyxy=p_xy, tracker_id=p_t) if len(p_xy) else sv.Detections(xyxy=np.zeros((0,4)))
            o_sv = sv.Detections(xyxy=o_xy, tracker_id=o_t) if len(o_xy) else sv.Detections(xyxy=np.zeros((0,4)))

        for td in trackers.values():
            t0t=time.time()
            rp = td["t"].update(p_sv) if len(p_sv.xyxy)>0 else td["t"].update(sv.Detections(xyxy=np.zeros((0,4))))
            td["p_t"]+=time.time()-t0t
            t0t=time.time()
            ro = td["t"].update(o_sv) if len(o_sv.xyxy)>0 else td["t"].update(sv.Detections(xyxy=np.zeros((0,4))))
            td["o_t"]+=time.time()-t0t
            td["p"].append(int(len(rp.tracker_id))) if hasattr(rp,'tracker_id') and rp.tracker_id is not None else td["p"].append(0)
            td["o"].append(int(len(ro.tracker_id))) if hasattr(ro,'tracker_id') and ro.tracker_id is not None else td["o"].append(0)
            td["frames"]+=1
        frame_idx+=1
    cap.release()
    elapsed=time.time()-t0

    res = {"rf_bytetrack": tk_stats(trackers["rf_bytetrack"]),
           "ocsort": tk_stats(trackers["ocsort"]),
           "botsort": tk_stats(trackers["botsort"])}
    res["wall_sec"]=round(elapsed,2); res["fps"]=round(frame_idx/max(1e-6,elapsed),1)
    results[name]=res
    print(f"  {name}: {frame_idx}f in {elapsed:.1f}s ({res['fps']}fps)")
    print(json.dumps(res,indent=2))

with open(os.path.join(OUT,"bench_results.json"),"w") as f: json.dump(results,f,indent=2)
print("\nSaved:", os.path.join(OUT,"bench_results.json"))
print(json.dumps(results,indent=2))
