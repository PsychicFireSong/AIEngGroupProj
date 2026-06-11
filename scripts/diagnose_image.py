"""
Diagnose a specific image: run at conf=0.05 and report ALL predictions.
This reveals whether it is a true feature miss or a calibration miss.
"""
from pathlib import Path
import sys, cv2, numpy as np

DETECTOR_PT = Path(r"C:\Users\User\AIEngGroupProj\weights\candidates\current\defect_detector_hn_weak_candidate.pt")
CLASS_NAMES  = ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path:
    print("Usage: python diagnose_image.py <image_path>")
    sys.exit(1)

from ultralytics import YOLO
print(f"Loading model from {DETECTOR_PT}...")
det = YOLO(str(DETECTOR_PT))
print("Done.\n")

img = cv2.imread(img_path)
if img is None:
    print(f"Could not read image: {img_path}"); sys.exit(1)
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

for conf_floor in [0.05, 0.10, 0.20, 0.30]:
    results = det.predict(source=img, conf=conf_floor, iou=0.45,
                          imgsz=640, max_det=300, device="0", verbose=False)
    preds = []
    if results:
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            xyxy   = boxes.xyxy.cpu().numpy()
            confs  = boxes.conf.cpu().numpy()
            clsids = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                preds.append({
                    "cls": CLASS_NAMES[int(clsids[i])],
                    "conf": float(confs[i]),
                    "box": xyxy[i].tolist()
                })
    print(f"\n--- conf_floor={conf_floor} → {len(preds)} predictions ---")
    for p in preds:
        x1,y1,x2,y2 = [int(v) for v in p["box"]]
        bw = x2-x1; bh = y2-y1
        print(f"  {p['cls']:22s}  conf={p['conf']:.3f}  box=[{x1},{y1},{x2},{y2}]  size={bw}x{bh}px")

# Also try at scale 1280
print("\n--- imgsz=1280 at conf_floor=0.05 ---")
results = det.predict(source=img, conf=0.05, iou=0.45,
                      imgsz=1280, max_det=300, device="0", verbose=False)
preds = []
if results:
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is not None and len(boxes):
        xyxy   = boxes.xyxy.cpu().numpy()
        confs  = boxes.conf.cpu().numpy()
        clsids = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            preds.append({
                "cls": CLASS_NAMES[int(clsids[i])],
                "conf": float(confs[i]),
                "box": xyxy[i].tolist()
            })
print(f"  {len(preds)} predictions at 1280px")
for p in preds:
    x1,y1,x2,y2 = [int(v) for v in p["box"]]
    print(f"  {p['cls']:22s}  conf={p['conf']:.3f}  box=[{x1},{y1},{x2},{y2}]")
