"""
conf_analysis.py  —  Answer two questions:
  1. Is the model still worth improving via training?
  2. Are confidence scores good enough?

Method:
  - Run detector at very low conf floor (0.05) on all val images
  - For each GT box, find the best-matching prediction (any conf)
  - Record: was a match found? at what conf? what class?
  - This separates:
      A) "Model generates prediction but below threshold" (calibration issue → fix with threshold)
      B) "Model generates NO prediction at all" (feature miss → only training can fix)
  - Also report FP rate at current thresholds vs lower thresholds
"""
from __future__ import annotations
from pathlib import Path
import csv, collections
import numpy as np
import cv2

DETECTOR_PT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
VAL_IMAGES  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
VAL_LABELS  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\labels\val")
CLASS_NAMES = ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']

# Thresholds to evaluate FP/TP at
EVAL_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.38, 0.45]
IOU_MATCH_THR   = 0.50   # IoU >= 0.5 = true positive


def _load_gt(lp: Path, w: int, h: int):
    boxes = []
    if not lp.exists(): return boxes
    for line in lp.read_text().strip().splitlines():
        p = line.strip().split()
        if len(p) < 5: continue
        cls = int(p[0])
        cx,cy,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        boxes.append((cls, (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return boxes

def _iou(a,b):
    ix1,iy1 = max(a[0],b[0]),max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if not inter: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0


def main():
    import ctypes
    try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|0x00000001)
    except: pass

    from ultralytics import YOLO
    print("Loading detector...")
    det = YOLO(str(DETECTOR_PT))
    print("  Done.\n")

    # ── Collect ALL predictions at conf=0.05 (near-zero floor) ───────────────
    # For each GT box: record best-matching prediction conf, class match, iou
    # For each prediction: record if it matched a GT box (TP) or not (FP)

    per_class_gt_info: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    per_class_preds:   dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}

    val_img_paths = sorted(VAL_IMAGES.glob("*.jpg")) + sorted(VAL_IMAGES.glob("*.png"))
    print(f"Running at conf=0.05 on {len(val_img_paths)} val images...")

    for img_path in val_img_paths:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        lp   = VAL_LABELS / (img_path.stem + ".txt")
        gt   = _load_gt(lp, w, h)

        results = det.predict(source=img, conf=0.05, iou=0.45,
                              imgsz=640, max_det=300, device="0", verbose=False)
        preds = []
        if results:
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            if boxes is not None and len(boxes):
                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clsids= boxes.cls.cpu().numpy().astype(int)
                for i in range(len(xyxy)):
                    x1,y1,x2,y2 = xyxy[i].tolist()
                    preds.append({'cls': int(clsids[i]), 'conf': float(confs[i]),
                                  'box': (x1,y1,x2,y2)})

        # For each GT box, find best-matching prediction
        for (cls, gx1,gy1,gx2,gy2) in gt:
            if cls >= len(CLASS_NAMES): continue
            cls_name = CLASS_NAMES[cls]
            best_conf   = 0.0
            best_iou    = 0.0
            best_cls_match = False
            for p in preds:
                iou_val = _iou((gx1,gy1,gx2,gy2), p['box'])
                if iou_val > best_iou:
                    best_iou    = iou_val
                    best_conf   = p['conf']
                    best_cls_match = (p['cls'] == cls)
            per_class_gt_info[cls_name].append({
                'best_pred_conf': best_conf,
                'best_iou':       best_iou,
                'cls_match':      best_cls_match,
                'any_pred':       best_iou >= 0.01,  # any spatial overlap
                'tp_at_05':       best_iou >= IOU_MATCH_THR and best_cls_match,
            })

        # For each prediction, record if it's a TP at IoU>=0.5
        for p in preds:
            cls_name = CLASS_NAMES[p['cls']] if p['cls'] < len(CLASS_NAMES) else 'other'
            if cls_name not in per_class_preds: continue
            is_tp = False
            for (cls, gx1,gy1,gx2,gy2) in gt:
                if cls != p['cls']: continue
                if _iou((gx1,gy1,gx2,gy2), p['box']) >= IOU_MATCH_THR:
                    is_tp = True; break
            per_class_preds[cls_name].append({'conf': p['conf'], 'is_tp': is_tp})

    # ── REPORT 1: Is model generating predictions vs true miss? ──────────────
    print("\n" + "="*70)
    print("QUESTION 1: Is the model generating predictions at all?")
    print("  (For each GT box: did the model produce ANY overlapping prediction?)")
    print("="*70)
    print(f"\n{'Class':<22} {'GT boxes':>9} {'Any pred':>10} {'Cls match':>11} {'True miss':>11}")
    print("-"*66)
    for cls_name in CLASS_NAMES:
        entries = per_class_gt_info[cls_name]
        if not entries:
            print(f"  {cls_name:<20} {'N/A':>9}"); continue
        n = len(entries)
        any_pred  = sum(1 for e in entries if e['any_pred'])
        cls_match = sum(1 for e in entries if e['cls_match'] and e['best_iou']>=0.3)
        true_miss = n - any_pred
        print(f"  {cls_name:<20} {n:>9} {any_pred:>9} ({any_pred/n*100:.0f}%)"
              f" {cls_match:>9} ({cls_match/n*100:.0f}%)"
              f" {true_miss:>9} ({true_miss/n*100:.0f}%)")

    # ── REPORT 2: Confidence distribution of GT-matched predictions ───────────
    print("\n" + "="*70)
    print("QUESTION 2: Where do TP predictions land on the confidence scale?")
    print("  (Distribution of best-match conf for GT boxes where model DID respond)")
    print("="*70)
    BINS = [(0.05,0.20,'0.05-0.20'), (0.20,0.30,'0.20-0.30'),
            (0.30,0.38,'0.30-0.38'), (0.38,0.45,'0.38-0.45'),
            (0.45,0.60,'0.45-0.60'), (0.60,0.75,'0.60-0.75'),
            (0.75,1.01,'0.75-1.00')]
    for cls_name in CLASS_NAMES:
        entries = [e for e in per_class_gt_info[cls_name] if e['any_pred'] and e['cls_match']]
        if not entries:
            print(f"\n  {cls_name}: no matched predictions"); continue
        confs = [e['best_pred_conf'] for e in entries]
        print(f"\n  {cls_name} (n={len(entries)} GT boxes with model response):")
        for lo, hi, label in BINS:
            count = sum(1 for c in confs if lo <= c < hi)
            bar   = '█' * count
            print(f"    conf {label}: {count:>3}  {bar}")
        print(f"    median conf={np.median(confs):.3f}  mean={np.mean(confs):.3f}  "
              f"min={np.min(confs):.3f}  max={np.max(confs):.3f}")

    # ── REPORT 3: TP vs FP at each confidence threshold ──────────────────────
    print("\n" + "="*70)
    print("QUESTION 3: TP / FP / precision at different confidence thresholds")
    print("="*70)
    for cls_name in CLASS_NAMES:
        preds = per_class_preds[cls_name]
        gt_n  = len(per_class_gt_info[cls_name])
        if not preds:
            print(f"\n  {cls_name}: no predictions above floor"); continue
        print(f"\n  {cls_name}  (GT boxes: {gt_n})")
        print(f"    {'Thresh':>7}  {'Preds':>6}  {'TP':>5}  {'FP':>5}  {'Recall':>8}  {'Precision':>10}")
        for thr in EVAL_THRESHOLDS:
            above = [p for p in preds if p['conf'] >= thr]
            tp = sum(1 for p in above if p['is_tp'])
            fp = len(above) - tp
            recall    = tp / gt_n if gt_n else 0.0
            precision = tp / len(above) if above else 0.0
            marker = ' <-- current' if thr in (0.45,) else (
                     ' <-- corrosion floor' if (thr==0.25 and cls_name=='corrosion') else
                     ' <-- paint_deg floor' if (thr==0.30 and cls_name=='paint_degradation') else '')
            print(f"    {thr:>7.2f}  {len(above):>6}  {tp:>5}  {fp:>5}  "
                  f"{recall:>7.1%}  {precision:>9.1%}{marker}")

    # ── REPORT 4: Training verdict ────────────────────────────────────────────
    print("\n" + "="*70)
    print("TRAINING VERDICT")
    print("="*70)
    for cls_name in CLASS_NAMES:
        entries = per_class_gt_info[cls_name]
        if not entries: continue
        n = len(entries)
        true_miss = sum(1 for e in entries if not e['any_pred'])
        low_conf  = sum(1 for e in entries if e['any_pred'] and e['cls_match']
                        and e['best_pred_conf'] < 0.45)
        tp_ok     = sum(1 for e in entries if e['tp_at_05'])
        print(f"\n  {cls_name}:")
        print(f"    True feature miss (model silent):  {true_miss}/{n} = {true_miss/n*100:.0f}%")
        print(f"    Calibration miss (conf < 0.45):    {low_conf}/{n} = {low_conf/n*100:.0f}%")
        print(f"    True TP @ IoU>=0.5 & conf>=0.05:   {tp_ok}/{n} = {tp_ok/n*100:.0f}%")
        if true_miss/n > 0.3:
            print(f"    => NEEDS TRAINING (>30% true misses — model doesn't see this defect)")
        elif low_conf/n > 0.2:
            print(f"    => CALIBRATION ISSUE (>20% detections below 0.45 — threshold fix enough)")
        else:
            print(f"    => OK — confidence and recall are adequate for this class")


if __name__ == "__main__":
    main()
