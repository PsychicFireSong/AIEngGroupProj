# Model Domain Audit

This audit checks how the two-stage YOLO pipeline behaves on images outside the training/validation split.

## Sweep Command

```powershell
python scripts/domain_sweep.py `
  --manifest configs/domain_sweep_manifest.csv `
  --detector weights/defect_detector.pt `
  --severity weights/severity_cls.pt `
  --output output/domain_sweep `
  --thresholds 0.45,0.30,0.20,0.10 `
  --annotate-conf 0.20
```

Generated outputs:

- `output/domain_sweep/domain_sweep_summary.csv`
- `output/domain_sweep/domain_sweep_detections.csv`
- `output/domain_sweep/annotated/`
- `output/domain_sweep/annotated_contact_sheet.jpg`

## Main Finding

The model validates well on the merged dataset split, but the external sweep shows clear domain mismatch.

Training validation summary from the downloaded run:

- Stage 1 detector final aggregate validation: precision `0.879`, recall `0.783`, mAP50 `0.851`, mAP50-95 `0.564`.
- Stage 2 classifier final aggregate validation: top-1 accuracy `0.951`.

External sweep behavior is weaker:

| Domain | Expected | Result at confidence 0.45 | Result at confidence 0.20 |
|---|---|---|---|
| Concrete crack texture | crack | correct crack, 0.653 | correct crack, 0.653 |
| Building wall cracks | crack | no detection | no detection |
| Concrete fracture/spalling column | spalling | no detection | correct spalling, 0.237 |
| Broken concrete with exposed rebar | spalling | no detection | no detection |
| Rust/corrosion texture | corrosion | no detection | no detection |
| Peeling paint wall | paint_degradation | wrong spalling, 0.701 | wrong spalling, 0.701 |
| Bathroom peeling paint | paint_degradation | wrong pothole, 0.521 | wrong pothole, 0.521 |
| User exterior siding/wall damage | paint_degradation or spalling | no detection | wrong pothole, 0.228 |

## Dataset Source Clues

The notebook/config combines datasets with very different visual domains:

- `Finale.yolov11`: large multi-class concrete/tile/defect dataset with 37 classes. Its published Roboflow model metrics are low, which suggests label/domain noise.
- `Internal Wall Defect.yolov11`: internal wall defects only, with classes such as paint drips, peeling paint, pin holes, rough and patchy surface, stain marks, and trowel marks.
- `Concrete defect detection.yolov11`: concrete-specific defects such as crack, efflorescence, exposed reinforcement, rust stain, scaling, and spalling.
- `Corrosion YOLOv8.yolov11` and `metal corrosion.yolov11`: metal corrosion/manufacturing surface defects, not necessarily building facade corrosion.
- `Pothole detection YOLOv8.yolov11` plus Kaggle `archive`: road pothole datasets.

The merge config also maps visually broad labels into only five output classes:

- `paint_degradation` absorbs peeling paint, paint drips, pin holes, rough/patchy surface, stain marks, patches, scratches, and inclusion.
- `spalling` absorbs spalling, exposed reinforcement, rebar, and scaling.
- `corrosion` absorbs rust, rust stains, pitted surface, and rolled-in scale.

This is probably why exterior facade damage gets confused with pothole/spalling textures: the model has learned texture shortcuts rather than stable facility-level defect semantics.

## Practical Interpretation

The current model is most comfortable with:

- close-up concrete crack texture
- some obvious concrete spalling/fracture, but often only at low confidence

The current model is weak on:

- exterior siding/facade damage
- wall cracks photographed from wider building context
- paint degradation on painted walls
- corrosion/rust textures from public-domain examples
- ambiguous mixed materials such as flaking paint over rust

## Recommended Fix

For real facility deployment, do a targeted fine-tuning pass:

1. Add 50-100 real or representative images per class from your target environment.
2. Include exterior wall/facade examples specifically for `paint_degradation` and `spalling`.
3. Add hard negatives: clean walls, windows, plants, shadows, stains, intact concrete, old but non-defective surfaces.
4. Keep pothole data separate or down-weight it if the deployment target is buildings/facilities rather than roads.
5. Split evaluation by domain, not only random image split:
   - concrete close-up
   - internal wall
   - exterior facade
   - road/pavement
   - metal corrosion
6. Add a review mode threshold around `0.20-0.30`, but keep operational default around `0.45` until fine-tuning improves calibration.

The dashboard should keep low-confidence outputs visibly marked as review candidates, not official detections.
