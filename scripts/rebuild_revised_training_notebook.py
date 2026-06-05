from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MATERIALIZED_FILES = [
    "configs/merge_config.yaml",
    "configs/production_eval_manifest.csv",
    "scripts/merge_datasets.py",
    "scripts/extract_severity_crops.py",
    "scripts/domain_sweep.py",
    "scripts/auto_collect_domain_sources.py",
    "scripts/audit_label_mapping_coverage.py",
    "scripts/bootstrap_dataset_archive.py",
    "scripts/bootstrap_merged_dataset_archive_from_sources.py",
    "scripts/build_balanced_detection_dataset.py",
    "scripts/build_anchor_balanced_dataset.py",
    "scripts/build_robust_augmented_dataset.py",
    "scripts/audit_defect_feature_coverage.py",
    "scripts/wider_production_sweep.py",
    "scripts/revised_production_gate.py",
    "apps/inference_api.py",
]


def source_lines(text: str) -> list[str]:
    return (text.strip() + "\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


def architecture_diagram_png_data_uri() -> str:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 2200, 1480
    image = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    title_font = font(48, True)
    subtitle_font = font(25)
    lane_font = font(26, True)
    box_title_font = font(24, True)
    text_font = font(20)
    small_font = font(17)
    guard_font = font(18, True)

    colors = {
        "ink": "#172033",
        "muted": "#526071",
        "text": "#3c4858",
        "guard": "#7b341e",
        "line": "#4b5b70",
        "source": ("#eef7ff", "#8fc5ee"),
        "quality": ("#fff7ed", "#f5b971"),
        "dataset": ("#effcf4", "#8dd7a8"),
        "train": ("#f2f0ff", "#b3a7f7"),
        "eval": ("#fff1f2", "#f0a3ad"),
        "deploy": ("#edfafa", "#76c7cf"),
        "feedback": ("#f7f4ea", "#d3bd70"),
    }

    def wrap_text(text: str, max_width: int, draw_font) -> list[str]:
        words = text.split()
        lines: list[str] = []
        line = ""
        for word in words:
            trial = word if not line else f"{line} {word}"
            bbox = draw.textbbox((0, 0), trial, font=draw_font)
            if bbox[2] - bbox[0] <= max_width:
                line = trial
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    def rounded_box(x: int, y: int, w: int, h: int, title: str, lines: list[str], kind: str, guard: str = "", note: str = "") -> None:
        fill, outline = colors[kind]
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=outline, width=3)
        draw.text((x + 30, y + 28), title, font=box_title_font, fill=colors["ink"])
        text_y = y + 72
        for line in lines:
            for wrapped in wrap_text(line, w - 60, text_font):
                draw.text((x + 30, text_y), wrapped, font=text_font, fill=colors["text"])
                text_y += 29
        if note:
            text_y += 5
            for wrapped in wrap_text(note, w - 60, small_font):
                draw.text((x + 30, text_y), wrapped, font=small_font, fill=colors["muted"])
                text_y += 24
        if guard:
            guard_y = y + h - 42
            draw.text((x + 30, guard_y), guard, font=guard_font, fill=colors["guard"])

    def arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> None:
        line_color = "#8793a3" if dashed else colors["line"]
        if dashed:
            segments = 18
            for i in range(segments):
                if i % 2 == 0:
                    sx = x1 + (x2 - x1) * i / segments
                    sy = y1 + (y2 - y1) * i / segments
                    ex = x1 + (x2 - x1) * (i + 1) / segments
                    ey = y1 + (y2 - y1) * (i + 1) / segments
                    draw.line((sx, sy, ex, ey), fill=line_color, width=4)
        else:
            draw.line((x1, y1, x2, y2), fill=line_color, width=4)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 16
        p1 = (x2, y2)
        p2 = (x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
        p3 = (x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
        draw.polygon([p1, p2, p3], fill=line_color)

    import math

    draw.text((80, 58), "AIEngGroupProj Two-Stage Defect Detection Architecture", font=title_font, fill=colors["ink"])
    draw.text((80, 118), "Wall/facility research alignment, baseline class protection, robust training, deployment gating, and feedback.", font=subtitle_font, fill=colors["muted"])
    draw.line((80, 170, width - 80, 170), fill="#d9e1ea", width=2)

    col_x = [80, 550, 1040, 1590]
    box_w = [390, 410, 440, 440]
    lane_titles = ["1. Evidence Sources", "2. Data Governance", "3. Robust Dataset Builder", "4. Training and Deployment"]
    for x, title in zip(col_x, lane_titles):
        draw.text((x + 8, 218), title, font=lane_font, fill=colors["ink"])

    rounded_box(col_x[0], 255, box_w[0], 180, "Baseline YOLO Sources", ["7 raw datasets with preserved splits", "crack, spalling, corrosion", "pothole, paint_degradation"], "source")
    rounded_box(col_x[0], 470, box_w[0], 210, "Wall-Maintenance Sources", ["crack, stairstep crack", "mold, water seepage, dampness", "peeling paint and stains"], "source", note="Mapped into the same 5-class contract")
    rounded_box(col_x[0], 715, box_w[0], 180, "Operational Evidence", ["Hard negatives from sweeps", "Previous Drive weights", "User camera and upload cases"], "source")

    rounded_box(col_x[1], 255, box_w[1], 200, "Recovery and Merge", ["Download/cache raw sources", "Synonym mapping to 5 classes", "Reject partial missing-class rebuilds"], "quality", guard="Guard: all classes in train and val")
    rounded_box(col_x[1], 490, box_w[1], 220, "Target Quality Filter", ["Remove unreadable, tiny, blurry images", "Reject invalid boxes or huge boxes", "Reject over-mixed target labels"], "quality", guard="Guard: coverage and keep-rate")
    rounded_box(col_x[1], 745, box_w[1], 205, "Baseline Anchor", ["Baseline data is never optional", "Target data is capped", "Prevents class forgetting"], "quality", guard="Guard: no target-only training")

    rounded_box(col_x[2], 255, box_w[2], 190, "Class and Domain Balancing", ["Oversample weak classes after anchoring", "Round-robin road, wall, facade, corrosion"], "dataset", guard="Guard: max/min class ratio")
    rounded_box(col_x[2], 480, box_w[2], 250, "Defect-Cue Augmentation", ["Low light, overexposure, shadow", "Distance blur and compression", "Occlusion and perspective", "Edge, contrast, grayscale structure"], "dataset", note="Goal: learn geometry and texture cues, not color shortcuts")
    rounded_box(col_x[2], 765, box_w[2], 220, "Scale-Space ROI Crops", ["Close, mid, and context windows", "YOLO boxes transformed into crop space", "Zoomed defect plus real-scene context"], "dataset", guard="Guard: labels preserved")
    rounded_box(col_x[2], 1020, box_w[2], 165, "Feature Coverage Audit", ["Scale, aspect, edge density, texture", "brightness, saturation by class"], "dataset")

    rounded_box(col_x[3], 255, box_w[3], 180, "Stage 1 Curriculum Detector", ["YOLO11m detector at 768 px", "Clean warm-up -> robust fine-tune", "Validation selects candidate"], "train")
    rounded_box(col_x[3], 470, box_w[3], 170, "Stage 2 Severity", ["Detected crop becomes classifier input", "minor / moderate / critical"], "train", guard="Guard: retrain only with coverage")
    rounded_box(col_x[3], 675, box_w[3], 210, "Production Sweep Gate", ["External and cached visual cases", "Original, low light, blur, shadow", "overexposure, occlusion scenarios"], "eval", guard="Gate: 75% overall, 70% per bucket")
    rounded_box(col_x[3], 920, box_w[3], 180, "Dashboard Inference", ["Image, video, camera streams", "Boxes with class + severity", "Visual evidence, history, analytics"], "deploy")

    rounded_box(610, 1065, 360, 170, "Human Review Loop", ["Save misses and false positives", "Feed next collection cycle"], "feedback")

    for y in [345, 575, 805]:
        arrow(col_x[0] + box_w[0], y, col_x[1], y)
    for y in [355, 600, 835]:
        arrow(col_x[1] + box_w[1], y, col_x[2], y)
    for y in [350, 590, 875]:
        arrow(col_x[2] + box_w[2], y, col_x[3], y)
    arrow(col_x[3] + 220, 435, col_x[3] + 220, 470)
    arrow(col_x[3] + 220, 640, col_x[3] + 220, 675)
    arrow(col_x[3] + 220, 885, col_x[3] + 220, 920)
    draw.text((610, 1260), "Reviewed deployment misses are reused as future evidence and hard negatives.", font=small_font, fill=colors["muted"])

    draw.text((80, 1400), "Design principle: paper-style wall defect learning is integrated, while deployment keeps the original five-class industrial taxonomy and blocks unsafe promotion.", font=small_font, fill=colors["muted"])

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


ARCHITECTURE_DIAGRAM_SVG = r'''
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1040" viewBox="0 0 1600 1040" role="img" aria-label="Two-stage YOLO defect detection architecture">
  <defs>
    <style>
      .bg{fill:#f7f9fc}.title{font:700 34px Arial,sans-serif;fill:#172033}.sub{font:400 18px Arial,sans-serif;fill:#526071}
      .lane{font:700 18px Arial,sans-serif;fill:#172033}.h{font:700 17px Arial,sans-serif;fill:#172033}.t{font:400 14px Arial,sans-serif;fill:#3f4c5f}
      .g{font:700 13px Arial,sans-serif;fill:#7b341e}.s{font:400 12px Arial,sans-serif;fill:#5d6b7d}
      .b{fill:#fff;stroke:#cad3df;stroke-width:1.4;rx:12}.src{fill:#eef7ff;stroke:#8fc5ee}.q{fill:#fff7ed;stroke:#f5b971}
      .d{fill:#effcf4;stroke:#8dd7a8}.m{fill:#f2f0ff;stroke:#b3a7f7}.e{fill:#fff1f2;stroke:#f0a3ad}.dep{fill:#edfafa;stroke:#76c7cf}.fb{fill:#f7f4ea;stroke:#d3bd70}
      .a{stroke:#4b5b70;stroke-width:2.2;fill:none;marker-end:url(#ah)}.sa{stroke:#8793a3;stroke-width:1.8;fill:none;stroke-dasharray:7 6;marker-end:url(#ash)}
    </style>
    <marker id="ah" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto"><path d="M0,0 L12,4 L0,8 z" fill="#4b5b70"/></marker>
    <marker id="ash" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto"><path d="M0,0 L12,4 L0,8 z" fill="#8793a3"/></marker>
  </defs>
  <rect class="bg" width="1600" height="1040"/>
  <text class="title" x="64" y="62">AIEngGroupProj Two-Stage Defect Detection Architecture</text>
  <text class="sub" x="64" y="92">Wall/facility research alignment, baseline class protection, robust training, deployment gating, and feedback.</text>

  <text class="lane" x="86" y="150">1. Evidence Sources</text>
  <rect class="b src" x="64" y="170" width="300" height="150"/><text class="h" x="88" y="206">Baseline YOLO Sources</text><text class="t" x="88" y="236">7 raw datasets with preserved splits</text><text class="t" x="88" y="260">crack, spalling, corrosion</text><text class="t" x="88" y="284">pothole, paint_degradation</text>
  <rect class="b src" x="64" y="350" width="300" height="170"/><text class="h" x="88" y="386">Wall-Maintenance Sources</text><text class="t" x="88" y="416">crack, stairstep crack</text><text class="t" x="88" y="440">mold, water seepage, dampness</text><text class="t" x="88" y="464">peeling paint, stains</text><text class="s" x="88" y="494">Mapped into the same 5-class contract</text>
  <rect class="b src" x="64" y="550" width="300" height="140"/><text class="h" x="88" y="586">Operational Evidence</text><text class="t" x="88" y="616">Hard negatives from sweeps</text><text class="t" x="88" y="640">Previous Drive weights</text><text class="t" x="88" y="664">User camera and upload cases</text>

  <text class="lane" x="426" y="150">2. Data Governance</text>
  <rect class="b q" x="404" y="170" width="324" height="170"/><text class="h" x="428" y="206">Recovery and Merge</text><text class="t" x="428" y="236">Download/cache raw sources</text><text class="t" x="428" y="260">Synonym mapping to 5 classes</text><text class="t" x="428" y="284">Reject partial missing-class rebuilds</text><text class="g" x="428" y="316">Guard: all classes in train and val</text>
  <rect class="b q" x="404" y="372" width="324" height="178"/><text class="h" x="428" y="408">Target Quality Filter</text><text class="t" x="428" y="438">Remove unreadable, tiny, blurry images</text><text class="t" x="428" y="462">Reject invalid boxes or huge boxes</text><text class="t" x="428" y="486">Reject over-mixed target labels</text><text class="g" x="428" y="522">Guard: coverage and keep-rate</text>
  <rect class="b q" x="404" y="582" width="324" height="146"/><text class="h" x="428" y="618">Baseline Anchor</text><text class="t" x="428" y="648">Baseline data is never optional</text><text class="t" x="428" y="672">Target data is capped</text><text class="g" x="428" y="704">Guard: no target-only detector training</text>

  <text class="lane" x="788" y="150">3. Robust Dataset Builder</text>
  <rect class="b d" x="768" y="170" width="372" height="164"/><text class="h" x="792" y="206">Class and Domain Balancing</text><text class="t" x="792" y="236">Oversample weak classes after anchoring</text><text class="t" x="792" y="260">Round-robin road, wall, facade, corrosion</text><text class="g" x="792" y="300">Guard: max/min class ratio</text>
  <rect class="b d" x="768" y="366" width="372" height="212"/><text class="h" x="792" y="402">Defect-Cue Augmentation</text><text class="t" x="792" y="432">Low light, overexposure, shadow</text><text class="t" x="792" y="456">Distance blur and compression</text><text class="t" x="792" y="480">Occlusion and perspective</text><text class="t" x="792" y="504">Edge, contrast, grayscale structure</text><text class="s" x="792" y="540">Goal: learn geometry and texture cues, not color shortcuts</text>
  <rect class="b d" x="768" y="610" width="372" height="178"/><text class="h" x="792" y="646">Scale-Space ROI Crops</text><text class="t" x="792" y="676">Close, mid, and context windows</text><text class="t" x="792" y="700">YOLO boxes transformed into crop space</text><text class="t" x="792" y="724">Zoomed defect plus real-scene context</text><text class="g" x="792" y="756">Guard: labels preserved, no pseudo boxes</text>
  <rect class="b d" x="768" y="820" width="372" height="120"/><text class="h" x="792" y="856">Feature Coverage Audit</text><text class="t" x="792" y="886">Scale, aspect, edge density, texture</text><text class="t" x="792" y="910">brightness, saturation by class</text>

  <text class="lane" x="1192" y="150">4. Training and Deployment</text>
  <rect class="b m" x="1180" y="170" width="350" height="154"/><text class="h" x="1204" y="206">Stage 1 Detector</text><text class="t" x="1204" y="236">YOLO11m detector at 768 px</text><text class="t" x="1204" y="260">Five-class defect localization</text><text class="s" x="1204" y="292">Stronger box/class loss emphasis</text>
  <rect class="b m" x="1180" y="356" width="350" height="142"/><text class="h" x="1204" y="392">Stage 2 Severity</text><text class="t" x="1204" y="422">Detected crop becomes classifier input</text><text class="t" x="1204" y="446">minor / moderate / critical</text><text class="g" x="1204" y="474">Guard: retrain only with coverage</text>
  <rect class="b e" x="1180" y="530" width="350" height="164"/><text class="h" x="1204" y="566">Production Sweep Gate</text><text class="t" x="1204" y="596">External and cached visual cases</text><text class="t" x="1204" y="620">Original, low light, blur, shadow</text><text class="t" x="1204" y="644">overexposure, occlusion scenarios</text><text class="g" x="1204" y="672">Gate: 75% overall, 70% per bucket</text>
  <rect class="b dep" x="1180" y="728" width="350" height="150"/><text class="h" x="1204" y="764">Dashboard Inference</text><text class="t" x="1204" y="794">Image, video, camera streams</text><text class="t" x="1204" y="818">Boxes with class + severity</text><text class="t" x="1204" y="842">Visual evidence, history, analytics</text>

  <rect class="b fb" x="456" y="842" width="260" height="116"/><text class="h" x="480" y="878">Human Review Loop</text><text class="t" x="480" y="908">Save misses and false positives</text><text class="t" x="480" y="932">Feed next collection cycle</text>

  <path class="a" d="M364 245 H404"/><path class="a" d="M364 435 H404"/><path class="a" d="M364 620 H404"/>
  <path class="a" d="M728 255 H768"/><path class="a" d="M728 460 H768"/><path class="a" d="M728 655 H768"/>
  <path class="a" d="M1140 252 H1180"/><path class="a" d="M1140 472 H1180"/><path class="a" d="M1140 700 H1180"/>
  <path class="a" d="M1355 324 V356"/><path class="a" d="M1355 498 V530"/><path class="a" d="M1355 694 V728"/>
  <path class="sa" d="M1180 810 C940 816,820 946,716 900"/><path class="sa" d="M456 900 C230 884,214 718,214 690"/>
  <text class="s" x="64" y="1000">Design principle: paper-style wall defect learning is integrated, while deployment keeps the original five-class industrial taxonomy and blocks unsafe promotion.</text>
</svg>
'''


def materialize_project_code() -> str:
    contents = {path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in MATERIALIZED_FILES if (REPO_ROOT / path).exists()}
    return (
        "PROJECT_FILE_CONTENTS = "
        + repr(contents)
        + r'''

import py_compile

changed = []
for relative_path, content in PROJECT_FILE_CONTENTS.items():
    path = REPO_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
        changed.append(relative_path)

if changed:
    print("Materialized updated files:")
    for item in changed:
        print(f"- {item}")
else:
    print("All required project files already match this notebook.")

for relative_path in PROJECT_FILE_CONTENTS:
    if relative_path.endswith(".py"):
        py_compile.compile(str(REPO_ROOT / relative_path), doraise=True)

print("Python scripts compiled successfully.")
mark_done("materialized_revised_training_files", {"files": sorted(PROJECT_FILE_CONTENTS)})
'''
    )


SETUP_CODE = r'''
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

try:
    from google.colab import drive
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
except Exception as exc:
    print(f"Drive mount skipped or unavailable: {exc}")

GITHUB_REPO_URL = globals().get("GITHUB_REPO_URL", "https://github.com/PsychicFireSong/AIEngGroupProj.git")
REPO_ROOT_OVERRIDE = globals().get("REPO_ROOT_OVERRIDE", "")

SAFE_RECOVERY_MODE = globals().get("SAFE_RECOVERY_MODE", True)
REQUIRE_FULL_REVISED_RUN = globals().get("REQUIRE_FULL_REVISED_RUN", True)
MIRROR_FULL_DATASET_FOLDERS = globals().get("MIRROR_FULL_DATASET_FOLDERS", False)
PREFER_SOURCE_ARCHIVE_BOOTSTRAP = globals().get("PREFER_SOURCE_ARCHIVE_BOOTSTRAP", True)
ALLOW_SLOW_DRIVE_FOLDER_RESTORE = globals().get("ALLOW_SLOW_DRIVE_FOLDER_RESTORE", False)
FORCE_MATERIALIZE = globals().get("FORCE_MATERIALIZE", False)
FORCE_DOMAIN_COLLECTION = globals().get("FORCE_DOMAIN_COLLECTION", REQUIRE_FULL_REVISED_RUN)
FORCE_REBUILD_ANCHOR_DATASET = globals().get("FORCE_REBUILD_ANCHOR_DATASET", True)
FORCE_REVISED_SWEEP = globals().get("FORCE_REVISED_SWEEP", True)
ALLOW_UNCACHED_SWEEP_RUNS = globals().get("ALLOW_UNCACHED_SWEEP_RUNS", True)
ALLOW_REVISED_RETRAIN = globals().get("ALLOW_REVISED_RETRAIN", True)
ALLOW_SEVERITY_RETRAIN = globals().get("ALLOW_SEVERITY_RETRAIN", True)
PROMOTE_REVISED_MODEL = globals().get("PROMOTE_REVISED_MODEL", False)
FORCE_REVISED_STAGE1_RETRAIN = globals().get("FORCE_REVISED_STAGE1_RETRAIN", REQUIRE_FULL_REVISED_RUN)

REVISED_MODEL_SEED = globals().get("REVISED_MODEL_SEED", "yolo11m.pt")
REVISED_EPOCHS = int(globals().get("REVISED_EPOCHS", 120))
REVISED_CLEAN_EPOCHS = int(globals().get("REVISED_CLEAN_EPOCHS", max(30, round(REVISED_EPOCHS * 0.30))))
REVISED_ROBUST_EPOCHS = int(globals().get("REVISED_ROBUST_EPOCHS", max(45, REVISED_EPOCHS - REVISED_CLEAN_EPOCHS)))
REVISED_IMGSZ = int(globals().get("REVISED_IMGSZ", 768))
REVISED_BATCH = int(globals().get("REVISED_BATCH", 8))
REVISED_PATIENCE = int(globals().get("REVISED_PATIENCE", 30))

DRIVE_OUTPUT_FOLDER_ID = "1X4IGra-ySuPqbc_PYIs2pl3yhO1HyIbX"
DRIVE_MOUNT_ROOT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT_ROOT / "MyDrive"
DRIVE_SHARED_ROOT = DRIVE_MOUNT_ROOT / "Shareddrives"
DRIVE_SHORTCUT_ROOT = DRIVE_MOUNT_ROOT / ".shortcut-targets-by-id"


def now_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_existing_dirs(paths) -> list[Path]:
    seen = set()
    out = []
    for raw in paths:
        path = Path(raw)
        key = str(path)
        if key in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(key)
        out.append(path)
    return out


def drive_search_roots() -> list[Path]:
    roots = [DRIVE_ROOT, DRIVE_SHORTCUT_ROOT / DRIVE_OUTPUT_FOLDER_ID]
    if DRIVE_SHORTCUT_ROOT.exists():
        roots.extend(DRIVE_SHORTCUT_ROOT.glob("*"))
        roots.extend(DRIVE_SHORTCUT_ROOT.glob("*/*"))
    if DRIVE_SHARED_ROOT.exists():
        roots.extend(DRIVE_SHARED_ROOT.glob("*"))
        roots.extend(DRIVE_SHARED_ROOT.glob("*/*"))
    return unique_existing_dirs(roots)


def resolve_drive_output_root() -> Path:
    candidates = [
        DRIVE_SHORTCUT_ROOT / DRIVE_OUTPUT_FOLDER_ID,
        DRIVE_SHORTCUT_ROOT / DRIVE_OUTPUT_FOLDER_ID / "AIEngGroupProj_colab_outputs",
        DRIVE_ROOT / "AIEngGroupProj_colab_outputs",
    ]
    for root in drive_search_roots():
        candidates.extend(
            [
                root,
                root / "AIEngGroupProj_colab_outputs",
                root / "AIEngGroupProj" / "AIEngGroupProj_colab_outputs",
            ]
        )
    candidates = unique_existing_dirs(candidates) or [DRIVE_ROOT / "AIEngGroupProj_colab_outputs"]
    for candidate in candidates:
        if (candidate / "weights").exists() or (candidate / "runs").exists() or (candidate / "datasets").exists():
            print(f"Using Drive output root: {candidate}")
            return candidate
    print("No existing Drive output root found; creating:", candidates[0])
    return candidates[0]


DRIVE_OUTPUT_ROOT = resolve_drive_output_root()
DRIVE_WEIGHTS_ROOT = DRIVE_OUTPUT_ROOT / "weights"
DRIVE_RUNS_ROOT = DRIVE_OUTPUT_ROOT / "runs"
DRIVE_DATASETS_ROOT = DRIVE_OUTPUT_ROOT / "datasets"
DRIVE_DATASET_ARCHIVES_ROOT = DRIVE_OUTPUT_ROOT / "dataset_archives"
DRIVE_LOG_ROOT = DRIVE_OUTPUT_ROOT / "logs"
STATE_PATH = DRIVE_OUTPUT_ROOT / "pipeline_state_revised_training.json"
for path in (DRIVE_OUTPUT_ROOT, DRIVE_WEIGHTS_ROOT, DRIVE_RUNS_ROOT, DRIVE_DATASETS_ROOT, DRIVE_DATASET_ARCHIVES_ROOT, DRIVE_LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def get_secret(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value or default
    except Exception:
        return default


ROBOFLOW_API_KEY = globals().get("ROBOFLOW_API_KEY", get_secret("ROBOFLOW_API_KEY", ""))
KAGGLE_USERNAME = globals().get("KAGGLE_USERNAME", get_secret("KAGGLE_USERNAME", ""))
KAGGLE_KEY = globals().get("KAGGLE_KEY", get_secret("KAGGLE_KEY", ""))


def ensure_repo_root() -> Path:
    if REPO_ROOT_OVERRIDE:
        return Path(REPO_ROOT_OVERRIDE).resolve()
    if Path("/content/AIEngGroupProj").exists():
        return Path("/content/AIEngGroupProj").resolve()
    if Path.cwd().name == "AIEngGroupProj":
        return Path.cwd().resolve()
    if GITHUB_REPO_URL:
        subprocess.run(["git", "clone", GITHUB_REPO_URL, "/content/AIEngGroupProj"], check=True)
        return Path("/content/AIEngGroupProj").resolve()
    raise FileNotFoundError("Set REPO_ROOT_OVERRIDE or clone AIEngGroupProj before running this notebook.")


REPO_ROOT = ensure_repo_root()
os.chdir(REPO_ROOT)
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

if not globals().get("_REVISED_REQUIREMENTS_INSTALLED", False):
    requirements_path = REPO_ROOT / "requirements.txt"
    if requirements_path.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], check=True)
    globals()["_REVISED_REQUIREMENTS_INSTALLED"] = True

DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector.pt"
SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls.pt"
ANCHOR_CANDIDATE_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_anchor_balanced_candidate.pt"
ANCHOR_PROMOTED_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_anchor_balanced_promoted.pt"
BALANCED_CANDIDATE_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_balanced_candidate.pt"
REQUIRED_DETECTION_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"State read failed, starting fresh: {exc}")
    return {"steps": {}, "decisions": {}}


STATE = load_state()


def save_state() -> None:
    STATE_PATH.write_text(json.dumps(STATE, indent=2), encoding="utf-8")


def mark_done(step: str, payload: dict | None = None) -> None:
    STATE.setdefault("steps", {})[step] = {"done_at": now_token(), "payload": payload or {}}
    save_state()


def block_or_raise(step: str, message: str, payload: dict | None = None) -> None:
    details = {"message": message}
    if payload:
        details.update(payload)
    mark_done(f"{step}_blocked", details)
    if REQUIRE_FULL_REVISED_RUN:
        raise RuntimeError(message)
    print("SKIP:", message)


def step_done(step: str, expected_paths=()) -> bool:
    if not STATE.get("steps", {}).get(step):
        return False
    return all(Path(path).exists() for path in expected_paths)


def run_process(args: list[str], log_name: str, allow_failure: bool = False) -> subprocess.CompletedProcess:
    log_path = DRIVE_LOG_ROOT / f"{now_token()}_{log_name}.log"
    print("Running:", " ".join(map(str, args)))
    started = time.perf_counter()
    result = subprocess.run(list(map(str, args)), cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    log_path.write_text(result.stdout or "", encoding="utf-8")
    tail = (result.stdout or "").strip()[-4000:]
    if tail:
        print(tail)
    print(f"Log saved to: {log_path} ({elapsed:.1f}s)")
    if result.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(map(str, args))}")
    return result


def mirror_path(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)


def restore_path(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    mirror_path(src, dst)
    return True


def dataset_archive_name(dataset_name: str) -> str:
    return f"{dataset_name}.tar.gz"


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination)):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination)


def archive_dataset(dataset_root: Path, dataset_name: str) -> Path | None:
    dataset_root = Path(dataset_root)
    if not (dataset_root / "data.yaml").exists():
        return None
    archive_path = DRIVE_DATASET_ARCHIVES_ROOT / dataset_archive_name(dataset_name)
    if archive_path.exists() and archive_path.stat().st_size > 1_000_000:
        return archive_path
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = REPO_ROOT / f"_{dataset_name}_archive_tmp.tar.gz"
    if tmp_archive.exists():
        tmp_archive.unlink()
    print(f"Creating Drive dataset archive for fast future recovery: {archive_path}")
    started = time.perf_counter()
    with tarfile.open(tmp_archive, "w:gz") as archive:
        archive.add(dataset_root, arcname=dataset_name)
    shutil.copy2(tmp_archive, archive_path)
    tmp_archive.unlink(missing_ok=True)
    print(f"Archive ready: {archive_path} ({(time.perf_counter() - started):.1f}s)")
    return archive_path


def restore_dataset_from_archive(dataset_name: str, destination: Path) -> bool:
    if (destination / "data.yaml").exists():
        return True
    candidates = [
        DRIVE_DATASET_ARCHIVES_ROOT / dataset_archive_name(dataset_name),
        DRIVE_OUTPUT_ROOT / "dataset_archives" / dataset_archive_name(dataset_name),
        DRIVE_OUTPUT_ROOT / "dataset_cache" / dataset_archive_name(dataset_name),
        DRIVE_OUTPUT_ROOT / "datasets" / dataset_archive_name(dataset_name),
    ]
    for root in drive_search_roots():
        candidates.extend(
            [
                root / "dataset_archives" / dataset_archive_name(dataset_name),
                root / "dataset_cache" / dataset_archive_name(dataset_name),
                root / "datasets" / dataset_archive_name(dataset_name),
                root / "AIEngGroupProj_colab_outputs" / "dataset_archives" / dataset_archive_name(dataset_name),
            ]
        )
    local_cache = REPO_ROOT / "_dataset_archive_cache"
    local_cache.mkdir(parents=True, exist_ok=True)
    for archive_path in candidates:
        if not archive_path.exists() or not archive_path.is_file():
            continue
        local_archive = local_cache / archive_path.name
        print(f"Restoring {dataset_name} from archive: {archive_path}")
        started = time.perf_counter()
        shutil.copy2(archive_path, local_archive)
        extract_tmp = REPO_ROOT / f"_extract_{dataset_name}"
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)
        extract_tmp.mkdir(parents=True, exist_ok=True)
        safe_extract_tar(local_archive, extract_tmp)
        direct = extract_tmp / dataset_name
        if (direct / "data.yaml").exists():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(direct), str(destination))
        elif (extract_tmp / "data.yaml").exists():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(extract_tmp), str(destination))
            extract_tmp = None
        else:
            matches = [path for path in extract_tmp.rglob("data.yaml") if path.parent.name == dataset_name]
            if not matches:
                raise RuntimeError(f"Archive did not contain a recognizable {dataset_name}/data.yaml: {archive_path}")
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(matches[0].parent), str(destination))
        if extract_tmp is not None and extract_tmp.exists():
            shutil.rmtree(extract_tmp, ignore_errors=True)
        print(f"Archive restore complete for {dataset_name} ({(time.perf_counter() - started):.1f}s)")
        return True
    return False


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_completed_best(run_name_contains: str) -> Path | None:
    candidates = []
    for root in [DRIVE_RUNS_ROOT, DRIVE_OUTPUT_ROOT / "runs", DRIVE_ROOT / "runs"]:
        if not root.exists():
            continue
        for path in root.glob(f"**/*{run_name_contains}*/weights/best.pt"):
            if path.exists() and path.stat().st_size > 1_000_000:
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def recover_weight(target: Path, local_name: str, run_name_contains: str = "") -> bool:
    if target.exists() and target.stat().st_size > 1_000_000:
        shutil.copy2(target, WEIGHTS_DIR / local_name)
        print(f"Recovered canonical Drive weight: {target}")
        return True
    if run_name_contains:
        best = find_completed_best(run_name_contains)
        if best:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, target)
            shutil.copy2(best, WEIGHTS_DIR / local_name)
            print(f"Recovered completed run weight: {best}")
            return True
    print(f"Missing recoverable weight: {target}")
    return False


def dataset_counts_from_labels(dataset_root: Path) -> dict[str, dict[str, int]]:
    names = REQUIRED_DETECTION_CLASSES
    counts = {split: {name: 0 for name in names} for split in ["train", "val", "test"]}
    for split in counts:
        labels_dir = dataset_root / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(names):
                    counts[split][names[class_id]] += 1
    return counts


def dataset_has_all_classes(dataset_root: Path, min_train: int = 1, min_val: int = 1) -> bool:
    if not (dataset_root / "data.yaml").exists():
        return False
    counts = dataset_counts_from_labels(dataset_root)
    return all(counts["train"].get(name, 0) >= min_train and counts["val"].get(name, 0) >= min_val for name in REQUIRED_DETECTION_CLASSES)


print(f"Repo: {REPO_ROOT}")
print(f"Drive output root: {DRIVE_OUTPUT_ROOT}")
print(f"State: {STATE_PATH}")
'''


BASELINE_RECOVERY_CODE = r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"

for dataset_name, destination in [
    ("merged_dataset", MERGED_DATASET),
    ("merged_dataset_anchor_balanced", ANCHOR_DATASET),
    ("merged_dataset_anchor_robust", ROBUST_DATASET),
]:
    if not restore_dataset_from_archive(dataset_name, destination) and ALLOW_SLOW_DRIVE_FOLDER_RESTORE:
        restore_path(DRIVE_DATASETS_ROOT / dataset_name, destination)


def bootstrap_merged_dataset_archive_from_sources() -> None:
    if dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1):
        return
    archive_path = DRIVE_DATASET_ARCHIVES_ROOT / "merged_dataset.tar.gz"
    if restore_dataset_from_archive("merged_dataset", MERGED_DATASET):
        return
    if not PREFER_SOURCE_ARCHIVE_BOOTSTRAP:
        return
    print("No merged_dataset archive found. Building it directly from Roboflow/Kaggle sources before any slow Drive folder copy.")
    os.environ["ROBOFLOW_API_KEY"] = ROBOFLOW_API_KEY or ""
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME or ""
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY or ""
    run_process(
        [
            sys.executable,
            "scripts/bootstrap_merged_dataset_archive_from_sources.py",
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(archive_path),
        ],
        log_name="bootstrap_merged_dataset_archive_from_sources",
        allow_failure=not REQUIRE_FULL_REVISED_RUN,
    )
    restore_dataset_from_archive("merged_dataset", MERGED_DATASET)


bootstrap_merged_dataset_archive_from_sources()

def restore_dataset_by_name(dataset_name: str, destination: Path) -> bool:
    if (destination / "data.yaml").exists():
        return True
    if restore_dataset_from_archive(dataset_name, destination):
        return True
    if not ALLOW_SLOW_DRIVE_FOLDER_RESTORE:
        print(
            f"Slow Drive folder restore disabled for {dataset_name}. "
            "Use dataset_archives/*.tar.gz or set ALLOW_SLOW_DRIVE_FOLDER_RESTORE=True only on CPU runtime."
        )
        return False
    candidates = [
        DRIVE_DATASETS_ROOT / dataset_name,
        DRIVE_OUTPUT_ROOT / dataset_name,
        Path("/content/drive/MyDrive") / "AIEngGroupProj_colab_outputs" / dataset_name,
        Path("/content/drive/MyDrive") / "AIEngGroupProj_colab_outputs" / "datasets" / dataset_name,
        Path("/content/drive/MyDrive") / "AIEngGroupProj" / dataset_name,
        Path("/content/drive/MyDrive") / dataset_name,
    ]
    for root in drive_search_roots():
        candidates.extend(
            [
                root / dataset_name,
                root / "datasets" / dataset_name,
                root / "dataset_cache" / dataset_name,
                root / "AIEngGroupProj_colab_outputs" / dataset_name,
                root / "AIEngGroupProj_colab_outputs" / "datasets" / dataset_name,
                root / "AIEngGroupProj_colab_outputs" / "dataset_cache" / dataset_name,
                root / "AIEngGroupProj" / dataset_name,
            ]
        )
    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and (candidate / "data.yaml").exists():
            print(f"Restoring dataset {dataset_name} from: {candidate}")
            restored = restore_path(candidate, destination)
            if restored:
                archive_dataset(destination, dataset_name)
            return restored
    print(f"No recoverable dataset folder found for: {dataset_name}")
    print("Checked dataset locations:")
    for item in checked[:80]:
        print(" -", item)
    if len(checked) > 80:
        print(f" - ... {len(checked) - 80} more locations")
    return False

restore_dataset_by_name("merged_dataset", MERGED_DATASET)
restore_dataset_by_name("merged_dataset_anchor_balanced", ANCHOR_DATASET)
restore_dataset_by_name("merged_dataset_anchor_robust", ROBUST_DATASET)

def extract_raw_zip_if_available() -> None:
    zip_candidates = []
    for root in drive_search_roots():
        zip_candidates.extend(
            [
                root / "raw_datasets.zip",
                root / "AIEngGroupProj_colab_outputs" / "raw_datasets.zip",
                root / "AIEngGroupProj" / "raw_datasets.zip",
                root / "datasets" / "raw_datasets.zip",
            ]
        )
    for zip_path in zip_candidates:
        if zip_path.exists() and zip_path.is_file():
            print(f"Extracting raw dataset zip from: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(REPO_ROOT)
            return


def restore_raw_dataset_folder(dataset_name: str, dataset_path: str) -> bool:
    destination = REPO_ROOT / dataset_path
    if (destination / "data.yaml").exists():
        return True
    if not ALLOW_SLOW_DRIVE_FOLDER_RESTORE:
        print(
            f"Slow Drive raw-folder restore disabled for {dataset_name}. "
            "The preferred path is direct source download via bootstrap_merged_dataset_archive_from_sources.py."
        )
        return False
    candidates = []
    for root in drive_search_roots():
        candidates.extend(
            [
                root / dataset_path,
                root / dataset_name,
                root / "raw_dataset_cache" / dataset_path,
                root / "raw_dataset_cache" / dataset_name,
                root / "raw_datasets" / dataset_path,
                root / "raw_datasets" / dataset_name,
                root / "datasets" / "raw_sources" / dataset_path,
                root / "datasets" / "raw_sources" / dataset_name,
                root / "AIEngGroupProj" / dataset_path,
                root / "AIEngGroupProj_colab_outputs" / "raw_datasets" / dataset_path,
                root / "AIEngGroupProj_colab_outputs" / "raw_dataset_cache" / dataset_path,
                root / "AIEngGroupProj_colab_outputs" / "datasets" / "raw_sources" / dataset_path,
            ]
        )
        backup_root = root / "raw_dataset_cache_backups"
        if backup_root.exists():
            candidates.extend(sorted(backup_root.glob(f"{dataset_path}*")))
            candidates.extend(sorted(backup_root.glob(f"{dataset_name}*")))
    for candidate in candidates:
        if candidate.exists() and (candidate / "data.yaml").exists():
            print(f"Restoring raw source {dataset_name} from: {candidate}")
            return restore_path(candidate, destination)
    print(f"Raw source not found yet: {dataset_name}")
    return False


def rebuild_baseline_merged_dataset_if_needed() -> None:
    if dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1):
        print("Baseline merged_dataset already contains all five classes.")
        return

    extract_raw_zip_if_available()
    merge_config_path = REPO_ROOT / "configs" / "merge_config.yaml"
    merge_config = yaml.safe_load(merge_config_path.read_text(encoding="utf-8")) or {}
    missing_sources = []
    for dataset in merge_config.get("datasets", []):
        name = str(dataset.get("name") or dataset.get("path"))
        path = str(dataset.get("path") or name)
        if not restore_raw_dataset_folder(name, path):
            missing_sources.append(name)

    if missing_sources:
        block_or_raise(
            "baseline_raw_sources",
            "Cannot rebuild baseline merged_dataset because raw source folders are missing: " + ", ".join(missing_sources),
            {
                "missing_sources": missing_sources,
                "searched_roots": [str(path) for path in drive_search_roots()],
                "expected_zip": "raw_datasets.zip",
            },
        )
        return

    run_process(
        [
            sys.executable,
            "scripts/merge_datasets.py",
            "--config",
            str(merge_config_path),
            "--preserve-splits",
            "--force",
        ],
        log_name="strict_rebuild_baseline_merged_dataset",
    )

    if not dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1):
        counts = dataset_counts_from_labels(MERGED_DATASET) if MERGED_DATASET.exists() else {}
        block_or_raise(
            "baseline_merged_dataset",
            "Baseline merged_dataset rebuild completed but does not contain all five classes in train and val.",
            {"counts": counts, "merged_dataset": str(MERGED_DATASET)},
        )
        return

    archive_dataset(MERGED_DATASET, "merged_dataset")
    if MIRROR_FULL_DATASET_FOLDERS:
        mirror_path(MERGED_DATASET, DRIVE_DATASETS_ROOT / "merged_dataset")
    print("Baseline merged_dataset rebuilt and persisted to Drive archive.")


rebuild_baseline_merged_dataset_if_needed()

baseline_ok = dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1)
anchor_ok = dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30)
robust_ok = dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30)
print("Baseline merged_dataset all-class ready:", baseline_ok)
print("Recovered anchor-balanced dataset ready:", anchor_ok)
print("Recovered robust dataset ready:", robust_ok)
if MERGED_DATASET.exists():
    print("Baseline merged_dataset counts:")
    print(json.dumps(dataset_counts_from_labels(MERGED_DATASET), indent=2))
if ANCHOR_DATASET.exists():
    print("Anchor-balanced counts:")
    print(json.dumps(dataset_counts_from_labels(ANCHOR_DATASET), indent=2))
if ROBUST_DATASET.exists():
    print("Robust dataset counts:")
    print(json.dumps(dataset_counts_from_labels(ROBUST_DATASET), indent=2))

if baseline_ok:
    archive_dataset(MERGED_DATASET, "merged_dataset")
if anchor_ok:
    archive_dataset(ANCHOR_DATASET, "merged_dataset_anchor_balanced")
if robust_ok:
    archive_dataset(ROBUST_DATASET, "merged_dataset_anchor_robust")

recover_weight(DETECTOR_WEIGHT, "defect_detector.pt", "stage1")
recover_weight(SEVERITY_WEIGHT, "severity_cls.pt", "stage2")
recover_weight(BALANCED_CANDIDATE_WEIGHT, "defect_detector_balanced_candidate.pt", "stage1_balanced")
recover_weight(ANCHOR_CANDIDATE_WEIGHT, "defect_detector_anchor_balanced_candidate.pt", "stage1_anchor_balanced")

if not baseline_ok:
    block_or_raise(
        "revised_baseline_recovered",
        "Strict full run requires an all-class baseline merged_dataset before any revised training can proceed.",
        {
            "merged_dataset": str(MERGED_DATASET),
            "counts": dataset_counts_from_labels(MERGED_DATASET) if MERGED_DATASET.exists() else {},
        },
    )

mark_done(
    "revised_baseline_recovered",
    {
        "merged_dataset": str(MERGED_DATASET),
        "baseline_ok": baseline_ok,
        "anchor_dataset": str(ANCHOR_DATASET),
        "anchor_ok": anchor_ok,
        "robust_dataset": str(ROBUST_DATASET),
        "robust_ok": robust_ok,
        "counts": dataset_counts_from_labels(MERGED_DATASET) if MERGED_DATASET.exists() else {},
    },
)
'''


TARGET_COLLECTION_CODE = r'''
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"
BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"

target_summary = TARGET_YOLO / "auto_collection_summary.json"
if FORCE_DOMAIN_COLLECTION or not target_summary.exists():
    if not ROBOFLOW_API_KEY:
        raise RuntimeError("ROBOFLOW_API_KEY is required for target-domain collection.")
    run_process(
        [
            sys.executable,
            "scripts/auto_collect_domain_sources.py",
            "--api-key",
            ROBOFLOW_API_KEY,
            "--raw-output",
            "domain_adaptation/auto_raw",
            "--target-output",
            str(TARGET_YOLO),
            "--force-rebuild",
        ],
        log_name="revised_collect_target_domain",
    )
else:
    print(f"Using cached target-domain collection: {target_summary}")

print("Target-domain summary:")
if target_summary.exists():
    print(target_summary.read_text(encoding="utf-8")[-5000:])

HARD_NEGATIVES.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    (HARD_NEGATIVES / split / "images").mkdir(parents=True, exist_ok=True)
    (HARD_NEGATIVES / split / "labels").mkdir(parents=True, exist_ok=True)

resolved_manifest = BASELINE_SWEEP_DIR / "resolved_manifest.csv"
copied = 0
if resolved_manifest.exists():
    with resolved_manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("expected_class") != "none":
                continue
            src = Path(row.get("resolved_path", ""))
            if not src.exists():
                continue
            dst = HARD_NEGATIVES / "train" / "images" / f"hardneg_{src.name}"
            label = HARD_NEGATIVES / "train" / "labels" / f"{dst.stem}.txt"
            if not dst.exists():
                shutil.copy2(src, dst)
            label.write_text("", encoding="utf-8")
            copied += 1
print(f"Hard-negative images prepared from cached sweeps: {copied}")

mark_done("revised_target_collection_ready", {"target": str(TARGET_YOLO), "hard_negatives": str(HARD_NEGATIVES)})
'''


ANCHOR_DATASET_CODE = r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
ANCHOR_AUDIT_DIR = REPO_ROOT / "output" / "_anchor_dataset_preflight_audit"
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"

summary_path = ANCHOR_DATASET / "anchor_balanced_summary.json"
if FORCE_REBUILD_ANCHOR_DATASET or not summary_path.exists() or not dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30):
    if ANCHOR_AUDIT_DIR.exists():
        shutil.rmtree(ANCHOR_AUDIT_DIR)
    preflight_args = [
        sys.executable,
        "scripts/build_anchor_balanced_dataset.py",
        "--base",
        str(MERGED_DATASET),
        "--target",
        str(TARGET_YOLO),
        "--hard-negatives",
        str(HARD_NEGATIVES),
        "--output",
        str(ANCHOR_AUDIT_DIR),
        "--target-box-goal-per-class",
        "1500",
        "--max-target-images-per-class",
        "900",
        "--max-target-train-images",
        "3600",
        "--target-val-images-per-class",
        "35",
        "--oversample-target-boxes",
        "6500",
        "--max-repeat-per-image",
        "4",
        "--max-hard-negative-train",
        "450",
        "--max-hard-negative-val",
        "100",
        "--min-image-side",
        "96",
        "--min-blur-variance",
        "4.0",
        "--max-box-area",
        "0.92",
        "--max-boxes-per-image",
        "90",
        "--max-target-classes-per-image",
        "3",
        "--max-invalid-label-fraction",
        "0.0",
        "--min-quality-target-boxes-per-class",
        "25",
        "--min-target-keep-rate",
        "0.75",
        "--min-final-train-boxes-per-class",
        "1200",
        "--min-final-val-boxes-per-class",
        "30",
        "--max-final-class-ratio",
        "6",
        "--audit-only",
    ]
    run_process(preflight_args, log_name="revised_anchor_dataset_preflight_audit")
    audit_summary_path = ANCHOR_AUDIT_DIR / "anchor_balanced_audit_summary.json"
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    mirror_path(audit_summary_path, DRIVE_OUTPUT_ROOT / "summaries" / "anchor_balanced_preflight_audit_summary.json")
    print("Preflight target keep rate:", audit_summary["quality_filter"]["target"]["keep_rate"])
    print("Preflight rejected target samples:", audit_summary["quality_filter"]["target"]["rejected_reasons"])
    print("Preflight final guard:", audit_summary["guard"])
    shutil.rmtree(ANCHOR_AUDIT_DIR, ignore_errors=True)

    args = [
        sys.executable,
        "scripts/build_anchor_balanced_dataset.py",
        "--base",
        str(MERGED_DATASET),
        "--target",
        str(TARGET_YOLO),
        "--hard-negatives",
        str(HARD_NEGATIVES),
        "--output",
        str(ANCHOR_DATASET),
        "--target-box-goal-per-class",
        "1500",
        "--max-target-images-per-class",
        "900",
        "--max-target-train-images",
        "3600",
        "--target-val-images-per-class",
        "35",
        "--oversample-target-boxes",
        "6500",
        "--max-repeat-per-image",
        "4",
        "--max-hard-negative-train",
        "450",
        "--max-hard-negative-val",
        "100",
        "--min-image-side",
        "96",
        "--min-blur-variance",
        "4.0",
        "--max-box-area",
        "0.92",
        "--max-boxes-per-image",
        "90",
        "--max-target-classes-per-image",
        "3",
        "--max-invalid-label-fraction",
        "0.0",
        "--min-quality-target-boxes-per-class",
        "25",
        "--min-target-keep-rate",
        "0.75",
        "--min-final-train-boxes-per-class",
        "1200",
        "--min-final-val-boxes-per-class",
        "30",
        "--max-final-class-ratio",
        "6",
    ]
    if FORCE_REBUILD_ANCHOR_DATASET:
        args.append("--force")
    run_process(args, log_name="revised_build_anchor_balanced_dataset")
else:
    print(f"Using cached anchor-balanced dataset: {summary_path}")

print(summary_path.read_text(encoding="utf-8")[-7000:])
archive_dataset(ANCHOR_DATASET, "merged_dataset_anchor_balanced")
if MIRROR_FULL_DATASET_FOLDERS:
    mirror_path(ANCHOR_DATASET, DRIVE_DATASETS_ROOT / "merged_dataset_anchor_balanced")

robust_summary_path = ROBUST_DATASET / "robust_augmentation_summary.json"
if FORCE_REBUILD_ANCHOR_DATASET or not robust_summary_path.exists() or not dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30):
    run_process(
        [
            sys.executable,
            "scripts/build_robust_augmented_dataset.py",
            "--input",
            str(ANCHOR_DATASET),
            "--output",
            str(ROBUST_DATASET),
            "--aug-box-goal-per-class",
            "1800",
            "--max-aug-images-per-class",
            "650",
            "--max-negative-aug-images",
            "240",
            "--max-occlusion-box-overlap",
            "0.35",
            "--scale-space-crops-per-class",
            "320",
            "--scale-space-contexts",
            "1.35,2.20,3.40",
            "--scale-space-min-visible-fraction",
            "0.55",
            "--min-train-boxes-per-class",
            "1200",
            "--min-val-boxes-per-class",
            "30",
            "--max-train-class-ratio",
            "6",
        ],
        log_name="revised_build_robust_augmented_dataset",
    )
else:
    print(f"Using cached robust augmented dataset: {robust_summary_path}")

print(robust_summary_path.read_text(encoding="utf-8")[-7000:])
feature_summary_path = DRIVE_OUTPUT_ROOT / "summaries" / "robust_feature_coverage_summary.json"
run_process(
    [
        sys.executable,
        "scripts/audit_defect_feature_coverage.py",
        "--dataset",
        str(ROBUST_DATASET),
        "--output",
        str(feature_summary_path),
        "--min-train-boxes-per-class",
        "1200",
        "--min-val-boxes-per-class",
        "30",
        "--max-train-class-ratio",
        "6",
        "--min-feature-bins",
        "2",
        "--min-bin-boxes",
        "40",
    ],
    log_name="revised_audit_feature_coverage",
)
print(feature_summary_path.read_text(encoding="utf-8")[-7000:])
archive_dataset(ROBUST_DATASET, "merged_dataset_anchor_robust")
if MIRROR_FULL_DATASET_FOLDERS:
    mirror_path(ROBUST_DATASET, DRIVE_DATASETS_ROOT / "merged_dataset_anchor_robust")
mark_done(
    "revised_anchor_robust_dataset_ready",
    {
        "anchor_dataset": str(ANCHOR_DATASET),
        "robust_dataset": str(ROBUST_DATASET),
        "anchor_summary": str(summary_path),
        "robust_summary": str(robust_summary_path),
        "feature_summary": str(feature_summary_path),
    },
)
'''


REVISED_TRAIN_CODE = r'''
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
run_name = "stage1_anchor_balanced_yolo11m_detector"
run_dir = DRIVE_RUNS_ROOT / run_name
best_path = run_dir / "weights" / "best.pt"
last_path = run_dir / "weights" / "last.pt"

if ANCHOR_CANDIDATE_WEIGHT.exists() and not ALLOW_REVISED_RETRAIN:
    print(f"Using existing revised candidate: {ANCHOR_CANDIDATE_WEIGHT}")
elif best_path.exists() and best_path.stat().st_size > 1_000_000 and not ALLOW_REVISED_RETRAIN:
    shutil.copy2(best_path, ANCHOR_CANDIDATE_WEIGHT)
    print(f"Recovered revised candidate from completed run: {best_path}")
elif not ALLOW_REVISED_RETRAIN:
    raise RuntimeError(
        "No revised candidate weight exists yet. Set ALLOW_REVISED_RETRAIN=True to train. "
        "This guard prevents accidental Colab compute use."
    )
else:
    train_args = [
        "yolo",
        "detect",
        "train",
        f"model={REVISED_MODEL_SEED}",
        f"data={ROBUST_DATASET / 'data.yaml'}",
        f"epochs={REVISED_EPOCHS}",
        f"imgsz={REVISED_IMGSZ}",
        f"batch={REVISED_BATCH}",
        f"patience={REVISED_PATIENCE}",
        "optimizer=AdamW",
        "lr0=0.001",
        "lrf=0.01",
        "weight_decay=0.0005",
        "box=8.0",
        "cls=0.8",
        "dfl=1.6",
        "warmup_epochs=3",
        "close_mosaic=25",
        "cos_lr=True",
        "multi_scale=True",
        "hsv_h=0.015",
        "hsv_s=0.55",
        "hsv_v=0.55",
        "degrees=4.0",
        "translate=0.08",
        "scale=0.42",
        "shear=1.5",
        "perspective=0.0008",
        "fliplr=0.5",
        "flipud=0.0",
        "mosaic=0.55",
        "mixup=0.04",
        "cutmix=0.04",
        "erasing=0.18",
        "bgr=0.08",
        "cache=disk",
        "seed=42",
        f"project={DRIVE_RUNS_ROOT}",
        f"name={run_name}",
        "exist_ok=True",
    ]
    if last_path.exists():
        print(f"Previous last.pt exists at {last_path}; Ultralytics will continue in the same run folder if possible.")
    run_process(train_args, log_name="revised_train_anchor_balanced_detector")
    if not best_path.exists():
        raise RuntimeError(f"Training finished but best.pt is missing: {best_path}")
    shutil.copy2(best_path, ANCHOR_CANDIDATE_WEIGHT)
    print(f"Saved revised candidate: {ANCHOR_CANDIDATE_WEIGHT}")

shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, WEIGHTS_DIR / "defect_detector_anchor_balanced_candidate.pt")
print(f"Candidate sha256: {file_sha(ANCHOR_CANDIDATE_WEIGHT)[:16]}")
mark_done("revised_detector_candidate_ready", {"weight": str(ANCHOR_CANDIDATE_WEIGHT), "sha256": file_sha(ANCHOR_CANDIDATE_WEIGHT)})
'''


REVISED_TRAIN_CODE = r'''
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
clean_run_name = "stage1_clean_anchor_warmup_yolo11m_detector"
robust_run_name = "stage1_anchor_robust_curriculum_yolo11m_detector"
clean_run_dir = DRIVE_RUNS_ROOT / clean_run_name
robust_run_dir = DRIVE_RUNS_ROOT / robust_run_name
clean_best_path = clean_run_dir / "weights" / "best.pt"
clean_last_path = clean_run_dir / "weights" / "last.pt"
robust_best_path = robust_run_dir / "weights" / "best.pt"
robust_last_path = robust_run_dir / "weights" / "last.pt"


def metric_from_row(row: dict, predicate) -> float | None:
    for key, value in row.items():
        normalized = key.strip().lower().replace(" ", "")
        if predicate(normalized):
            try:
                return float(value)
            except Exception:
                return None
    return None


def read_training_summary(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"run_dir": str(run_dir), "has_results": False}
    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"run_dir": str(run_dir), "has_results": False}

    metric_rows = []
    for index, row in enumerate(rows):
        map50 = metric_from_row(row, lambda key: "map50" in key and "map50-95" not in key and "map50_95" not in key)
        map5095 = metric_from_row(row, lambda key: "map50-95" in key or "map50_95" in key)
        precision = metric_from_row(row, lambda key: "precision" in key)
        recall = metric_from_row(row, lambda key: "recall" in key)
        if map50 is None and map5095 is None:
            continue
        score = map5095 if map5095 is not None else map50
        try:
            epoch = int(float(row.get("epoch", index)))
        except Exception:
            epoch = index
        metric_rows.append(
            {
                "row_index": index,
                "epoch": epoch,
                "map50": map50,
                "map50_95": map5095,
                "precision": precision,
                "recall": recall,
                "score": score,
            }
        )
    if not metric_rows:
        return {"run_dir": str(run_dir), "has_results": True, "has_detection_metrics": False, "epochs": len(rows)}
    best = max(metric_rows, key=lambda item: item["score"] if item["score"] is not None else -1)
    last = metric_rows[-1]
    return {
        "run_dir": str(run_dir),
        "has_results": True,
        "has_detection_metrics": True,
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "best_score": best["score"],
        "last_epoch": last["epoch"],
        "last_map50": last["map50"],
        "last_map50_95": last["map50_95"],
        "last_score": last["score"],
        "epochs_since_best": max(0, len(metric_rows) - best["row_index"] - 1),
    }


def train_phase(args: list[str], best_path: Path, log_name: str) -> None:
    run_process(args, log_name=log_name)
    if not best_path.exists():
        raise RuntimeError(f"Training finished but best.pt is missing: {best_path}")


if ANCHOR_CANDIDATE_WEIGHT.exists() and not ALLOW_REVISED_RETRAIN:
    print(f"Using existing revised candidate: {ANCHOR_CANDIDATE_WEIGHT}")
elif robust_best_path.exists() and robust_best_path.stat().st_size > 1_000_000 and not ALLOW_REVISED_RETRAIN:
    shutil.copy2(robust_best_path, ANCHOR_CANDIDATE_WEIGHT)
    print(f"Recovered revised candidate from completed robust run: {robust_best_path}")
elif not ALLOW_REVISED_RETRAIN:
    raise RuntimeError(
        "No revised candidate weight exists yet. Set ALLOW_REVISED_RETRAIN=True to train. "
        "This guard prevents accidental Colab compute use."
    )
else:
    if FORCE_REVISED_STAGE1_RETRAIN or not clean_best_path.exists():
        clean_model = clean_last_path if clean_last_path.exists() and not clean_best_path.exists() else REVISED_MODEL_SEED
        clean_args = [
            "yolo", "detect", "train",
            f"model={clean_model}",
            f"data={ANCHOR_DATASET / 'data.yaml'}",
            f"epochs={REVISED_CLEAN_EPOCHS}",
            f"imgsz={REVISED_IMGSZ}",
            f"batch={REVISED_BATCH}",
            f"patience={max(12, min(REVISED_PATIENCE, 22))}",
            "optimizer=AdamW",
            "lr0=0.0012",
            "lrf=0.03",
            "weight_decay=0.0005",
            "box=8.0",
            "cls=0.75",
            "dfl=1.6",
            "warmup_epochs=3",
            "close_mosaic=10",
            "cos_lr=True",
            "multi_scale=True",
            "hsv_h=0.012",
            "hsv_s=0.35",
            "hsv_v=0.35",
            "degrees=2.0",
            "translate=0.05",
            "scale=0.25",
            "shear=0.5",
            "perspective=0.0003",
            "fliplr=0.5",
            "flipud=0.0",
            "mosaic=0.35",
            "mixup=0.0",
            "cutmix=0.0",
            "erasing=0.04",
            "bgr=0.02",
            "cache=disk",
            "seed=42",
            f"project={DRIVE_RUNS_ROOT}",
            f"name={clean_run_name}",
            "exist_ok=True",
        ]
        train_phase(clean_args, clean_best_path, "revised_train_clean_anchor_warmup")
    else:
        print(f"Reusing completed clean warm-up checkpoint: {clean_best_path}")

    if FORCE_REVISED_STAGE1_RETRAIN or not robust_best_path.exists():
        robust_model = robust_last_path if robust_last_path.exists() and not robust_best_path.exists() else clean_best_path
        robust_args = [
            "yolo", "detect", "train",
            f"model={robust_model}",
            f"data={ROBUST_DATASET / 'data.yaml'}",
            f"epochs={REVISED_ROBUST_EPOCHS}",
            f"imgsz={REVISED_IMGSZ}",
            f"batch={REVISED_BATCH}",
            f"patience={max(REVISED_PATIENCE, 35)}",
            "optimizer=AdamW",
            "lr0=0.00055",
            "lrf=0.02",
            "weight_decay=0.0005",
            "box=8.2",
            "cls=0.85",
            "dfl=1.7",
            "warmup_epochs=4",
            "close_mosaic=35",
            "cos_lr=True",
            "multi_scale=True",
            "hsv_h=0.010",
            "hsv_s=0.28",
            "hsv_v=0.30",
            "degrees=2.0",
            "translate=0.05",
            "scale=0.25",
            "shear=0.5",
            "perspective=0.0003",
            "fliplr=0.5",
            "flipud=0.0",
            "mosaic=0.20",
            "mixup=0.0",
            "cutmix=0.0",
            "erasing=0.05",
            "bgr=0.02",
            "cache=disk",
            "seed=43",
            f"project={DRIVE_RUNS_ROOT}",
            f"name={robust_run_name}",
            "exist_ok=True",
        ]
        train_phase(robust_args, robust_best_path, "revised_train_robust_curriculum_finetune")
    else:
        print(f"Reusing completed robust curriculum checkpoint: {robust_best_path}")

    clean_metrics = read_training_summary(clean_run_dir)
    robust_metrics = read_training_summary(robust_run_dir)
    selected_path = robust_best_path
    selected_reason = "robust_curriculum_selected"
    clean_score = clean_metrics.get("best_score")
    robust_score = robust_metrics.get("best_score")
    if clean_best_path.exists() and isinstance(clean_score, (int, float)) and isinstance(robust_score, (int, float)):
        if robust_score + 0.03 < clean_score:
            selected_path = clean_best_path
            selected_reason = "clean_warmup_selected_because_robust_validation_degraded"
    elif not robust_best_path.exists() and clean_best_path.exists():
        selected_path = clean_best_path
        selected_reason = "clean_warmup_selected_because_robust_missing"

    convergence_summary = {
        "clean_run": str(clean_run_dir),
        "robust_run": str(robust_run_dir),
        "clean_metrics": clean_metrics,
        "robust_metrics": robust_metrics,
        "selected_weight": str(selected_path),
        "selected_reason": selected_reason,
        "design": "curriculum: clean anchor warm-up first, robust/scale-space fine-tune second, conservative built-in augmentations because robust data already contains generated variants",
    }
    convergence_path = DRIVE_OUTPUT_ROOT / "summaries" / "stage1_curriculum_convergence_summary.json"
    convergence_path.parent.mkdir(parents=True, exist_ok=True)
    convergence_path.write_text(json.dumps(convergence_summary, indent=2), encoding="utf-8")
    print(json.dumps(convergence_summary, indent=2))

    shutil.copy2(selected_path, ANCHOR_CANDIDATE_WEIGHT)
    print(f"Saved revised candidate from {selected_reason}: {ANCHOR_CANDIDATE_WEIGHT}")

shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, WEIGHTS_DIR / "defect_detector_anchor_balanced_candidate.pt")
print(f"Candidate sha256: {file_sha(ANCHOR_CANDIDATE_WEIGHT)[:16]}")
mark_done(
    "revised_detector_candidate_ready",
    {
        "weight": str(ANCHOR_CANDIDATE_WEIGHT),
        "sha256": file_sha(ANCHOR_CANDIDATE_WEIGHT),
        "clean_run": str(clean_run_dir),
        "robust_run": str(robust_run_dir),
    },
)
'''


SEVERITY_CODE = r'''
SEVERITY_DATASET = REPO_ROOT / "severity_dataset"
severity_classes = ["critical", "minor", "moderate"]
MERGED_DATASET = globals().get("MERGED_DATASET", REPO_ROOT / "merged_dataset")
ANCHOR_DATASET = globals().get("ANCHOR_DATASET", REPO_ROOT / "merged_dataset_anchor_balanced")
ROBUST_DATASET = globals().get("ROBUST_DATASET", REPO_ROOT / "merged_dataset_anchor_robust")
ALLOW_COARSE_SEVERITY_BOOTSTRAP = globals().get("ALLOW_COARSE_SEVERITY_BOOTSTRAP", True)
TRAIN_COARSE_SEVERITY_BOOTSTRAP = globals().get("TRAIN_COARSE_SEVERITY_BOOTSTRAP", False)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEVERITY_MIN_TRAIN = 50
SEVERITY_MIN_VAL = 10

def count_severity_dataset(root: Path) -> dict[str, dict[str, int]]:
    counts = {split: {name: 0 for name in severity_classes} for split in ["train", "val", "test"]}
    split_aliases = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    for source_split, canonical_split in split_aliases.items():
        for class_name in severity_classes:
            folder = root / source_split / class_name
            if folder.exists():
                counts[canonical_split][class_name] += sum(
                    1 for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                )
    return counts

def severity_is_ready(counts: dict[str, dict[str, int]]) -> bool:
    return all(
        counts["train"].get(name, 0) >= SEVERITY_MIN_TRAIN
        and counts["val"].get(name, 0) >= SEVERITY_MIN_VAL
        for name in severity_classes
    )

def total_severity_images(counts: dict[str, dict[str, int]]) -> int:
    return sum(sum(class_counts.values()) for class_counts in counts.values())

def find_existing_severity_weight() -> Path | None:
    direct_candidates = [
        SEVERITY_WEIGHT,
        WEIGHTS_DIR / "severity_cls.pt",
        REPO_ROOT / "weights" / "severity_cls.pt",
        DRIVE_WEIGHTS_ROOT / "severity_cls.pt",
        DRIVE_WEIGHTS_ROOT / "severity_cls_domain_adapted.pt",
        DRIVE_RUNS_ROOT / "stage2_severity" / "weights" / "best.pt",
        DRIVE_RUNS_ROOT / "stage2_severity" / "weights" / "last.pt",
        DRIVE_RUNS_ROOT / "stage2_revised_severity_yolo11s_cls" / "weights" / "best.pt",
        DRIVE_RUNS_ROOT / "stage2_revised_severity_yolo11s_cls" / "weights" / "last.pt",
        DRIVE_RUNS_ROOT / "stage2_domain_adapted_severity" / "weights" / "best.pt",
    ]
    for candidate in direct_candidates:
        candidate = Path(candidate)
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            return candidate

    pattern_roots = [DRIVE_RUNS_ROOT, DRIVE_OUTPUT_ROOT / "runs", REPO_ROOT / "runs", REPO_ROOT / "weights"]
    patterns = ["**/*severity*/weights/best.pt", "**/*severity*/weights/last.pt", "**/*severity*.pt", "**/*stage2*/weights/best.pt"]
    found = []
    for root in pattern_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                candidate_text = str(candidate).lower()
                if "detect" in candidate_text or "detector" in candidate_text:
                    continue
                if candidate.exists() and candidate.stat().st_size > 1_000_000:
                    found.append(candidate)
    return max(found, key=lambda path: path.stat().st_mtime) if found else None

def sync_existing_severity_weight(source: Path, reason: str) -> None:
    SEVERITY_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if source.resolve() != SEVERITY_WEIGHT.resolve():
        shutil.copy2(source, SEVERITY_WEIGHT)
    shutil.copy2(SEVERITY_WEIGHT, WEIGHTS_DIR / "severity_cls.pt")
    print(f"Severity classifier ready from {reason}: {source}")
    print(f"Canonical severity weight: {SEVERITY_WEIGHT}")

def restore_severity_dataset_from_persistent_storage() -> str | None:
    if restore_dataset_from_archive("severity_dataset", SEVERITY_DATASET):
        return "Drive dataset archive"

    candidates = [
        DRIVE_DATASETS_ROOT / "severity_dataset",
        DRIVE_OUTPUT_ROOT / "severity_dataset",
        DRIVE_OUTPUT_ROOT / "dataset_cache" / "severity_dataset",
        Path("/content/drive/MyDrive") / "AIEngGroupProj_colab_outputs" / "datasets" / "severity_dataset",
        Path("/content/drive/MyDrive") / "AIEngGroupProj_colab_outputs" / "dataset_cache" / "severity_dataset",
    ]
    for root in drive_search_roots():
        candidates.extend(
            [
                root / "severity_dataset",
                root / "datasets" / "severity_dataset",
                root / "dataset_cache" / "severity_dataset",
                root / "AIEngGroupProj_colab_outputs" / "severity_dataset",
                root / "AIEngGroupProj_colab_outputs" / "datasets" / "severity_dataset",
                root / "AIEngGroupProj_colab_outputs" / "dataset_cache" / "severity_dataset",
            ]
        )

    for candidate in candidates:
        if not candidate.exists() or not (candidate / "data.yaml").exists():
            continue
        candidate_counts = count_severity_dataset(candidate)
        if total_severity_images(candidate_counts) == 0:
            continue
        if SEVERITY_DATASET.exists():
            shutil.rmtree(SEVERITY_DATASET)
        mirror_path(candidate, SEVERITY_DATASET)
        archive_dataset(SEVERITY_DATASET, "severity_dataset")
        print("Restored severity_dataset from:", candidate)
        return "Drive dataset folder"
    return None

def rebuild_severity_from_available_raw_sources() -> str | None:
    config_path = REPO_ROOT / "configs" / "merge_config.yaml"
    if not config_path.exists():
        print("No merge_config.yaml found, so raw-source severity extraction cannot run.")
        return None

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    available = []
    missing = []
    for dataset in config.get("datasets", []):
        raw_root = Path(dataset.get("path", ""))
        if not raw_root.is_absolute():
            raw_root = REPO_ROOT / raw_root
        if (raw_root / "data.yaml").exists():
            available.append(dataset)
        else:
            missing.append(dataset.get("name", str(raw_root)))

    if not available:
        print("Raw-source severity extraction skipped: none of the original raw dataset folders are present in this runtime.")
        return None

    tmp_config = REPO_ROOT / "configs" / "severity_extract_available_sources.yaml"
    tmp_output = REPO_ROOT / "_severity_dataset_extract_tmp"
    extract_config = dict(config)
    extract_config["datasets"] = available
    tmp_config.write_text(yaml.safe_dump(extract_config, sort_keys=False), encoding="utf-8")

    print("Trying severity extraction from available raw sources:")
    print("Available:", [item.get("name", item.get("path")) for item in available])
    if missing:
        print("Missing raw sources skipped:", missing)

    result = run_process(
        [
            sys.executable,
            "scripts/extract_severity_crops.py",
            "--config",
            str(tmp_config),
            "--output",
            str(tmp_output),
            "--force",
        ],
        log_name="revised_extract_severity_crops_from_available_sources",
        allow_failure=True,
    )
    if result.returncode != 0:
        print("Raw-source severity extraction failed; continuing to next recovery path.")
        return None

    tmp_counts = count_severity_dataset(tmp_output)
    print("Raw-source severity counts:")
    print(json.dumps(tmp_counts, indent=2))
    if total_severity_images(tmp_counts) == 0:
        shutil.rmtree(tmp_output, ignore_errors=True)
        return None

    if SEVERITY_DATASET.exists():
        shutil.rmtree(SEVERITY_DATASET)
    shutil.move(str(tmp_output), str(SEVERITY_DATASET))
    archive_dataset(SEVERITY_DATASET, "severity_dataset")
    return "available raw-source annotations"

def read_detection_names(dataset_root: Path) -> list[str]:
    data = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999):
            if not str(key).isdigit():
                continue
            idx = int(key)
            while len(ordered) <= idx:
                ordered.append("")
            ordered[idx] = str(value)
        return ordered
    return [str(name) for name in names]

def find_image_for_label(images_dir: Path, label_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def bbox_from_label_values(values: list[float]) -> tuple[float, float, float, float] | None:
    if len(values) == 4:
        return values[0], values[1], values[2], values[3]
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    return None

def severity_from_detection_class(class_name: str, box: tuple[float, float, float, float]) -> str:
    class_name = class_name.strip().lower()
    area = max(0.0, box[2]) * max(0.0, box[3])
    if class_name == "pothole":
        return "critical"
    if class_name == "spalling":
        return "critical" if area >= 0.075 else "moderate"
    if class_name == "corrosion":
        return "critical" if area >= 0.12 else ("minor" if area <= 0.012 else "moderate")
    if class_name == "paint_degradation":
        return "minor" if area <= 0.018 else "moderate"
    if class_name == "crack":
        return "minor" if area <= 0.008 else "moderate"
    return "moderate"

def crop_xyxy_from_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    xc, yc, bw, bh = box
    x1 = int(round((xc - bw / 2.0) * width))
    y1 = int(round((yc - bh / 2.0) * height))
    x2 = int(round((xc + bw / 2.0) * width))
    y2 = int(round((yc + bh / 2.0) * height))
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

def write_severity_data_yaml(root: Path) -> None:
    payload = {"path": str(root.resolve()), "train": "train", "val": "val", "test": "test", "names": ["minor", "moderate", "critical"]}
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def build_coarse_severity_from_detection_dataset() -> str | None:
    if not ALLOW_COARSE_SEVERITY_BOOTSTRAP:
        print("Coarse severity bootstrap disabled.")
        return None

    try:
        import cv2
    except Exception:
        run_process([sys.executable, "-m", "pip", "install", "opencv-python-headless"], log_name="install_cv2_for_severity_bootstrap")
        import cv2

    source_candidates = [ROBUST_DATASET, ANCHOR_DATASET, MERGED_DATASET]
    source_root = next((Path(root) for root in source_candidates if (Path(root) / "data.yaml").exists()), None)
    if source_root is None:
        print("Coarse severity bootstrap skipped: no merged/anchor/robust detection dataset is available.")
        return None

    names = read_detection_names(source_root)
    samples = {name: [] for name in severity_classes}
    for split in ["train", "val", "test"]:
        labels_dir = source_root / "labels" / split
        images_dir = source_root / "images" / split
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image_for_label(images_dir, label_path)
            if image_path is None:
                continue
            for row_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    values = [float(item) for item in parts[1:]]
                except ValueError:
                    continue
                if class_id < 0 or class_id >= len(names):
                    continue
                box = bbox_from_label_values(values)
                if box is None:
                    continue
                severity = severity_from_detection_class(names[class_id], box)
                samples[severity].append((image_path, box, split, label_path.stem, row_index))

    rng = random.Random(42)
    for items in samples.values():
        rng.shuffle(items)

    availability = {name: len(items) for name, items in samples.items()}
    print("Coarse severity bootstrap sample availability:", availability)
    if any(availability.get(name, 0) < SEVERITY_MIN_TRAIN + SEVERITY_MIN_VAL for name in severity_classes):
        print("Coarse severity bootstrap cannot satisfy the minimum balanced coverage.")
        return None

    if SEVERITY_DATASET.exists():
        shutil.rmtree(SEVERITY_DATASET)
    for split in ["train", "val", "test"]:
        for severity in severity_classes:
            (SEVERITY_DATASET / split / severity).mkdir(parents=True, exist_ok=True)

    max_train_per_class = 1600
    max_val_per_class = 250
    max_test_per_class = 250
    output_counts = {split: {name: 0 for name in severity_classes} for split in ["train", "val", "test"]}

    for severity, items in samples.items():
        selected = items[: max_train_per_class + max_val_per_class + max_test_per_class]
        split_plan = [
            ("train", selected[:max_train_per_class]),
            ("val", selected[max_train_per_class : max_train_per_class + max_val_per_class]),
            ("test", selected[max_train_per_class + max_val_per_class : max_train_per_class + max_val_per_class + max_test_per_class]),
        ]
        if len(split_plan[1][1]) < SEVERITY_MIN_VAL:
            split_plan = [
                ("train", selected[: max(0, len(selected) - 2 * SEVERITY_MIN_VAL)]),
                ("val", selected[max(0, len(selected) - 2 * SEVERITY_MIN_VAL) : max(0, len(selected) - SEVERITY_MIN_VAL)]),
                ("test", selected[max(0, len(selected) - SEVERITY_MIN_VAL) :]),
            ]

        for target_split, split_items in split_plan:
            for image_path, box, source_split, stem, row_index in split_items:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                height, width = image.shape[:2]
                x1, y1, x2, y2 = crop_xyxy_from_yolo(box, width, height)
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 12 or (y2 - y1) < 12:
                    continue
                crop = image[y1:y2, x1:x2]
                token = hashlib.sha1(f"{image_path}|{row_index}|{target_split}|{severity}".encode("utf-8")).hexdigest()[:12]
                output_path = SEVERITY_DATASET / target_split / severity / f"coarse_{severity}_{token}.jpg"
                if cv2.imwrite(str(output_path), crop):
                    output_counts[target_split][severity] += 1

    write_severity_data_yaml(SEVERITY_DATASET)
    summary = {
        "source": "coarse_detection_bootstrap",
        "source_detection_dataset": str(source_root),
        "warning": "These severity labels are weak labels derived from defect class and bounding-box area, not true inspection severity annotations.",
        "counts": output_counts,
    }
    (SEVERITY_DATASET / "severity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    archive_dataset(SEVERITY_DATASET, "severity_dataset")
    print("Built coarse severity_dataset from detection labels:")
    print(json.dumps(summary, indent=2))
    return "coarse detection-label bootstrap"

counts = count_severity_dataset(SEVERITY_DATASET)
severity_source = "existing local severity_dataset"
used_coarse_bootstrap = False
print("Initial severity_dataset counts:")
print(json.dumps(counts, indent=2))

if not severity_is_ready(counts):
    if total_severity_images(counts) == 0:
        print("Why zero? This runtime has no usable severity crop folders yet. The revised notebook restored detection datasets, but Cell 16 must separately recover or rebuild severity_dataset.")
    recovered = restore_severity_dataset_from_persistent_storage()
    if recovered:
        severity_source = recovered
        counts = count_severity_dataset(SEVERITY_DATASET)
        print("Counts after severity dataset restore:")
        print(json.dumps(counts, indent=2))

if not severity_is_ready(counts):
    rebuilt = rebuild_severity_from_available_raw_sources()
    if rebuilt:
        severity_source = rebuilt
        counts = count_severity_dataset(SEVERITY_DATASET)
        print("Counts after raw-source severity extraction:")
        print(json.dumps(counts, indent=2))

if not severity_is_ready(counts):
    coarse = build_coarse_severity_from_detection_dataset()
    if coarse:
        severity_source = coarse
        used_coarse_bootstrap = True
        counts = count_severity_dataset(SEVERITY_DATASET)
        print("Counts after coarse severity bootstrap:")
        print(json.dumps(counts, indent=2))

severity_ready = severity_is_ready(counts)
existing_severity = find_existing_severity_weight()

if not severity_ready:
    if existing_severity is not None:
        sync_existing_severity_weight(existing_severity, "existing model fallback because severity crops are not trainable")
        mark_done(
            "revised_severity_checked",
            {
                "counts": counts,
                "trainable": False,
                "retrained": False,
                "fallback_weight": str(existing_severity),
                "reason": "severity_dataset_missing_or_insufficient",
            },
        )
    else:
        block_or_raise(
            "revised_severity_training",
            "Severity crop dataset is still insufficient and no existing severity classifier weight was found.",
            {"counts": counts, "severity_dataset": str(SEVERITY_DATASET), "expected_weight": str(SEVERITY_WEIGHT)},
        )
elif used_coarse_bootstrap and not TRAIN_COARSE_SEVERITY_BOOTSTRAP and existing_severity is not None:
    print(
        "Severity crops were created with coarse weak labels. To avoid spending compute on noisy severity labels, "
        "this cell will reuse the existing severity model. Set TRAIN_COARSE_SEVERITY_BOOTSTRAP=True only if you intentionally want to train Stage 2 from weak labels."
    )
    sync_existing_severity_weight(existing_severity, "existing model fallback with coarse severity dataset available")
    mark_done(
        "revised_severity_checked",
        {
            "counts": counts,
            "trainable": True,
            "retrained": False,
            "severity_source": severity_source,
            "coarse_bootstrap": True,
            "fallback_weight": str(existing_severity),
        },
    )
elif not ALLOW_SEVERITY_RETRAIN:
    if existing_severity is not None:
        sync_existing_severity_weight(existing_severity, "existing model fallback because ALLOW_SEVERITY_RETRAIN=False")
        mark_done(
            "revised_severity_checked",
            {"counts": counts, "trainable": True, "retrained": False, "severity_source": severity_source, "fallback_weight": str(existing_severity)},
        )
    else:
        block_or_raise("revised_severity_training", "ALLOW_SEVERITY_RETRAIN=False and no existing severity weight was found.")
else:
    run_name = "stage2_revised_severity_yolo11s_cls"
    run_process(
        [
            "yolo",
            "classify",
            "train",
            "model=yolo11s-cls.pt",
            f"data={SEVERITY_DATASET}",
            "epochs=60",
            "imgsz=224",
            "batch=32",
            "patience=15",
            "optimizer=AdamW",
            "lr0=0.001",
            "cos_lr=True",
            f"project={DRIVE_RUNS_ROOT}",
            f"name={run_name}",
            "exist_ok=True",
        ],
        log_name="revised_train_severity_classifier",
    )
    best = DRIVE_RUNS_ROOT / run_name / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, SEVERITY_WEIGHT)
        shutil.copy2(best, WEIGHTS_DIR / "severity_cls.pt")
        mark_done(
            "revised_severity_checked",
            {"counts": counts, "trainable": True, "retrained": True, "severity_source": severity_source, "weight": str(SEVERITY_WEIGHT)},
        )
    else:
        block_or_raise("revised_severity_training", f"Severity training finished but best.pt is missing: {best}", {"best": str(best)})

if SEVERITY_WEIGHT.exists():
    print(f"Severity weight ready for Cell 17: {SEVERITY_WEIGHT}")
else:
    raise RuntimeError(f"Severity classifier missing after Cell 16 recovery: {SEVERITY_WEIGHT}")
'''


PRODUCTION_CHECK_CODE = r'''
if not DETECTOR_WEIGHT.exists():
    raise RuntimeError(f"Current detector missing: {DETECTOR_WEIGHT}")
if not ANCHOR_CANDIDATE_WEIGHT.exists():
    raise RuntimeError(f"Revised candidate missing: {ANCHOR_CANDIDATE_WEIGHT}")
if not SEVERITY_WEIGHT.exists():
    raise RuntimeError(f"Severity classifier missing: {SEVERITY_WEIGHT}")

check_dir = DRIVE_OUTPUT_ROOT / "revised_training_production_check"
comparison_path = check_dir / "comparison_summary.json"
candidate_summary = check_dir / "anchor" / "domain_sweep_summary.csv"
current_summary = check_dir / "current" / "domain_sweep_summary.csv"

needs_sweep = FORCE_REVISED_SWEEP or not comparison_path.exists() or not current_summary.exists() or not candidate_summary.exists()
if needs_sweep:
    if not ALLOW_UNCACHED_SWEEP_RUNS and not FORCE_REVISED_SWEEP:
        raise RuntimeError(
            "Production sweep outputs are missing. Set ALLOW_UNCACHED_SWEEP_RUNS=True or FORCE_REVISED_SWEEP=True "
            "to run the visual production check."
        )
    run_process(
        [
            sys.executable,
            "scripts/wider_production_sweep.py",
            "--base-manifest",
            "configs/production_eval_manifest.csv",
            "--output",
            str(check_dir),
            "--models",
            f"current={DETECTOR_WEIGHT},anchor={ANCHOR_CANDIDATE_WEIGHT}",
            "--severity",
            str(SEVERITY_WEIGHT),
            "--per-group",
            "7",
            "--max-per-query",
            "14",
            "--thresholds",
            "0.45,0.30,0.20,0.10",
            "--annotate-conf",
            "0.20",
            "--scenario-variants",
            "original,low_light,overexposed,blur_distance,shadow,occlusion",
        ],
        log_name="revised_wider_production_sweep",
    )
else:
    print(f"Using cached revised production check: {comparison_path}")

gate_path = DRIVE_OUTPUT_ROOT / "summaries" / "revised_anchor_production_gate.json"
gate_result = run_process(
    [
        sys.executable,
        "scripts/revised_production_gate.py",
        "--current-summary",
        str(current_summary),
        "--candidate-summary",
        str(candidate_summary),
        "--output",
        str(gate_path),
        "--threshold",
        "0.2",
        "--min-evaluation-rows",
        "30",
        "--min-candidate-pass-rate",
        "0.75",
        "--min-domain-pass-rate",
        "0.70",
        "--min-expected-pass-rate",
        "0.70",
        "--min-bucket-samples",
        "3",
        "--min-pass-rate-delta",
        "0.03",
        "--allowed-class-regression",
        "0",
        "--max-false-positive-increase",
        "0",
    ],
    log_name="revised_production_gate",
    allow_failure=True,
)

decision = json.loads(gate_path.read_text(encoding="utf-8"))
if decision.get("promote") and PROMOTE_REVISED_MODEL:
    shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, DETECTOR_WEIGHT)
    shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, ANCHOR_PROMOTED_WEIGHT)
    shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, WEIGHTS_DIR / "defect_detector.pt")
    print(f"Promoted revised detector to canonical deployment weight: {DETECTOR_WEIGHT}")
elif decision.get("promote"):
    print("Candidate passed the revised gate, but PROMOTE_REVISED_MODEL=False, so no deployment weight was overwritten.")
else:
    print("Candidate failed the revised gate; deployment weight was not changed.")

print(f"Visual report: {check_dir / 'visual_report.html'}")
print(f"Gate report: {gate_path}")
mark_done("revised_production_check_done", {"gate": str(gate_path), "visual_report": str(check_dir / "visual_report.html")})
'''


TARGET_SOURCE_COLLECTION_CODE = r'''
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
target_summary = TARGET_YOLO / "auto_collection_summary.json"

import importlib.util
import site


def refresh_python_package_paths():
    candidate_paths = []
    try:
        candidate_paths.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        candidate_paths.extend(site.getsitepackages())
    except Exception:
        pass
    for candidate in candidate_paths:
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
    importlib.invalidate_caches()


def ensure_importable_package(package_name, import_name=None):
    import_name = import_name or package_name.replace("-", "_")
    refresh_python_package_paths()
    if importlib.util.find_spec(import_name) is not None:
        return
    print(f"Installing {package_name} for target-domain collection...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", package_name],
        check=True,
    )
    refresh_python_package_paths()
    if importlib.util.find_spec(import_name) is None:
        raise ModuleNotFoundError(
            f"{package_name} installed but {import_name} is still not importable. "
            "Restart the Colab runtime once, rerun setup/materialization, then rerun this cell."
        )


for package_name, import_name in [
    ("filetype", "filetype"),
    ("pillow-avif-plugin", "pillow_avif"),
    ("python-dotenv", "dotenv"),
    ("requests-toolbelt", "requests_toolbelt"),
    ("roboflow", "roboflow"),
]:
    ensure_importable_package(package_name, import_name)

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

collector_text = (REPO_ROOT / "scripts" / "auto_collect_domain_sources.py").read_text(encoding="utf-8")
if "SOURCE_SPECIFIC_LABEL_MAP" not in collector_text or "Building Damage Insurance Wall Defects" not in collector_text:
    raise RuntimeError(
        "Cell 4 needs the refreshed source-specific label mapper. "
        "Rerun Cell 2 once to materialize the updated scripts, then rerun Cell 4. "
        "Cell 2 is CPU-only and fast; it will not rerun training."
    )


def target_summary_is_usable(path):
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    counts = summary.get("counts") or {}
    mapped_labels = sum(int(value) for key, value in counts.items() if str(key).startswith("mapped/"))
    mapped_images = sum(int(value) for key, value in counts.items() if str(key).startswith("images/"))
    sources = summary.get("sources") or []
    usable_sources = [
        source for source in sources
        if str(source.get("status", "")).startswith(("downloaded", "cached"))
        and (source.get("counts") or {})
    ]
    return mapped_labels > 0 and mapped_images > 0 and len(usable_sources) > 0


should_rebuild_target = FORCE_DOMAIN_COLLECTION or not target_summary_is_usable(target_summary)
if target_summary.exists() and should_rebuild_target:
    print("Existing target-domain summary is missing usable mapped data; rebuilding Cell 4 outputs.")
    print(target_summary.read_text(encoding="utf-8")[-2000:])

if should_rebuild_target:
    if not ROBOFLOW_API_KEY:
        block_or_raise(
            "revised_target_sources",
            "ROBOFLOW_API_KEY is missing, so target-domain collection cannot run.",
            {"target": str(TARGET_YOLO)},
        )
    else:
        run_process(
            [
                sys.executable,
                "scripts/auto_collect_domain_sources.py",
                "--api-key",
                ROBOFLOW_API_KEY,
                "--raw-output",
                "domain_adaptation/auto_raw",
                "--target-output",
                str(TARGET_YOLO),
                "--force-download",
                "--force-rebuild",
            ],
            log_name="revised_collect_target_domain",
            allow_failure=not REQUIRE_FULL_REVISED_RUN,
        )
else:
    print(f"Using cached target-domain collection: {target_summary}")

if not target_summary.exists():
    block_or_raise(
        "revised_target_sources",
        f"Target-domain collection summary missing: {target_summary}",
        {"target": str(TARGET_YOLO)},
    )
elif not target_summary_is_usable(target_summary):
    summary = json.loads(target_summary.read_text(encoding="utf-8"))
    failed_sources = [
        {"folder": source.get("folder"), "status": source.get("status")}
        for source in summary.get("sources", [])
        if not str(source.get("status", "")).startswith(("downloaded", "cached"))
    ]
    block_or_raise(
        "revised_target_sources",
        "Target-domain collection finished, but no usable mapped labels/images were produced.",
        {
            "counts": summary.get("counts", {}),
            "failed_sources": failed_sources[:12],
            "hint": "Rerun this Cell 4 once after dependency repair. If it still fails, the Roboflow source/API access is the blocker.",
        },
    )
else:
    print(target_summary.read_text(encoding="utf-8")[-6000:])
    mark_done("revised_target_sources_collected", {"target": str(TARGET_YOLO), "summary": str(target_summary)})
'''


LABEL_MAPPING_AUDIT_CODE = r'''
label_mapping_summary_path = DRIVE_OUTPUT_ROOT / "summaries" / "label_mapping_coverage_summary.json"
run_process(
    [
        sys.executable,
        "scripts/audit_label_mapping_coverage.py",
        "--merge-config", "configs/merge_config.yaml",
        "--target-raw", "domain_adaptation/auto_raw",
        "--target-yolo", "domain_adaptation/target_yolo",
        "--canonical-dataset", "merged_dataset",
        "--canonical-dataset", "merged_dataset_anchor_balanced",
        "--canonical-dataset", "merged_dataset_anchor_robust",
        "--output", str(label_mapping_summary_path),
    ],
    log_name="revised_label_mapping_coverage_audit",
    allow_failure=not REQUIRE_FULL_REVISED_RUN,
)

if not label_mapping_summary_path.exists():
    block_or_raise(
        "revised_label_mapping_coverage",
        f"Label mapping audit did not produce a summary: {label_mapping_summary_path}",
        {"summary": str(label_mapping_summary_path)},
    )

label_mapping_summary = json.loads(label_mapping_summary_path.read_text(encoding="utf-8"))
print("Label mapping coverage ok:", label_mapping_summary.get("ok"))
print("Label mapping errors:", label_mapping_summary.get("errors", [])[:20])
print(json.dumps(label_mapping_summary, indent=2)[-8000:])
if label_mapping_summary.get("ok"):
    mark_done("revised_label_mapping_coverage_audited", {"summary": str(label_mapping_summary_path)})
else:
    block_or_raise(
        "revised_label_mapping_coverage",
        "Label mapping coverage audit failed. Fix unmapped source labels before strict full retraining.",
        {"summary": str(label_mapping_summary_path), "errors": label_mapping_summary.get("errors", [])[:20]},
    )
'''


HARD_NEGATIVE_PREP_CODE = r'''
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"
BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"

HARD_NEGATIVES.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    (HARD_NEGATIVES / split / "images").mkdir(parents=True, exist_ok=True)
    (HARD_NEGATIVES / split / "labels").mkdir(parents=True, exist_ok=True)

resolved_manifest = BASELINE_SWEEP_DIR / "resolved_manifest.csv"
copied = 0
if resolved_manifest.exists():
    with resolved_manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("expected_class") != "none":
                continue
            src = Path(row.get("resolved_path", ""))
            if not src.exists():
                continue
            dst = HARD_NEGATIVES / "train" / "images" / f"hardneg_{src.name}"
            label = HARD_NEGATIVES / "train" / "labels" / f"{dst.stem}.txt"
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            label.write_text("", encoding="utf-8")
else:
    print(f"No cached baseline sweep manifest found at {resolved_manifest}; continuing without sweep hard negatives.")

print(f"Hard-negative image files available: {sum(1 for p in (HARD_NEGATIVES / 'train' / 'images').glob('*') if p.is_file())}")
print(f"New hard-negative images copied in this run: {copied}")
mark_done("revised_hard_negatives_ready", {"hard_negatives": str(HARD_NEGATIVES), "copied": copied})
'''


ANCHOR_PREFLIGHT_CODE = r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"
ANCHOR_AUDIT_DIR = REPO_ROOT / "output" / "_anchor_dataset_preflight_audit"
preflight_drive_summary = DRIVE_OUTPUT_ROOT / "summaries" / "anchor_balanced_preflight_audit_summary.json"

if ANCHOR_AUDIT_DIR.exists():
    shutil.rmtree(ANCHOR_AUDIT_DIR)

baseline_ok = dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1)
anchor_ready = dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30)
if not baseline_ok and anchor_ready:
    recovered_summary = {
        "audit_only": True,
        "recovered_anchor_dataset": str(ANCHOR_DATASET),
        "reason": "baseline_merged_dataset_missing_but_anchor_dataset_recovered",
        "class_counts_by_split": dataset_counts_from_labels(ANCHOR_DATASET),
        "guard": {
            "ok": True,
            "source": "recovered_anchor_dataset",
            "missing_or_too_low_train": [],
            "missing_or_too_low_val": [],
            "train_max_min_ratio": None,
            "max_allowed_train_ratio": 6,
        },
        "quality_filter": {
            "target": {
                "keep_rate": None,
                "rejected_reasons": {},
                "note": "Preflight rebuild skipped because baseline merged_dataset is unavailable and recovered anchor dataset is already all-class ready.",
            }
        },
    }
    preflight_drive_summary.parent.mkdir(parents=True, exist_ok=True)
    preflight_drive_summary.write_text(json.dumps(recovered_summary, indent=2), encoding="utf-8")
    print("Using recovered anchor-balanced dataset; preflight rebuild skipped.")
    print(json.dumps(recovered_summary, indent=2)[-6000:])
elif not baseline_ok:
    blocked_summary = {
        "audit_only": True,
        "blocked": True,
        "reason": "baseline_merged_dataset_missing_and_no_recovered_anchor",
        "baseline_dataset": str(MERGED_DATASET),
        "anchor_dataset": str(ANCHOR_DATASET),
        "guard": {"ok": False, "source": "recovery_preflight"},
    }
    preflight_drive_summary.parent.mkdir(parents=True, exist_ok=True)
    preflight_drive_summary.write_text(json.dumps(blocked_summary, indent=2), encoding="utf-8")
    block_or_raise(
        "revised_anchor_preflight",
        "Cannot build anchor-balanced dataset because baseline merged_dataset is missing and no valid recovered anchor dataset exists.",
        {"summary": str(preflight_drive_summary), "reason": blocked_summary["reason"]},
    )
else:
    preflight_args = [
        sys.executable,
        "scripts/build_anchor_balanced_dataset.py",
        "--base", str(MERGED_DATASET),
        "--target", str(TARGET_YOLO),
        "--hard-negatives", str(HARD_NEGATIVES),
        "--output", str(ANCHOR_AUDIT_DIR),
        "--target-box-goal-per-class", "1500",
        "--max-target-images-per-class", "900",
        "--max-target-train-images", "3600",
        "--target-val-images-per-class", "35",
        "--oversample-target-boxes", "6500",
        "--max-repeat-per-image", "4",
        "--max-hard-negative-train", "450",
        "--max-hard-negative-val", "100",
        "--min-image-side", "96",
        "--min-blur-variance", "4.0",
        "--max-box-area", "0.92",
        "--max-boxes-per-image", "90",
        "--max-target-classes-per-image", "3",
        "--max-invalid-label-fraction", "0.0",
        "--min-quality-target-boxes-per-class", "25",
        "--min-target-keep-rate", "0.75",
        "--min-final-train-boxes-per-class", "1200",
        "--min-final-val-boxes-per-class", "30",
        "--max-final-class-ratio", "6",
        "--audit-only",
    ]
    run_process(preflight_args, log_name="revised_anchor_dataset_preflight_audit")

    audit_summary_path = ANCHOR_AUDIT_DIR / "anchor_balanced_audit_summary.json"
    if not audit_summary_path.exists():
        raise RuntimeError(f"Preflight audit did not produce expected summary: {audit_summary_path}")
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    mirror_path(audit_summary_path, preflight_drive_summary)
    print("Target keep rate:", audit_summary["quality_filter"]["target"]["keep_rate"])
    print("Target rejected reasons:", audit_summary["quality_filter"]["target"]["rejected_reasons"])
    print("Final guard:", audit_summary["guard"])
    shutil.rmtree(ANCHOR_AUDIT_DIR, ignore_errors=True)

mark_done("revised_anchor_preflight_done", {"summary": str(preflight_drive_summary)})
'''


ANCHOR_BUILD_CODE = r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"

summary_path = ANCHOR_DATASET / "anchor_balanced_summary.json"
needs_build = FORCE_REBUILD_ANCHOR_DATASET or not summary_path.exists() or not dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30)
if needs_build:
    if not dataset_has_all_classes(MERGED_DATASET, min_train=1, min_val=1):
        if dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30):
            print(f"Using recovered anchor-balanced dataset because baseline merged_dataset is unavailable: {ANCHOR_DATASET}")
        else:
            block_or_raise(
                "revised_anchor_dataset",
                "Cannot rebuild anchor-balanced dataset because baseline merged_dataset is missing. Recover Drive dataset cache or rebuild baseline first.",
                {"baseline_dataset": str(MERGED_DATASET), "anchor_dataset": str(ANCHOR_DATASET)},
            )
    else:
        args = [
            sys.executable,
            "scripts/build_anchor_balanced_dataset.py",
            "--base", str(MERGED_DATASET),
            "--target", str(TARGET_YOLO),
            "--hard-negatives", str(HARD_NEGATIVES),
            "--output", str(ANCHOR_DATASET),
            "--target-box-goal-per-class", "1500",
            "--max-target-images-per-class", "900",
            "--max-target-train-images", "3600",
            "--target-val-images-per-class", "35",
            "--oversample-target-boxes", "6500",
            "--max-repeat-per-image", "4",
            "--max-hard-negative-train", "450",
            "--max-hard-negative-val", "100",
            "--min-image-side", "96",
            "--min-blur-variance", "4.0",
            "--max-box-area", "0.92",
            "--max-boxes-per-image", "90",
            "--max-target-classes-per-image", "3",
            "--max-invalid-label-fraction", "0.0",
            "--min-quality-target-boxes-per-class", "25",
            "--min-target-keep-rate", "0.75",
            "--min-final-train-boxes-per-class", "1200",
            "--min-final-val-boxes-per-class", "30",
            "--max-final-class-ratio", "6",
        ]
        if FORCE_REBUILD_ANCHOR_DATASET:
            args.append("--force")
        run_process(args, log_name="revised_build_anchor_balanced_dataset")
else:
    print(f"Using cached anchor-balanced dataset: {summary_path}")

anchor_ready = dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30)
if summary_path.exists():
    print(summary_path.read_text(encoding="utf-8")[-7000:])
elif anchor_ready:
    print("Anchor-balanced summary missing, but dataset counts are valid; writing recovery summary.")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "recovered": True,
                "output": str(ANCHOR_DATASET),
                "names": REQUIRED_DETECTION_CLASSES,
                "class_counts_by_split": dataset_counts_from_labels(ANCHOR_DATASET),
                "guard": {"ok": True, "source": "recovered_anchor_dataset"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
else:
    block_or_raise(
        "revised_anchor_dataset",
        f"Anchor-balanced dataset is not ready, so no summary is available: {summary_path}",
        {"anchor_dataset": str(ANCHOR_DATASET)},
    )

if anchor_ready:
    archive_dataset(ANCHOR_DATASET, "merged_dataset_anchor_balanced")
    if MIRROR_FULL_DATASET_FOLDERS:
        mirror_path(ANCHOR_DATASET, DRIVE_DATASETS_ROOT / "merged_dataset_anchor_balanced")
    mark_done("revised_anchor_dataset_ready", {"anchor_dataset": str(ANCHOR_DATASET), "summary": str(summary_path)})
'''


ROBUST_DATASET_BUILD_CODE = r'''
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"

robust_summary_path = ROBUST_DATASET / "robust_augmentation_summary.json"
needs_build = FORCE_REBUILD_ANCHOR_DATASET or not robust_summary_path.exists() or not dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30)
if needs_build:
    anchor_ready = dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30)
    robust_ready = dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30)
    if not anchor_ready and robust_ready:
        print(f"Using recovered robust augmented dataset because anchor dataset is unavailable: {ROBUST_DATASET}")
    elif not anchor_ready:
        block_or_raise(
            "revised_robust_dataset",
            "Cannot build robust augmented dataset because anchor-balanced dataset is missing.",
            {"anchor_dataset": str(ANCHOR_DATASET), "robust_dataset": str(ROBUST_DATASET)},
        )
    else:
        run_process(
            [
                sys.executable,
                "scripts/build_robust_augmented_dataset.py",
                "--input", str(ANCHOR_DATASET),
                "--output", str(ROBUST_DATASET),
                "--aug-box-goal-per-class", "1800",
                "--max-aug-images-per-class", "650",
                "--max-negative-aug-images", "240",
                "--max-occlusion-box-overlap", "0.35",
                "--scale-space-crops-per-class", "320",
                "--scale-space-contexts", "1.35,2.20,3.40",
                "--scale-space-min-visible-fraction", "0.55",
                "--min-train-boxes-per-class", "1200",
                "--min-val-boxes-per-class", "30",
                "--max-train-class-ratio", "6",
            ],
            log_name="revised_build_robust_augmented_dataset",
        )
else:
    print(f"Using cached robust augmented dataset: {robust_summary_path}")

if robust_summary_path.exists():
    print(robust_summary_path.read_text(encoding="utf-8")[-7000:])
elif dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30):
    print("Robust augmentation summary missing, but dataset counts are valid; writing recovery summary.")
    robust_summary_path.parent.mkdir(parents=True, exist_ok=True)
    robust_summary_path.write_text(
        json.dumps(
            {
                "recovered": True,
                "output": str(ROBUST_DATASET),
                "names": REQUIRED_DETECTION_CLASSES,
                "class_counts_by_split": dataset_counts_from_labels(ROBUST_DATASET),
                "guard": {"ok": True, "source": "recovered_robust_dataset"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
else:
    block_or_raise(
        "revised_robust_dataset",
        f"Robust dataset is not ready, so no robust summary is available: {robust_summary_path}",
        {"robust_dataset": str(ROBUST_DATASET)},
    )

if dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30):
    archive_dataset(ROBUST_DATASET, "merged_dataset_anchor_robust")
    if MIRROR_FULL_DATASET_FOLDERS:
        mirror_path(ROBUST_DATASET, DRIVE_DATASETS_ROOT / "merged_dataset_anchor_robust")
    mark_done("revised_robust_dataset_ready", {"robust_dataset": str(ROBUST_DATASET), "summary": str(robust_summary_path)})
'''


FEATURE_COVERAGE_AUDIT_CODE = r'''
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
feature_summary_path = DRIVE_OUTPUT_ROOT / "summaries" / "robust_feature_coverage_summary.json"

if not dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30):
    block_or_raise(
        "revised_feature_coverage",
        "Robust dataset is missing or incomplete, so feature coverage audit cannot run.",
        {"robust_dataset": str(ROBUST_DATASET), "feature_summary": str(feature_summary_path)},
    )
else:
    run_process(
        [
            sys.executable,
            "scripts/audit_defect_feature_coverage.py",
            "--dataset", str(ROBUST_DATASET),
            "--output", str(feature_summary_path),
            "--min-train-boxes-per-class", "1200",
            "--min-val-boxes-per-class", "30",
            "--max-train-class-ratio", "6",
            "--min-feature-bins", "2",
            "--min-bin-boxes", "40",
        ],
        log_name="revised_audit_feature_coverage",
        allow_failure=not REQUIRE_FULL_REVISED_RUN,
    )
    if not feature_summary_path.exists():
        block_or_raise(
            "revised_feature_coverage",
            f"Feature coverage audit did not produce a summary: {feature_summary_path}",
            {"feature_summary": str(feature_summary_path)},
        )

feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
print("Feature coverage guard:", feature_summary.get("guard"))
print("Feature coverage warnings:", feature_summary.get("warnings", [])[:20])
print(json.dumps(feature_summary, indent=2)[-7000:])
if feature_summary.get("guard", {}).get("ok") is True:
    mark_done("revised_feature_coverage_audited", {"feature_summary": str(feature_summary_path)})
else:
    block_or_raise(
        "revised_feature_coverage",
        "Feature coverage guard failed. Strict full training will not start with visually narrow data.",
        {"feature_summary": str(feature_summary_path), "guard": feature_summary.get("guard")},
    )
'''


DATASET_PREP_ASSURANCE_CODE = r'''
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"

summary_paths = {
    "target_collection": TARGET_YOLO / "auto_collection_summary.json",
    "label_mapping": DRIVE_OUTPUT_ROOT / "summaries" / "label_mapping_coverage_summary.json",
    "anchor_preflight": DRIVE_OUTPUT_ROOT / "summaries" / "anchor_balanced_preflight_audit_summary.json",
    "anchor_dataset": ANCHOR_DATASET / "anchor_balanced_summary.json",
    "robust_dataset": ROBUST_DATASET / "robust_augmentation_summary.json",
    "feature_coverage": DRIVE_OUTPUT_ROOT / "summaries" / "robust_feature_coverage_summary.json",
}

missing = [name for name, path in summary_paths.items() if not path.exists()]
if missing:
    print(f"SKIP: Dataset preparation summaries are missing: {missing}. Training will be blocked before GPU use.")
    fallback_summaries = {
        "target_collection": {"counts": {}, "sources": [], "blocked": True, "reason": "summary_missing"},
        "label_mapping": {"ok": False, "errors": ["summary_missing"], "blocked": True},
        "anchor_preflight": {"guard": {"ok": False, "source": "summary_missing"}, "blocked": True},
        "anchor_dataset": {
            "names": REQUIRED_DETECTION_CLASSES,
            "anchor_guard": {"ok": False, "source": "summary_missing"},
            "target_quality_guard": {"ok": False, "source": "summary_missing"},
            "guard": {"ok": False, "source": "summary_missing"},
            "class_counts_by_split": dataset_counts_from_labels(ANCHOR_DATASET) if ANCHOR_DATASET.exists() else {},
        },
        "robust_dataset": {
            "names": REQUIRED_DETECTION_CLASSES,
            "guard": {"ok": False, "source": "summary_missing"},
            "class_counts_by_split": dataset_counts_from_labels(ROBUST_DATASET) if ROBUST_DATASET.exists() else {},
            "variant_counts": {},
            "scale_space": {"created": 0, "class_counts": {}},
        },
        "feature_coverage": {"guard": {"ok": False, "warning_count": 1, "source": "summary_missing"}, "warnings": ["summary_missing"]},
    }
    for name in missing:
        path = summary_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fallback_summaries[name], indent=2), encoding="utf-8")
    block_or_raise(
        "revised_dataset_preparation",
        f"Dataset preparation summaries are missing: {missing}. Strict full run cannot train until every preparation cell has completed.",
        {"missing": missing, "summary_paths": {name: str(path) for name, path in summary_paths.items()}},
    )

summaries = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in summary_paths.items()
}

errors = []
warnings = []

def require(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)

def warn(condition: bool, message: str) -> None:
    if not condition:
        warnings.append(message)

required_names = REQUIRED_DETECTION_CLASSES
target_data_yaml = TARGET_YOLO / "data.yaml"
require(target_data_yaml.exists(), f"Converted target dataset is missing data.yaml: {target_data_yaml}")
target_label_counts = {name: 0 for name in required_names}
target_label_errors = []
if target_data_yaml.exists():
    with target_data_yaml.open("r", encoding="utf-8") as handle:
        target_data = yaml.safe_load(handle) or {}
    target_names = target_data.get("names", [])
    if isinstance(target_names, dict):
        ordered_target_names = []
        for key, value in sorted(target_names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999):
            if str(key).isdigit():
                index = int(key)
                while len(ordered_target_names) <= index:
                    ordered_target_names.append("")
                ordered_target_names[index] = str(value)
        target_names = ordered_target_names
    target_names = [str(name) for name in target_names]
    require(target_names == required_names, f"Converted target data.yaml class order changed: {target_names}")

    for split in ["train", "valid", "val", "test"]:
        labels_dir = TARGET_YOLO / split / "labels"
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    target_label_errors.append(f"{label_path}:{line_number}: malformed label with {len(parts)} fields")
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x, y, width, height = [float(item) for item in parts[1:]]
                except ValueError:
                    target_label_errors.append(f"{label_path}:{line_number}: non-numeric label")
                    continue
                if not 0 <= class_id < len(required_names):
                    target_label_errors.append(f"{label_path}:{line_number}: class id out of range {class_id}")
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    target_label_errors.append(f"{label_path}:{line_number}: box out of normalized YOLO range")
                    continue
                target_label_counts[required_names[class_id]] += 1

target_counts = summaries["target_collection"].get("counts", {})
target_mapped = sum(int(value) for key, value in target_counts.items() if str(key).startswith("mapped/"))
downloaded_or_cached = [
    source for source in summaries["target_collection"].get("sources", [])
    if str(source.get("status", "")).startswith(("downloaded", "cached"))
]
require(target_mapped > 0, "Target-domain collection produced zero mapped YOLO labels.")
require(not target_label_errors, f"Converted target label errors found: {target_label_errors[:10]}")
for class_name in required_names:
    require(target_label_counts[class_name] > 0, f"Converted target labels contain zero boxes for {class_name}.")
warn(len(downloaded_or_cached) >= 4, f"Only {len(downloaded_or_cached)} target sources were downloaded/cached.")

preflight = summaries["anchor_preflight"]
label_mapping = summaries["label_mapping"]
anchor = summaries["anchor_dataset"]
robust = summaries["robust_dataset"]
feature = summaries["feature_coverage"]

require(label_mapping.get("ok") is True, f"Label mapping coverage audit failed: {label_mapping.get('errors')}")
require(anchor.get("names") == required_names, f"Anchor class names changed: {anchor.get('names')}")
require(robust.get("names") == required_names, f"Robust class names changed: {robust.get('names')}")
require(preflight.get("guard", {}).get("ok") is True, f"Preflight guard failed: {preflight.get('guard')}")
require(anchor.get("anchor_guard", {}).get("ok") is True, f"Baseline anchor guard failed: {anchor.get('anchor_guard')}")
require(anchor.get("target_quality_guard", {}).get("ok") is True, f"Target quality guard failed: {anchor.get('target_quality_guard')}")
require(anchor.get("guard", {}).get("ok") is True, f"Anchor final guard failed: {anchor.get('guard')}")
require(robust.get("guard", {}).get("ok") is True, f"Robust final guard failed: {robust.get('guard')}")
require(feature.get("guard", {}).get("ok") is True, f"Feature coverage guard failed: {feature.get('guard')}")
require(feature.get("guard", {}).get("warning_count", 0) == 0, f"Feature coverage warnings remain: {feature.get('warnings', [])[:10]}")

for dataset_name, summary, min_train, min_val in [
    ("anchor", anchor, 1200, 30),
    ("robust", robust, 1200, 30),
]:
    counts = summary.get("class_counts_by_split", {})
    for class_name in required_names:
        train_count = int(counts.get("train", {}).get(class_name, 0))
        val_count = int(counts.get("val", {}).get(class_name, 0))
        require(train_count >= min_train, f"{dataset_name} train {class_name} too low: {train_count} < {min_train}")
        require(val_count >= min_val, f"{dataset_name} val {class_name} too low: {val_count} < {min_val}")
    train_values = [int(counts.get("train", {}).get(class_name, 0)) for class_name in required_names]
    if min(train_values or [0]) > 0:
        ratio = max(train_values) / min(train_values)
        require(ratio <= 6.0, f"{dataset_name} train class ratio too high: {ratio:.2f} > 6.0")

variant_counts = robust.get("variant_counts", {})
for variant in ["low_light", "overexposed", "shadow", "blur_distance", "occlusion", "edge_emphasis", "perspective_view"]:
    require(int(variant_counts.get(variant, 0)) > 0, f"Robust dataset missing scenario variant: {variant}")
scale_space = robust.get("scale_space", {})
require(int(scale_space.get("created", 0)) > 0, "Robust dataset created zero scale-space crops.")
for class_name in required_names:
    require(int(scale_space.get("class_counts", {}).get(class_name, 0)) > 0, f"Scale-space crops missing class: {class_name}")

assurance = {
    "ok": not errors,
    "errors": errors,
    "warnings": warnings,
    "target_mapped_labels": target_mapped,
    "target_label_counts": target_label_counts,
    "target_label_errors": target_label_errors[:50],
    "target_sources_downloaded_or_cached": len(downloaded_or_cached),
    "anchor_guard": anchor.get("guard"),
    "robust_guard": robust.get("guard"),
    "feature_guard": feature.get("guard"),
    "anchor_counts": anchor.get("class_counts_by_split"),
    "robust_counts": robust.get("class_counts_by_split"),
    "robust_variant_counts": variant_counts,
    "scale_space": scale_space,
}

assurance_path = DRIVE_OUTPUT_ROOT / "summaries" / "dataset_preparation_assurance_summary.json"
assurance_path.parent.mkdir(parents=True, exist_ok=True)
assurance_path.write_text(json.dumps(assurance, indent=2), encoding="utf-8")
print(json.dumps(assurance, indent=2)[-8000:])

if errors:
    block_or_raise(
        "revised_dataset_preparation",
        "Dataset preparation assurance gate failed. Strict full run is blocked before GPU use.",
        {"summary": str(assurance_path), "errors": errors[:20]},
    )
else:
    print(f"Dataset preparation assurance passed. Summary: {assurance_path}")
    mark_done("revised_dataset_preparation_assured", {"summary": str(assurance_path)})
'''


STAGE1_CLEAN_WARMUP_CODE = r'''
ANCHOR_DATASET = REPO_ROOT / "merged_dataset_anchor_balanced"
assurance_path = DRIVE_OUTPUT_ROOT / "summaries" / "dataset_preparation_assurance_summary.json"
clean_run_name = "stage1_clean_anchor_warmup_yolo11m_detector"
clean_run_dir = DRIVE_RUNS_ROOT / clean_run_name
clean_best_path = clean_run_dir / "weights" / "best.pt"
clean_last_path = clean_run_dir / "weights" / "last.pt"

assurance_ok = False
if assurance_path.exists():
    try:
        assurance_ok = json.loads(assurance_path.read_text(encoding="utf-8")).get("ok") is True
    except Exception as exc:
        block_or_raise("revised_clean_warmup", f"Could not read assurance summary: {exc}", {"summary": str(assurance_path)})

if not ALLOW_REVISED_RETRAIN:
    block_or_raise("revised_clean_warmup", "ALLOW_REVISED_RETRAIN=False, but strict full run requires retraining.")
elif not assurance_ok:
    block_or_raise("revised_clean_warmup", f"Dataset assurance has not passed: {assurance_path}", {"summary": str(assurance_path)})
elif not dataset_has_all_classes(ANCHOR_DATASET, min_train=1200, min_val=30):
    block_or_raise("revised_clean_warmup", f"Anchor-balanced dataset is not ready: {ANCHOR_DATASET}", {"dataset": str(ANCHOR_DATASET)})
elif FORCE_REVISED_STAGE1_RETRAIN or not clean_best_path.exists():
    clean_model = clean_last_path if clean_last_path.exists() and not clean_best_path.exists() else REVISED_MODEL_SEED
    run_process(
        [
            "yolo", "detect", "train",
            f"model={clean_model}",
            f"data={ANCHOR_DATASET / 'data.yaml'}",
            f"epochs={REVISED_CLEAN_EPOCHS}",
            f"imgsz={REVISED_IMGSZ}",
            f"batch={REVISED_BATCH}",
            f"patience={max(12, min(REVISED_PATIENCE, 22))}",
            "optimizer=AdamW",
            "lr0=0.0012",
            "lrf=0.03",
            "weight_decay=0.0005",
            "box=8.0",
            "cls=0.75",
            "dfl=1.6",
            "warmup_epochs=3",
            "close_mosaic=10",
            "cos_lr=True",
            "multi_scale=True",
            "hsv_h=0.012",
            "hsv_s=0.35",
            "hsv_v=0.35",
            "degrees=2.0",
            "translate=0.05",
            "scale=0.25",
            "shear=0.5",
            "perspective=0.0003",
            "fliplr=0.5",
            "flipud=0.0",
            "mosaic=0.35",
            "mixup=0.0",
            "cutmix=0.0",
            "erasing=0.04",
            "bgr=0.02",
            "cache=disk",
            "seed=42",
            f"project={DRIVE_RUNS_ROOT}",
            f"name={clean_run_name}",
            "exist_ok=True",
        ],
        log_name="revised_train_clean_anchor_warmup",
    )
else:
    print(f"Reusing completed clean warm-up checkpoint: {clean_best_path}")

if not clean_best_path.exists():
    block_or_raise("revised_clean_warmup", f"Clean warm-up best.pt is not available after training: {clean_best_path}", {"best": str(clean_best_path)})
else:
    mark_done("revised_clean_warmup_ready", {"best": str(clean_best_path)})
'''


STAGE1_ROBUST_FINETUNE_CODE = r'''
ROBUST_DATASET = REPO_ROOT / "merged_dataset_anchor_robust"
assurance_path = DRIVE_OUTPUT_ROOT / "summaries" / "dataset_preparation_assurance_summary.json"
clean_run_name = "stage1_clean_anchor_warmup_yolo11m_detector"
robust_run_name = "stage1_anchor_robust_curriculum_yolo11m_detector"
clean_best_path = DRIVE_RUNS_ROOT / clean_run_name / "weights" / "best.pt"
robust_run_dir = DRIVE_RUNS_ROOT / robust_run_name
robust_best_path = robust_run_dir / "weights" / "best.pt"
robust_last_path = robust_run_dir / "weights" / "last.pt"

assurance_ok = False
if assurance_path.exists():
    try:
        assurance_ok = json.loads(assurance_path.read_text(encoding="utf-8")).get("ok") is True
    except Exception as exc:
        block_or_raise("revised_robust_finetune", f"Could not read assurance summary: {exc}", {"summary": str(assurance_path)})

if not clean_best_path.exists():
    block_or_raise("revised_robust_finetune", f"Clean warm-up checkpoint is missing: {clean_best_path}", {"clean_best": str(clean_best_path)})
elif not ALLOW_REVISED_RETRAIN:
    block_or_raise("revised_robust_finetune", "ALLOW_REVISED_RETRAIN=False, but strict full run requires retraining.")
elif not assurance_ok:
    block_or_raise("revised_robust_finetune", f"Dataset assurance has not passed: {assurance_path}", {"summary": str(assurance_path)})
elif not dataset_has_all_classes(ROBUST_DATASET, min_train=1200, min_val=30):
    block_or_raise("revised_robust_finetune", f"Robust dataset is not ready: {ROBUST_DATASET}", {"dataset": str(ROBUST_DATASET)})
elif FORCE_REVISED_STAGE1_RETRAIN or not robust_best_path.exists():
    robust_model = robust_last_path if robust_last_path.exists() and not robust_best_path.exists() else clean_best_path
    run_process(
        [
            "yolo", "detect", "train",
            f"model={robust_model}",
            f"data={ROBUST_DATASET / 'data.yaml'}",
            f"epochs={REVISED_ROBUST_EPOCHS}",
            f"imgsz={REVISED_IMGSZ}",
            f"batch={REVISED_BATCH}",
            f"patience={max(REVISED_PATIENCE, 35)}",
            "optimizer=AdamW",
            "lr0=0.00055",
            "lrf=0.02",
            "weight_decay=0.0005",
            "box=8.2",
            "cls=0.85",
            "dfl=1.7",
            "warmup_epochs=4",
            "close_mosaic=35",
            "cos_lr=True",
            "multi_scale=True",
            "hsv_h=0.010",
            "hsv_s=0.28",
            "hsv_v=0.30",
            "degrees=2.0",
            "translate=0.05",
            "scale=0.25",
            "shear=0.5",
            "perspective=0.0003",
            "fliplr=0.5",
            "flipud=0.0",
            "mosaic=0.20",
            "mixup=0.0",
            "cutmix=0.0",
            "erasing=0.05",
            "bgr=0.02",
            "cache=disk",
            "seed=43",
            f"project={DRIVE_RUNS_ROOT}",
            f"name={robust_run_name}",
            "exist_ok=True",
        ],
        log_name="revised_train_robust_curriculum_finetune",
    )
else:
    print(f"Reusing completed robust curriculum checkpoint: {robust_best_path}")

if not robust_best_path.exists():
    block_or_raise("revised_robust_finetune", f"Robust curriculum best.pt is not available after training: {robust_best_path}", {"best": str(robust_best_path)})
else:
    mark_done("revised_robust_finetune_ready", {"best": str(robust_best_path)})
'''


STAGE1_SELECT_CANDIDATE_CODE = r'''
clean_run_name = "stage1_clean_anchor_warmup_yolo11m_detector"
robust_run_name = "stage1_anchor_robust_curriculum_yolo11m_detector"
clean_run_dir = DRIVE_RUNS_ROOT / clean_run_name
robust_run_dir = DRIVE_RUNS_ROOT / robust_run_name
clean_best_path = clean_run_dir / "weights" / "best.pt"
robust_best_path = robust_run_dir / "weights" / "best.pt"


def metric_from_row(row: dict, predicate) -> float | None:
    for key, value in row.items():
        normalized = key.strip().lower().replace(" ", "")
        if predicate(normalized):
            try:
                return float(value)
            except Exception:
                return None
    return None


def read_training_summary(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {"run_dir": str(run_dir), "has_results": False}
    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_rows = []
    for index, row in enumerate(rows):
        map50 = metric_from_row(row, lambda key: "map50" in key and "map50-95" not in key and "map50_95" not in key)
        map5095 = metric_from_row(row, lambda key: "map50-95" in key or "map50_95" in key)
        precision = metric_from_row(row, lambda key: "precision" in key)
        recall = metric_from_row(row, lambda key: "recall" in key)
        if map50 is None and map5095 is None:
            continue
        try:
            epoch = int(float(row.get("epoch", index)))
        except Exception:
            epoch = index
        metric_rows.append(
            {
                "row_index": index,
                "epoch": epoch,
                "map50": map50,
                "map50_95": map5095,
                "precision": precision,
                "recall": recall,
                "score": map5095 if map5095 is not None else map50,
            }
        )
    if not metric_rows:
        return {"run_dir": str(run_dir), "has_results": bool(rows), "has_detection_metrics": False, "epochs": len(rows)}
    best = max(metric_rows, key=lambda item: item["score"] if item["score"] is not None else -1)
    last = metric_rows[-1]
    return {
        "run_dir": str(run_dir),
        "has_results": True,
        "has_detection_metrics": True,
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_map50": best["map50"],
        "best_map50_95": best["map50_95"],
        "best_score": best["score"],
        "last_epoch": last["epoch"],
        "last_map50": last["map50"],
        "last_map50_95": last["map50_95"],
        "last_score": last["score"],
        "epochs_since_best": max(0, len(metric_rows) - best["row_index"] - 1),
    }


if not clean_best_path.exists() and not robust_best_path.exists() and ANCHOR_CANDIDATE_WEIGHT.exists():
    print(f"Using existing selected candidate: {ANCHOR_CANDIDATE_WEIGHT}")
else:
    if not clean_best_path.exists():
        block_or_raise("revised_detector_candidate", f"Clean checkpoint missing, so candidate selection cannot run: {clean_best_path}", {"clean_best": str(clean_best_path)})
    elif not robust_best_path.exists():
        block_or_raise("revised_detector_candidate", f"Robust checkpoint missing, so candidate selection cannot run: {robust_best_path}", {"robust_best": str(robust_best_path)})
    else:
        clean_metrics = read_training_summary(clean_run_dir)
        robust_metrics = read_training_summary(robust_run_dir)
        selected_path = robust_best_path
        selected_reason = "robust_curriculum_selected"
        clean_score = clean_metrics.get("best_score")
        robust_score = robust_metrics.get("best_score")
        if isinstance(clean_score, (int, float)) and isinstance(robust_score, (int, float)) and robust_score + 0.03 < clean_score:
            selected_path = clean_best_path
            selected_reason = "clean_warmup_selected_because_robust_validation_degraded"

        convergence_summary = {
            "clean_run": str(clean_run_dir),
            "robust_run": str(robust_run_dir),
            "clean_metrics": clean_metrics,
            "robust_metrics": robust_metrics,
            "selected_weight": str(selected_path),
            "selected_reason": selected_reason,
            "design": "curriculum: clean anchor warm-up first, robust/scale-space fine-tune second, conservative built-in augmentations because robust data already contains generated variants",
        }
        convergence_path = DRIVE_OUTPUT_ROOT / "summaries" / "stage1_curriculum_convergence_summary.json"
        convergence_path.parent.mkdir(parents=True, exist_ok=True)
        convergence_path.write_text(json.dumps(convergence_summary, indent=2), encoding="utf-8")
        print(json.dumps(convergence_summary, indent=2))

        shutil.copy2(selected_path, ANCHOR_CANDIDATE_WEIGHT)
        print(f"Saved revised candidate from {selected_reason}: {ANCHOR_CANDIDATE_WEIGHT}")

if not ANCHOR_CANDIDATE_WEIGHT.exists():
    block_or_raise("revised_detector_candidate", f"Candidate weight is not available after selection: {ANCHOR_CANDIDATE_WEIGHT}", {"candidate": str(ANCHOR_CANDIDATE_WEIGHT)})
else:
    print(f"Candidate sha256: {file_sha(ANCHOR_CANDIDATE_WEIGHT)[:16]}")
    mark_done(
        "revised_detector_candidate_selected",
        {
            "weight": str(ANCHOR_CANDIDATE_WEIGHT),
            "sha256": file_sha(ANCHOR_CANDIDATE_WEIGHT),
            "clean_run": str(clean_run_dir),
            "robust_run": str(robust_run_dir),
        },
    )
'''


DEPLOYMENT_WEIGHT_SYNC_CODE = r'''
if not ANCHOR_CANDIDATE_WEIGHT.exists():
    block_or_raise("revised_local_weights_sync", f"Revised candidate missing, so local deployment sync cannot run: {ANCHOR_CANDIDATE_WEIGHT}", {"candidate": str(ANCHOR_CANDIDATE_WEIGHT)})
else:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    synced = {}
    candidate_local = WEIGHTS_DIR / "defect_detector_anchor_balanced_candidate.pt"
    shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, candidate_local)
    synced["candidate_detector"] = str(candidate_local)

    if DETECTOR_WEIGHT.exists():
        detector_local = WEIGHTS_DIR / "defect_detector.pt"
        shutil.copy2(DETECTOR_WEIGHT, detector_local)
        synced["current_detector"] = str(detector_local)
    else:
        print(f"Current Drive detector is missing, so only the revised candidate was synced: {DETECTOR_WEIGHT}")

    if SEVERITY_WEIGHT.exists():
        severity_local = WEIGHTS_DIR / "severity_cls.pt"
        shutil.copy2(SEVERITY_WEIGHT, severity_local)
        synced["severity_classifier"] = str(severity_local)
    else:
        print(f"Severity Drive weight is missing and will be checked in the severity cell: {SEVERITY_WEIGHT}")

    print("Available local deployment weights:")
    for name, path in synced.items():
        size_mb = Path(path).stat().st_size / 1024 / 1024
        print(f"- {name}: {path} ({size_mb:.1f} MB)")

    mark_done("revised_local_weights_synced", {"weights": synced})
'''


PRODUCTION_SWEEP_CODE = r'''
check_dir = DRIVE_OUTPUT_ROOT / "revised_training_production_check"
comparison_path = check_dir / "comparison_summary.json"
candidate_summary = check_dir / "anchor" / "domain_sweep_summary.csv"
current_summary = check_dir / "current" / "domain_sweep_summary.csv"

missing_weights = []
if not DETECTOR_WEIGHT.exists():
    missing_weights.append(str(DETECTOR_WEIGHT))
if not ANCHOR_CANDIDATE_WEIGHT.exists():
    missing_weights.append(str(ANCHOR_CANDIDATE_WEIGHT))
if not SEVERITY_WEIGHT.exists():
    missing_weights.append(str(SEVERITY_WEIGHT))

if missing_weights:
    block_or_raise("revised_production_sweep", "Production sweep requires missing weights.", {"missing": missing_weights})
else:
    needs_sweep = FORCE_REVISED_SWEEP or not comparison_path.exists() or not current_summary.exists() or not candidate_summary.exists()
    if needs_sweep and not ALLOW_UNCACHED_SWEEP_RUNS and not FORCE_REVISED_SWEEP:
        block_or_raise("revised_production_sweep", "Production sweep outputs are missing and uncached sweep runs are disabled.", {"output": str(check_dir)})
    elif needs_sweep:
        run_process(
            [
                sys.executable,
            "scripts/wider_production_sweep.py",
            "--base-manifest", "configs/production_eval_manifest.csv",
            "--output", str(check_dir),
            "--models", f"current={DETECTOR_WEIGHT},anchor={ANCHOR_CANDIDATE_WEIGHT}",
            "--severity", str(SEVERITY_WEIGHT),
            "--per-group", "7",
            "--max-per-query", "14",
            "--thresholds", "0.45,0.30,0.20,0.10",
            "--annotate-conf", "0.20",
            "--scenario-variants", "original,low_light,overexposed,blur_distance,shadow,occlusion",
        ],
        log_name="revised_wider_production_sweep",
        allow_failure=not REQUIRE_FULL_REVISED_RUN,
        )
    else:
        print(f"Using cached revised production check: {comparison_path}")

print(f"Current summary: {current_summary}")
print(f"Candidate summary: {candidate_summary}")
print(f"Visual report: {check_dir / 'visual_report.html'}")
if current_summary.exists() and candidate_summary.exists():
    mark_done("revised_production_sweep_done", {"visual_report": str(check_dir / "visual_report.html")})
else:
    block_or_raise(
        "revised_production_sweep",
        "Production sweep did not produce both current and candidate summaries.",
        {"current_summary": str(current_summary), "candidate_summary": str(candidate_summary)},
    )
'''


PRODUCTION_GATE_CODE = r'''
check_dir = DRIVE_OUTPUT_ROOT / "revised_training_production_check"
candidate_summary = check_dir / "anchor" / "domain_sweep_summary.csv"
current_summary = check_dir / "current" / "domain_sweep_summary.csv"
if not current_summary.exists() or not candidate_summary.exists():
    block_or_raise(
        "revised_production_gate",
        "Production sweep summaries are missing. Run the production sweep cell first.",
        {"current_summary": str(current_summary), "candidate_summary": str(candidate_summary)},
    )

gate_path = DRIVE_OUTPUT_ROOT / "summaries" / "revised_anchor_production_gate.json"
run_process(
        [
            sys.executable,
            "scripts/revised_production_gate.py",
            "--current-summary", str(current_summary),
            "--candidate-summary", str(candidate_summary),
            "--output", str(gate_path),
            "--threshold", "0.2",
            "--min-evaluation-rows", "30",
            "--min-candidate-pass-rate", "0.75",
            "--min-domain-pass-rate", "0.70",
            "--min-expected-pass-rate", "0.70",
            "--min-bucket-samples", "3",
            "--min-pass-rate-delta", "0.03",
            "--allowed-class-regression", "0",
            "--max-false-positive-increase", "0",
        ],
        log_name="revised_production_gate",
        allow_failure=not REQUIRE_FULL_REVISED_RUN,
    )
if not gate_path.exists():
    block_or_raise(
        "revised_production_gate",
        f"Production gate did not produce a decision summary: {gate_path}",
        {"gate": str(gate_path)},
    )
else:
    decision = json.loads(gate_path.read_text(encoding="utf-8"))
    if decision.get("promote") and PROMOTE_REVISED_MODEL:
        shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, DETECTOR_WEIGHT)
        shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, ANCHOR_PROMOTED_WEIGHT)
        shutil.copy2(ANCHOR_CANDIDATE_WEIGHT, WEIGHTS_DIR / "defect_detector.pt")
        print(f"Promoted revised detector to canonical deployment weight: {DETECTOR_WEIGHT}")
    elif decision.get("promote"):
        print("Candidate passed the revised gate, but PROMOTE_REVISED_MODEL=False, so no deployment weight was overwritten.")
    else:
        print("Candidate failed the revised gate; deployment weight was not changed.")

print(json.dumps(decision, indent=2)[-8000:])
mark_done("revised_production_check_done", {"gate": str(gate_path), "visual_report": str(check_dir / "visual_report.html")})
'''


cells = [
    md(
        """
# Revised Two-Stage YOLO Training Notebook

This notebook is the smaller-cell, recoverable training path for the revised detector. Run **Cell 1** after every Colab restart; after that, each code cell is designed to either reuse existing artifacts or rebuild only its own stage.

## Why This Flow Exists

The earlier fine-tuning attempts could fail for three practical reasons: incomplete baseline recovery, noisy target-domain labels, and strong augmentation before the detector had learned stable class boundaries. This version keeps the baseline classes anchored, adds wall-maintenance evidence carefully, uses curriculum training, and blocks deployment unless the production sweep gate passes.

## Recovery Rules

- Cell 1 sets Drive paths, flags, helper functions, and repo location.
- Later cells are intentionally smaller. If a later stage already exists in Drive or the repo, the cell prints and reuses it.
- Dataset and model outputs are mirrored into `AIEngGroupProj_colab_outputs` so a disconnected runtime can resume without repeating expensive work.
- Training remains forced by default for this improvement round, but completed `best.pt` checkpoints are reused unless `FORCE_REVISED_STAGE1_RETRAIN=True`.
"""
    ),
    md(
        """
## Architecture Diagram

The architecture is deliberately deeper than a plain train-and-predict YOLO flow. It separates evidence sources, data governance, robust dataset construction, model training, deployment evaluation, and the human feedback loop.

"""
        + f'<img src="{architecture_diagram_png_data_uri()}" alt="Two-stage YOLO defect detection architecture" width="100%">'
        + """

Key point for the report: the wall-defect paper's idea is integrated at the evidence and taxonomy-mapping layer, while this project adds baseline class protection, severity classification, class/domain balancing, scale-space crops, scenario-based production testing, and deployment gating.
"""
    ),
    md(
        """
## Cell 1: Setup, Drive, Flags, and Helpers

Run this first after every runtime restart. It mounts Drive, finds/clones the repo, installs requirements once, defines all shared paths, and loads the recoverable state file. Later cells assume these variables and helper functions already exist.
"""
    ),
    code(SETUP_CODE),
    md(
        """
## Cell 2: Materialize Revised Project Files

This writes the exact revised scripts into the Colab repo and compiles them. Run it once after setup, or rerun it if you suspect the cloned repo is stale. It is safe to rerun because unchanged files are skipped.
"""
    ),
    code(materialize_project_code()),
    md(
        """
## Cell 3: Recover Baseline Dataset and Existing Weights

This cell restores cached datasets and weights from Drive, then verifies that `merged_dataset` still contains all five baseline classes. It intentionally stops if the baseline is incomplete, because target-only fine-tuning can erase classes.
"""
    ),
    code(BASELINE_RECOVERY_CODE),
    md(
        """
## Cell 4: Collect Wall/Facility Target Sources

This downloads or reuses target-domain Roboflow datasets and converts their labels into the five-class project taxonomy. The wall-maintenance classes from the research paper are mapped safely instead of becoming incompatible new model outputs.
"""
    ),
    code(TARGET_SOURCE_COLLECTION_CODE),
    md(
        """
## Cell 5: Audit Source Label Mapping Coverage

This audits the actual source label files that are present. If any used class ID from the baseline raw sources or target raw sources is not mapped into the five-class taxonomy, the notebook stops here instead of discarding labels silently.
"""
    ),
    code(LABEL_MAPPING_AUDIT_CODE),
    md(
        """
## Cell 6: Prepare Hard Negatives

This collects clean negative examples from prior sweep caches. They are saved with empty YOLO label files and used to reduce false positives on old walls, generic texture, and non-defect surfaces.
"""
    ),
    code(HARD_NEGATIVE_PREP_CODE),
    md(
        """
## Cell 7: Anchor Dataset Preflight Audit

This is a cheap safety audit before writing the full balanced dataset. It checks target keep-rate, rejected-label reasons, final class coverage, and class ratio. It removes the temporary audit folder afterward to save storage.
"""
    ),
    code(ANCHOR_PREFLIGHT_CODE),
    md(
        """
## Cell 8: Build Anchor-Balanced Dataset

This builds `merged_dataset_anchor_balanced`. It preserves the clean baseline as an anchor, caps target data, reserves target validation examples, round-robins domains, oversamples weak classes, and mirrors the result into Drive for recovery.
"""
    ),
    code(ANCHOR_BUILD_CODE),
    md(
        """
## Cell 9: Build Robust and Scale-Space Dataset

This builds `merged_dataset_anchor_robust` from the anchor-balanced dataset. It adds controlled lighting, occlusion, blur, edge/contrast, perspective, surface color, hard-negative, and label-preserving scale-space crop variants.
"""
    ),
    code(ROBUST_DATASET_BUILD_CODE),
    md(
        """
## Cell 10: Audit Visual Feature Coverage

This checks whether each class has enough diversity in object scale, aspect shape, edge density, texture, brightness, and saturation. It is meant to catch a dataset that is numerically balanced but visually narrow.
"""
    ),
    code(FEATURE_COVERAGE_AUDIT_CODE),
    md(
        """
## Cell 11: Final Dataset Preparation Assurance Gate

This reads all dataset preparation summaries and refuses to start GPU training if any guard failed. It checks target mapping, baseline class preservation, target quality, final class balance, robust variants, scale-space crops, and feature coverage.
"""
    ),
    code(DATASET_PREP_ASSURANCE_CODE),
    md(
        """
## Cell 12: Stage 1 Clean Anchor Warm-Up

This is the first detector training phase. It trains YOLO11m on the cleaner anchor-balanced dataset with mild augmentation so the model learns stable class boundaries before seeing stronger synthetic variants.
"""
    ),
    code(STAGE1_CLEAN_WARMUP_CODE),
    md(
        """
## Cell 13: Stage 1 Robust Curriculum Fine-Tune

This fine-tunes from the clean warm-up checkpoint on the robust/scale-space dataset. Built-in YOLO augmentation is intentionally conservative because the dataset already contains robust variants; this reduces the risk of a low mAP plateau from excessive distortion.
"""
    ),
    code(STAGE1_ROBUST_FINETUNE_CODE),
    md(
        """
## Cell 14: Select Stage 1 Candidate Checkpoint

This compares clean and robust training summaries. If robust fine-tuning clearly degrades validation score, the clean warm-up checkpoint is selected instead. The chosen candidate is copied to Drive.
"""
    ),
    code(STAGE1_SELECT_CANDIDATE_CODE),
    md(
        """
## Cell 15: Sync Local Deployment Weights

This copies the selected candidate, current detector, and severity classifier into the repo `weights/` folder for dashboard testing. It does not overwrite the canonical Drive detector unless the promotion gate later passes.
"""
    ),
    code(DEPLOYMENT_WEIGHT_SYNC_CODE),
    md(
        """
## Cell 16: Stage 2 Severity Coverage and Optional Training

This checks whether the severity crop dataset has enough `minor`, `moderate`, and `critical` images. It only retrains the severity classifier when the class coverage is strong enough.
"""
    ),
    code(SEVERITY_CODE),
    md(
        """
## Cell 17: Production Sweep and Visual Report

This compares the current detector and revised candidate on external/cached images with scenario variants: original, low light, overexposed, distance blur, shadow, and occlusion. The HTML visual report is saved in Drive.
"""
    ),
    code(PRODUCTION_SWEEP_CODE),
    md(
        """
## Cell 18: Promotion Gate

This reads the production sweep summaries and applies the deployment-confidence gate. Promotion only occurs if the candidate passes and `PROMOTE_REVISED_MODEL=True`; otherwise the canonical deployment model is not overwritten.
"""
    ),
    code(PRODUCTION_GATE_CODE),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = REPO_ROOT / "two_stage_yolo_defect_pipeline_revised_training_colab.ipynb"
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {output_path} with {len(cells)} cells")
print(f"Markdown cells: {sum(cell['cell_type'] == 'markdown' for cell in cells)}")
print(f"Code cells: {sum(cell['cell_type'] == 'code' for cell in cells)}")
