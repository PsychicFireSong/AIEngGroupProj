"""
Stage 2 severity classifier v4 — EfficientNet-B3 transfer learning.

Root problem with YOLO-cls approach:
  - YOLO11n-cls (1.5M params) + 30 epochs = 65.8% -- feature set too limited
  - YOLO11s-cls (9.4M params) -- overfits, peaked epoch 3 at 62.2%
  - YOLO classifiers are optimised for detection tasks; weak at fine-grained severity

Fix: proper transfer learning with torchvision's EfficientNet-B3
  - Pretrained on ImageNet-1k (5.1M+ images, 1000 classes)
  - Rich visual features that generalise to severity recognition
  - 12M total params; we freeze backbone so ~400K are trainable in Phase 1
  - Two-phase training: frozen backbone -> then unfreeze top blocks

Expected result: 72-80% (vs 65.8% YOLO ceiling)
Realistic ceiling: ~75-80% with 1800 training images (label noise limits higher)
To reach 90%: need 10,000+ clean expert-labelled images
"""
from __future__ import annotations

import copy
import csv
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

SEVERITY_DS   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round4\severity_effb3")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch4_candidate.pt")

CLASSES = ["critical", "minor", "moderate"]
DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 300  # EfficientNet-B3 native size


def build_loaders(batch=32):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = ImageFolder(str(SEVERITY_DS / "train"), transform=train_tf)
    val_ds   = ImageFolder(str(SEVERITY_DS / "val"),   transform=val_tf)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_dl, val_dl, train_ds.classes


def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    # Replace classifier: 1536 -> 256 -> 3
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 3),
    )
    # Initialise new head
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.xavier_uniform_(model.classifier[4].weight)
    return model.to(DEVICE)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_top_blocks(model, n_blocks: int = 3):
    """Unfreeze last n_blocks of EfficientNet features (+ classifier stays unfrozen)."""
    features = list(model.features.children())
    for block in features[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    # Always keep bn running stats trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = True
            m.bias.requires_grad   = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, opt, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n


def run_phase(model, train_dl, val_dl, epochs, lr, wd, label_sm, tag):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_sm)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_acc, best_state, patience_count = 0.0, None, 0
    patience = 20

    print(f"\n--- Phase: {tag} | trainable params: {count_trainable(model):,} | LR: {lr} | WD: {wd} ---")
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, opt, scaler)
        vl_loss, vl_acc = eval_epoch(model, val_dl, criterion)
        sched.step()

        mark = "*" if vl_acc > best_acc else " "
        print(f"  ep {ep:03d}/{epochs}  tr_acc={tr_acc:.4f}  vl_acc={vl_acc:.4f}  "
              f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f} {mark}")

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at ep {ep} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    print(f"  Phase {tag} done. Best val acc: {best_acc:.4f}")
    return best_acc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STAGE 2 SEVERITY v4 — EfficientNet-B3 Transfer Learning")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {SEVERITY_DS}")

    train_dl, val_dl, class_names = build_loaders(batch=32)
    model = build_model()

    # Phase 1: Frozen backbone, train only new classifier head (20 epochs)
    freeze_backbone(model)
    acc1 = run_phase(model, train_dl, val_dl,
                     epochs=25, lr=3e-3, wd=1e-2, label_sm=0.1, tag="P1-frozen-backbone")

    # Phase 2: Unfreeze top 3 blocks + classifier, low LR (50 epochs)
    unfreeze_top_blocks(model, n_blocks=3)
    acc2 = run_phase(model, train_dl, val_dl,
                     epochs=60, lr=5e-5, wd=1e-2, label_sm=0.05, tag="P2-unfreeze-top3")

    # Phase 3: Unfreeze more (top 6 blocks), very low LR (30 epochs)
    unfreeze_top_blocks(model, n_blocks=6)
    acc3 = run_phase(model, train_dl, val_dl,
                     epochs=30, lr=1e-5, wd=5e-3, label_sm=0.02, tag="P3-unfreeze-top6")

    best_acc = max(acc1, acc2, acc3)
    print(f"\nFinal best val accuracy: {best_acc:.4f}  (target: >0.90)")

    # Save as ONNX-compatible checkpoint dict
    save_path = OUT_DIR / "severity_effb3_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "best_val_acc": best_acc,
        "arch": "efficientnet_b3",
    }, save_path)
    print(f"Saved: {save_path}")

    # Also copy to candidate slot
    shutil.copy2(save_path, CANDIDATE_OUT.parent / "severity_cls_patch4_effb3.pt")
    print(f"Candidate copy: severity_cls_patch4_effb3.pt")

    # Per-class breakdown on val set
    print("\nPer-class val accuracy:")
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    from torchvision.datasets import ImageFolder
    val_ds = ImageFolder(str(SEVERITY_DS / "val"), transform=val_tf)
    val_dl_full = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    per_class_correct = [0] * 3
    per_class_total   = [0] * 3
    with torch.no_grad():
        for imgs, labels in val_dl_full:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for cls_idx in range(3):
                mask = labels == cls_idx
                per_class_correct[cls_idx] += (preds[mask] == cls_idx).sum().item()
                per_class_total[cls_idx]   += mask.sum().item()

    for i, cls in enumerate(val_ds.classes):
        acc = per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        print(f"  {cls:12s}: {acc:.4f}  ({per_class_correct[i]}/{per_class_total[i]})")


if __name__ == "__main__":
    main()
