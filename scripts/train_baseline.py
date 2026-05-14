import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline YOLO model.")
    parser.add_argument("--data", default="merged_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device id or 'cpu'")
    parser.add_argument("--project", default="runs/train", help="Output directory")
    parser.add_argument("--name", default="baseline", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"Missing data file: {data_path}")

    model = YOLO(args.weights)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
