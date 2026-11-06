"""M2: YOLO11 phan loai tren tung frame.  python src/train_yolo_cls.py

Ultralytics dung som theo top-1 cap frame, khac M1/M3 (F1 cap clip). Hai lop can bang nen chap nhan duoc;
danh gia cuoi van o cap clip qua evaluate.py.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/frames")
    parser.add_argument("--out", default="results/m2")
    parser.add_argument("--weights", default="yolo11n-cls.pt")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings.update({"sync": False})
    out = Path(args.out).resolve()
    model = YOLO(str(args.weights))
    model.train(
        data=str(Path(args.frames).resolve()),
        imgsz=args.image_size,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        project=str(out),
        name="train",
        exist_ok=True,
        plots=False,
    )
    shutil.copy(Path(model.trainer.best), out / "best.pt")


if __name__ == "__main__":
    main()
