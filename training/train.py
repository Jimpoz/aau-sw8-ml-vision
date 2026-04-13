from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26 for an AAU facility.")
    parser.add_argument("--facility_id", required=True, help="Facility identifier (used for run + model naming)")
    parser.add_argument("--data_yaml", required=True, help="Ultralytics dataset YAML (train/val paths + class names)")
    parser.add_argument("--base_model", default="yolo26.pt", help="Base YOLO26 weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="", help="e.g. 'cpu' or '0'. Empty = let Ultralytics decide.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--project", default="training/runs", help="Ultralytics project directory")
    args = parser.parse_args()

    # Import lazily so "python -m training.train --help" doesn't pull ML deps.
    from ultralytics import YOLO

    run_dir = os.path.join(args.project, args.facility_id)
    _ensure_dir(run_dir)

    # Ultralytics will create its own run structure inside `project/name`.
    # We use `name=facility_id` so artifacts are grouped per facility.
    model = YOLO(args.base_model)

    start = time.time()
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device or None,
        seed=args.seed,
        project=args.project,
        name=args.facility_id,
        exist_ok=True,
    )
    elapsed_s = time.time() - start

    # Best-effort extraction of the "best" weights path.
    best_weights: Optional[str] = None
    try:
        # Ultralytics uses results object + model.trainer
        best_weights = getattr(model, "best", None) or getattr(results, "best", None)
    except Exception:
        best_weights = None

    meta = {
        "facility_id": args.facility_id,
        "data_yaml": args.data_yaml,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
        "elapsed_s": elapsed_s,
        "best_weights": best_weights,
        "ultralytics_results_type": str(type(results)),
    }

    out_path = os.path.join(run_dir, "training_metadata.json")
    _write_json(out_path, meta)
    print(f"[train] wrote {out_path}")


if __name__ == "__main__":
    main()
