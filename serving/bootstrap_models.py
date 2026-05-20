from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_ONNX_URL = "https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx"

COCO_80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dst: pathlib.Path) -> None:
    _ensure_dir(dst.parent)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".partial")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dst)


def _metadata_has_class_names(path: pathlib.Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    names = data.get("class_names") or data.get("names") or []
    return isinstance(names, list) and len(names) > 0


def _write_metadata(path: pathlib.Path, facility_id: str, class_names: Optional[List[str]], extra: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        "facility_id": facility_id,
        "timestamp_unix_ms": int(time.time() * 1000),
        "class_names": class_names or [],
        **extra,
    }
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ensure_server_model() -> Dict[str, Any]:
    """
    Ensures that the server-side ONNX model exists under:
      {MODEL_DIR}/{FACILITY_ID}/yolo26_indoor.onnx

    Behavior:
    - If it exists, do nothing.
    - Otherwise:
      - download a ready-to-run ONNX model into the facility folder
      - write a metadata.json alongside it
    """
    model_dir = pathlib.Path(os.getenv("MODEL_DIR", "./models"))
    facility_id = os.getenv("FACILITY_ID", "default_facility")

    facility_dir = model_dir / facility_id
    onnx_path = facility_dir / "yolo26_indoor.onnx"
    meta_path = facility_dir / "metadata.json"

    # Resolve class names: explicit override wins over the default
    # COCO list. Supplying MODEL_CLASS_NAMES is the escape hatch for
    # custom-trained models that aren't COCO-shaped.
    override = os.getenv("MODEL_CLASS_NAMES", "").strip()
    if override:
        class_names = [c.strip() for c in override.split(",") if c.strip()]
    else:
        class_names = list(COCO_80_CLASSES)

    if onnx_path.exists():
        # Backfill class_names into metadata even when the ONNX was
        # already downloaded by a previous run that didn't write them.
        # Without this the detector would keep emitting "0" / "67" /
        # ... numeric labels and the landmark match rules would never
        # fire against named COCO classes.
        if not meta_path.exists() or not _metadata_has_class_names(meta_path):
            _write_metadata(
                meta_path,
                facility_id=facility_id,
                class_names=class_names,
                extra={
                    "server_artifacts": {
                        "onnx": os.path.basename(str(onnx_path)),
                        "onnx_url": os.getenv("MODEL_ONNX_URL", DEFAULT_ONNX_URL),
                    },
                    "class_names_source": "override" if override else "default_coco_80",
                },
            )
            return {"status": "ok", "action": "metadata_backfilled", "facility_id": facility_id, "onnx_path": str(onnx_path)}
        return {"status": "ok", "action": "exists", "facility_id": facility_id, "onnx_path": str(onnx_path)}

    onnx_url = os.getenv("MODEL_ONNX_URL", DEFAULT_ONNX_URL)
    _download_file(onnx_url, onnx_path)

    _write_metadata(
        meta_path,
        facility_id=facility_id,
        class_names=class_names,
        extra={
            "server_artifacts": {
                "onnx": os.path.basename(str(onnx_path)),
                "onnx_url": onnx_url,
            },
            "class_names_source": "override" if override else "default_coco_80",
        },
    )

    return {
        "status": "ok",
        "action": "downloaded",
        "facility_id": facility_id,
        "onnx_url": onnx_url,
        "onnx_path": str(onnx_path),
        "metadata_path": str(meta_path),
    }

