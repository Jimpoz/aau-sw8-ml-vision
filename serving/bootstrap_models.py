from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_ONNX_URL = "https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx"


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

    if onnx_path.exists():
        return {"status": "ok", "action": "exists", "facility_id": facility_id, "onnx_path": str(onnx_path)}

    onnx_url = os.getenv("MODEL_ONNX_URL", DEFAULT_ONNX_URL)
    _download_file(onnx_url, onnx_path)

    _write_metadata(
        meta_path,
        facility_id=facility_id,
        class_names=None,
        extra={
            "server_artifacts": {
                "onnx": os.path.basename(str(onnx_path)),
                "onnx_url": onnx_url,
            }
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

