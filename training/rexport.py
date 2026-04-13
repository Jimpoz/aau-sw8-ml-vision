from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Any, Dict, Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _export_one(model: Any, *, format_name: str) -> Optional[str]:
    """
    Ultralytics export returns a path-like string (best effort).
    """
    try:
        out = model.export(format=format_name)
    except Exception:
        return None

    if isinstance(out, (str, os.PathLike)):
        return str(out)
    if isinstance(out, (list, tuple)) and out:
        # Often a list of exported file paths.
        first = out[0]
        if isinstance(first, (str, os.PathLike)):
            return str(first)
    return None


def _copy_if_exists(src: Optional[str], dst: str) -> Optional[str]:
    if not src:
        return None
    if not os.path.exists(src):
        return None
    shutil.copyfile(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained YOLO26 to platform artifacts.")
    parser.add_argument("--facility_id", required=True)
    parser.add_argument("--weights_path", required=True, help="Trained weights path (e.g. runs/.../weights/best.pt)")
    parser.add_argument("--models_dir", default="models", help="Root folder that stores exported artifacts")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported by export")
    parser.add_argument("--device", default="", help="e.g. cpu or 0 (default: let Ultralytics decide)")
    args = parser.parse_args()

    from ultralytics import YOLO

    out_dir = os.path.join(args.models_dir, args.facility_id)
    _ensure_dir(out_dir)

    model = YOLO(args.weights_path)

    # Export candidates. These names are part of the serving contract.
    target_files = {
        "onnx": os.path.join(out_dir, "yolo26_indoor.onnx"),
        "mlpackage": os.path.join(out_dir, "yolo26_indoor.mlpackage"),
        "tflite": os.path.join(out_dir, "yolo26_indoor.tflite"),
    }

    exported: Dict[str, Any] = {"facility_id": args.facility_id, "export_time_unix": int(time.time()), "imgsz": args.imgsz}
    errors: Dict[str, str] = {}

    # Note: Ultralytics export paths vary by version; we export and then copy to our stable names.
    for fmt in ("onnx", "mlpackage", "tflite"):
        desired = target_files[fmt]
        try:
            exported_src = _export_one(model, format_name=fmt)
            if fmt == "mlpackage":
                # mlpackage is typically a directory; copy/rename whole folder if needed.
                if exported_src and os.path.isdir(exported_src) and os.path.exists(exported_src):
                    # Remove pre-existing dst directory/file.
                    if os.path.exists(desired):
                        if os.path.isdir(desired):
                            shutil.rmtree(desired)
                        else:
                            os.remove(desired)
                    shutil.copytree(exported_src, desired)
                    exported[fmt] = {"dst": desired, "src": exported_src}
                else:
                    # If export returns a file, best effort copy.
                    _copy_if_exists(exported_src, desired)
                    exported[fmt] = {"dst": desired, "src": exported_src}
            else:
                _copy_if_exists(exported_src, desired)
                exported[fmt] = {"dst": desired, "src": exported_src}
        except Exception as e:
            errors[fmt] = str(e)

    # Class names, if available from the model.
    names = None
    try:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            # Ultralytics usually stores names as {id: name}
            names = [names[k] for k in sorted(names.keys())]
    except Exception:
        names = None

    meta = {
        "facility_id": args.facility_id,
        "trained_weights": args.weights_path,
        "input_size": args.imgsz,
        "class_names": names,
        "export": exported,
        "errors": errors,
        "generated_time_unix": int(time.time()),
        "contract": {
            "onnx": "models/{facility_id}/yolo26_indoor.onnx",
            "ios": "models/{facility_id}/yolo26_indoor.mlpackage",
            "android": "models/{facility_id}/yolo26_indoor.tflite",
        },
    }

    _write_json(os.path.join(out_dir, "metadata.json"), meta)
    print(f"[export] wrote {os.path.join(out_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
