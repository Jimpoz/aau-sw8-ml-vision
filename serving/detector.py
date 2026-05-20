from __future__ import annotations

import json
import os
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # (x1,y1,x2,y2) in original image coordinates


def _letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize + pad to square while preserving aspect ratio (YOLO-style).
    Returns (padded_img, ratio, (pad_w, pad_h)).
    """
    if img.ndim != 3:
        raise ValueError("Expected HWC image array")

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    # Resize
    pil = Image.fromarray(img)
    resized = pil.resize(new_unpad, resample=Image.BILINEAR)
    resized_np = np.asarray(resized)

    # Pad
    pad_w = new_shape - new_unpad[0]
    pad_h = new_shape - new_unpad[1]
    pad_left = int(round(pad_w / 2))
    pad_top = int(round(pad_h / 2))
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    pad_value = color[0] if isinstance(color, (tuple, list)) else color
    padded = np.pad(
        resized_np,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded, r, (pad_left, pad_top)


def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between one box a (4,) and b boxes (N,4).
    """
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = np.maximum(0.0, a[2] - a[0]) * np.maximum(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = np.maximum(1e-9, area_a + area_b - inter)
    return inter / union


def _nms_xyxy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    max_detections: int = 100,
) -> List[int]:
    """
    Simple NMS, class-agnostic. Returns indices kept (sorted by descending score).
    """
    if boxes_xyxy.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_detections:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _xyxy_iou(boxes_xyxy[i], boxes_xyxy[rest])
        order = rest[ious <= iou_threshold]
    return keep


def _parse_predictions(
    outputs: List[np.ndarray] | Tuple[np.ndarray, ...],
    input_size: int,
    conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Best-effort parsing of common YOLO ONNX export formats.

    Returns:
      boxes_xyxy: (M,4)
      scores: (M,)
      class_ids: (M,)
    """
    arrays: List[np.ndarray] = []
    for out in outputs:
        if isinstance(out, np.ndarray):
            arrays.append(out)

    if not arrays:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    # Pick an array that looks like "predictions".
    # Common shapes:
    # - (1, N, 6) or (N, 6)  -> [x1,y1,x2,y2,conf,cls]
    # - (1, N, 5+nc)         -> [cx,cy,w,h,obj, class_probs...]
    # - (1, 1, N, 6)         -> similar
    pred = None
    for a in arrays:
        if a.ndim < 2:
            continue
        last = a.shape[-1]
        if last >= 6 and last <= 8:
            pred = a
            break
    if pred is None:
        for a in arrays:
            if a.ndim >= 3:
                last = a.shape[-1]
                if last >= 6:
                    pred = a
                    break
    if pred is None:
        pred = arrays[0]

    p = pred

    # Flatten to (N, D) with D being last dim.
    if p.ndim == 3 and p.shape[0] == 1:
        p = p[0]
    elif p.ndim == 4 and p.shape[0] == 1 and p.shape[1] == 1:
        p = p[0, 0]
    elif p.ndim > 2:
        p = p.reshape(-1, p.shape[-1])

    if p.ndim != 2 or p.shape[1] < 6:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    d = p.shape[1]

    if d in (6, 7, 8):
        # Assume: x1,y1,x2,y2,conf,cls (extra columns ignored)
        x1 = p[:, 0]
        y1 = p[:, 1]
        x2 = p[:, 2]
        y2 = p[:, 3]
        conf = p[:, 4]
        cls = p[:, 5]
        cls = np.floor(cls).astype(np.int64)
        mask = conf >= conf_threshold
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)[mask]
        scores = conf.astype(np.float32)[mask]
        class_ids = cls[mask]
        return boxes_xyxy, scores, class_ids

    # Assume YOLO-like: cx,cy,w,h,obj, class_probs...
    # If the export differs, results may be empty, but this keeps the service functional.
    cx = p[:, 0]
    cy = p[:, 1]
    w = p[:, 2]
    h = p[:, 3]
    obj = p[:, 4]
    class_scores = p[:, 5:]
    class_ids = class_scores.argmax(axis=1).astype(np.int64)
    class_conf = class_scores.max(axis=1)
    scores = (obj * class_conf).astype(np.float32)

    mask = scores >= conf_threshold
    cx = cx[mask]
    cy = cy[mask]
    w = w[mask]
    h = h[mask]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1).astype(np.float32)
    return boxes_xyxy, scores[mask], class_ids[mask]


class Detector:
    """
    Per-facility detector that loads the facility-specific ONNX artifact (YOLO26 exported).
    """

    def __init__(
        self,
        *,
        facility_id: str,
        model_dir: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> None:
        self.facility_id = facility_id
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_dir = model_dir

        self._onnx_path = os.path.join(model_dir, facility_id, "yolo26_indoor.onnx")
        self._effective_facility = facility_id
        if not os.path.exists(self._onnx_path):
            fallback = os.getenv("MODEL_FALLBACK_FACILITY_ID", "default_facility")
            fallback_path = os.path.join(model_dir, fallback, "yolo26_indoor.onnx")
            if fallback and fallback != facility_id and os.path.exists(fallback_path):
                print(
                    f"[detector] facility={facility_id!r} has no ONNX; "
                    f"falling back to facility={fallback!r}",
                    flush=True,
                )
                self._onnx_path = fallback_path
                self._effective_facility = fallback
            else:
                raise FileNotFoundError(f"ONNX model not found: {self._onnx_path}")

        self._metadata_path = os.path.join(model_dir, self._effective_facility, "metadata.json")
        self._class_names: Optional[List[str]] = None
        if os.path.exists(self._metadata_path):
            try:
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self._class_names = meta.get("class_names") or meta.get("names")
            except Exception:
                self._class_names = None

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(self._onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])

        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name if inputs else "images"

    @property
    def onnx_path(self) -> str:
        return self._onnx_path

    def infer(
        self,
        image_bytes: bytes,
        *,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> Tuple[List[Detection], Dict[str, Any]]:
        start = time.time()

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        img_np = np.asarray(img)

        padded, ratio, (pad_left, pad_top) = _letterbox(img_np, new_shape=self.input_size)
        # CHW, float32 [0,1]
        chw = np.transpose(padded, (2, 0, 1)).astype(np.float32) / 255.0
        batch = np.expand_dims(chw, axis=0)

        outputs = self._session.run(None, {self._input_name: batch})
        conf_t = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou_t = self.iou_threshold if iou_threshold is None else float(iou_threshold)
        boxes_xyxy, scores, class_ids = _parse_predictions(outputs, input_size=self.input_size, conf_threshold=conf_t)

        if boxes_xyxy.shape[0] == 0:
            return [], {"inference_ms": (time.time() - start) * 1000.0}

        # Map boxes from input coords -> original coords.
        # boxes are assumed to be in the padded+resized coordinate system.
        # Undo padding and scaling.
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= max(ratio, 1e-9)

        # Clip to original size
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h - 1)

        keep = _nms_xyxy(boxes_xyxy, scores, iou_threshold=iou_t)

        detections: List[Detection] = []
        for idx in keep:
            cls_id = int(class_ids[idx])
            label = str(cls_id)
            if self._class_names and 0 <= cls_id < len(self._class_names):
                label = str(self._class_names[cls_id])
            x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
            detections.append(Detection(label=label, confidence=float(scores[idx]), bbox_xyxy=(float(x1), float(y1), float(x2), float(y2))))

        return detections, {
            "inference_ms": (time.time() - start) * 1000.0,
            "num_raw_predictions": int(boxes_xyxy.shape[0]),
            "num_after_nms": int(len(detections)),
        }
