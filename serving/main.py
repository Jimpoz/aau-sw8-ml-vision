from __future__ import annotations

import os
import time
from functools import lru_cache
import shutil
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from serving.detector import Detection, Detector
from serving.location_resolver import LocationEntity, LocationResolver, RouteStep


class DetectionDTO(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: List[float] = Field(description="(x1,y1,x2,y2) in image pixel coordinates")


class InferenceResponse(BaseModel):
    facility_id: str
    request_id: Optional[str] = None
    timestamp_unix_ms: int
    detections: List[DetectionDTO]
    location: LocationEntity
    route: Optional[List[RouteStep]] = None
    debug: Dict[str, Any] = Field(default_factory=dict)


MODEL_DIR = os.getenv("MODEL_DIR", "./models")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
_resolver = LocationResolver()


def _artifact_path(facility_id: str, artifact_name: str) -> str:
    return os.path.join(MODEL_DIR, facility_id, artifact_name)


@lru_cache(maxsize=16)
def _get_detector(facility_id: str) -> Detector:
    # Detector is cached per facility to avoid reloading ONNX sessions.
    return Detector(facility_id=facility_id, model_dir=MODEL_DIR)


def _load_detector_for_facility(facility_id: str) -> Detector:
    try:
        return _get_detector(facility_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


app = FastAPI(title="AAU Indoor Vision Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models/{facility_id}")
def list_models(facility_id: str) -> Dict[str, Any]:
    # Model filenames are part of the contract described in the project README.
    return {
        "facility_id": facility_id,
        "artifacts": {
            "server": os.path.basename(_artifact_path(facility_id, "yolo26_indoor.onnx")),
            "ios": os.path.basename(_artifact_path(facility_id, "yolo26_indoor.mlpackage")),
            "android": os.path.basename(_artifact_path(facility_id, "yolo26_indoor.tflite")),
            "metadata": os.path.basename(_artifact_path(facility_id, "metadata.json")),
        },
        "available": {
            "server": os.path.exists(_artifact_path(facility_id, "yolo26_indoor.onnx")),
            "ios": os.path.exists(_artifact_path(facility_id, "yolo26_indoor.mlpackage")),
            "android": os.path.exists(_artifact_path(facility_id, "yolo26_indoor.tflite")),
            "metadata": os.path.exists(_artifact_path(facility_id, "metadata.json")),
        },
        "model_dir": MODEL_DIR,
    }


@app.get("/v1/models/{facility_id}/{platform}")
def download_model(facility_id: str, platform: str) -> FileResponse:
    platform = platform.lower()
    if platform == "server":
        path = _artifact_path(facility_id, "yolo26_indoor.onnx")
        media_type = "application/octet-stream"
    elif platform in ("ios", "coreml"):
        path = _artifact_path(facility_id, "yolo26_indoor.mlpackage")
        media_type = "application/octet-stream"
    elif platform in ("android", "tflite"):
        path = _artifact_path(facility_id, "yolo26_indoor.tflite")
        media_type = "application/octet-stream"
    elif platform in ("metadata", "meta", "json"):
        path = _artifact_path(facility_id, "metadata.json")
        media_type = "application/json"
    else:
        raise HTTPException(status_code=400, detail="platform must be one of: server, ios, android, metadata")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")

    # `.mlpackage` is typically a directory bundle; zip it for transport.
    if os.path.isdir(path):
        artifact_mtime = int(os.path.getmtime(path))
        zip_name = f"{facility_id}_{os.path.basename(path)}_{artifact_mtime}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_name)
        if not os.path.exists(zip_path):
            # Create a deterministic zip from the directory contents.
            base_name = zip_path[: -len(".zip")]
            shutil.make_archive(base_name, "zip", root_dir=path)
        return FileResponse(zip_path, media_type="application/zip", filename=zip_name)

    filename = os.path.basename(path)
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/v1/detect", response_model=InferenceResponse)
async def detect(
    facility_id: str = Form(..., description="Facility identifier for choosing the correct model + Neo4j namespace"),
    image: UploadFile = File(..., description="Camera frame image (jpg/png)"),
    navigation_to: Optional[str] = Form(None, description="Optional target room/landmark name for navigation route"),
    request_id: Optional[str] = Form(None, description="Optional client request correlation id"),
    confidence_threshold: Optional[float] = Form(None, ge=0.0, le=1.0, description="Override minimum confidence"),
    iou_threshold: Optional[float] = Form(None, ge=0.0, le=1.0, description="Override NMS IoU threshold"),
) -> InferenceResponse:
    if image.content_type and image.content_type not in ("image/jpeg", "image/png", "application/octet-stream"):
        # Keep it permissive because mobile clients vary, but disallow obvious mis-types.
        # (Still allow octet-stream because some clients don't send a precise content-type.)
        pass

    # Read frame
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # Model inference
    detector = _load_detector_for_facility(facility_id)
    t0 = time.time()
    detections_raw, det_debug = detector.infer(image_bytes, conf_threshold=confidence_threshold, iou_threshold=iou_threshold)
    t1 = time.time()

    detections: List[DetectionDTO] = [
        DetectionDTO(label=d.label, confidence=d.confidence, bbox_xyxy=[d.bbox_xyxy[0], d.bbox_xyxy[1], d.bbox_xyxy[2], d.bbox_xyxy[3]])
        for d in detections_raw
    ]

    # Resolve to contextual location (Neo4j + spatial backend abstraction)
    resolved = _resolver.resolve(facility_id=facility_id, detections=detections_raw, navigation_to=navigation_to)

    timestamp_ms = int(time.time() * 1000)
    return InferenceResponse(
        facility_id=facility_id,
        request_id=request_id,
        timestamp_unix_ms=timestamp_ms,
        detections=detections,
        location=resolved.current_location,
        route=resolved.route,
        debug={
            **det_debug,
            "total_ms": (time.time() - t0) * 1000.0,
            "resolver_debug": resolved.debug,
            "model_onnx_path": detector.onnx_path,
        },
    )
