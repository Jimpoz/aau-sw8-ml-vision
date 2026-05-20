from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from serving.detector import Detection, Detector
from serving.location_resolver import LocationEntity, LocationResolver, RouteStep

_stream_executor = ThreadPoolExecutor(max_workers=2)


class StreamDetection(BaseModel):
    """Detection in Vision coordinate format for the iOS/Android live stream."""
    label: str
    confidence: float
    x: float = Field(description="Normalised left edge (0–1, left origin)")
    y: float = Field(description="Normalised bottom edge in Vision coords (0–1, bottom origin)")
    width: float = Field(description="Normalised width")
    height: float = Field(description="Normalised height")
    is_landmark_match: bool = False
    landmark_name: Optional[str] = None


class StreamFrame(BaseModel):
    detections: List[StreamDetection]
    location: LocationEntity


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


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    skip = os.getenv("SKIP_MODEL_BOOTSTRAP", "").strip().lower() in (
        "true", "1", "yes", "on"
    )
    if not skip:
        try:
            from serving.bootstrap_models import ensure_server_model
            result = ensure_server_model()
            print(f"[lifespan] model bootstrap: {result}", flush=True)
        except Exception as exc:
            print(f"[lifespan] model bootstrap failed: {exc}", flush=True)
    yield


app = FastAPI(
    title="AAU Indoor Vision Service",
    version="0.1.0",
    lifespan=_lifespan,
)

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


def _process_stream_frame(facility_id: str, image_bytes: bytes) -> dict:
    """Run inference + location resolution on a single frame. Runs in thread pool."""
    orb_result = _resolver.resolve_from_image_bytes(
        facility_id=facility_id, image_bytes=image_bytes
    )
    if orb_result is not None:
        loc = orb_result.current_location
        bbox = (loc.properties or {}).get("bbox_norm")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            box = {
                "x": round(x1, 4),
                "y": round(1.0 - y2, 4),
                "width": round(x2 - x1, 4),
                "height": round(y2 - y1, 4),
            }
        else:
            box = {"x": 0.20, "y": 0.20, "width": 0.60, "height": 0.60}
        return {
            "detections": [{
                "label": "landmark",
                "confidence": loc.confidence,
                **box,
                "is_landmark_match": True,
                "landmark_name": loc.name,
            }],
            "location": loc.model_dump(),
        }

    img = Image.open(BytesIO(image_bytes))
    orig_w, orig_h = img.size

    fake_mode = os.getenv("LANDMARKS_FAKE_DETECTION", "").strip().lower() in (
        "true", "1", "yes", "on"
    )
    if fake_mode:
        detections_raw = [
            Detection(
                label="landmark",
                confidence=0.99,
                bbox_xyxy=(
                    orig_w * 0.30,
                    orig_h * 0.30,
                    orig_w * 0.70,
                    orig_h * 0.70,
                ),
            )
        ]
    else:
        try:
            detector = _get_detector(facility_id)
        except FileNotFoundError as exc:
            return {
                "error": str(exc),
                "detections": [],
                "location": LocationEntity(
                    kind="unknown", id="unknown", name="unknown"
                ).model_dump(),
            }
        detections_raw, _ = detector.infer(image_bytes)

    stream_detections = []
    for d in detections_raw:
        x1, y1, x2, y2 = d.bbox_xyxy
        x1_n = x1 / orig_w
        y2_n = y2 / orig_h
        stream_detections.append({
            "label": d.label,
            "confidence": round(d.confidence, 3),
            "x": round(x1_n, 4),
            "y": round(1.0 - y2_n, 4),
            "width": round((x2 - x1) / orig_w, 4),
            "height": round((y2 - y1) / orig_h, 4),
            "is_landmark_match": False,
            "landmark_name": None,
        })

    resolved = _resolver.resolve(facility_id=facility_id, detections=detections_raw)

    match = _resolver.landmark_store.find_match(
        facility_id=facility_id, detections=detections_raw
    )
    if match is not None:
        landmark_name = resolved.current_location.name
        for idx in match.supporting_indices:
            if 0 <= idx < len(stream_detections):
                stream_detections[idx]["is_landmark_match"] = True
                stream_detections[idx]["landmark_name"] = landmark_name

    return {
        "detections": stream_detections,
        "location": resolved.current_location.model_dump(),
    }


@app.websocket("/ws/stream/{facility_id}")
async def stream_vision(websocket: WebSocket, facility_id: str):
    """
    Live camera feed endpoint.
    iOS/Android sends JPEG frames as binary messages.
    Server responds with a StreamFrame JSON per frame:
      { detections: [...], location: {...} }
    """
    await websocket.accept()
    try:
        _resolver.orb_matcher.invalidate(facility_id)
        loaded = _resolver.orb_matcher.entries_for(facility_id)
        print(
            f"[stream_vision:{facility_id}] WS opened — "
            f"refreshed ORB cache: {len(loaded)} landmark(s)",
            flush=True,
        )
    except Exception as exc:
        print(f"[stream_vision:{facility_id}] cache refresh failed: {exc}", flush=True)
    loop = asyncio.get_event_loop()
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            if not image_bytes:
                continue
            frame_result = await loop.run_in_executor(
                _stream_executor, _process_stream_frame, facility_id, image_bytes
            )
            await websocket.send_text(json.dumps(frame_result))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[stream_vision:{facility_id}] error: {exc}")


@app.get("/v1/landmarks/{facility_id}")
def list_facility_landmarks(facility_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Inspect the ORB matcher's cached landmarks for a facility."""
    if refresh:
        _resolver.orb_matcher.invalidate(facility_id)
    entries = _resolver.orb_matcher.entries_for(facility_id)
    return {
        "facility_id": facility_id,
        "count": len(entries),
        "min_good_matches": int(os.getenv("ORB_MIN_GOOD_MATCHES", "8")),
        "landmarks": [
            {
                "id": e.id,
                "name": e.name,
                "space_id": e.space_id,
                "building_id": e.building_id,
                "campus_id": e.campus_id,
                "keypoint_count": e.keypoint_count,
            }
            for e in entries
        ],
        "note": (
            "Empty list means the matcher hasn't found any Landmark "
            "nodes for this facility. Check that the campus_id on "
            "the registered landmark matches the WS facility_id."
        ) if not entries else None,
    }


@app.get("/v1/models/{facility_id}/classes")
def list_model_classes(facility_id: str) -> Dict[str, Any]:
    """Inspect the class vocabulary the loaded ONNX model actually
    emits."""
    try:
        detector = _get_detector(facility_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    names = detector._class_names or []
    return {
        "facility_id": facility_id,
        "effective_facility": detector._effective_facility,
        "onnx_path": detector.onnx_path,
        "metadata_path": detector._metadata_path,
        "class_count": len(names),
        "class_names": names,
        "note": (
            "Empty class_names means detections will be reported as "
            "numeric class IDs. Set MODEL_CLASS_NAMES env var or edit "
            "metadata.json to populate them."
        ) if not names else None,
    }


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
