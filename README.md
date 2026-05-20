# AAU SW8 ML-Vision

FastAPI service for real-time indoor object detection. Lives in its own repo and its own container — separate from the Ariadne compose stack — but the spatial backend's middleware bridges it at `/api/v1/ml-vision/*` so mobile clients reach it through the same gateway as everything else.

What it does:

- runs server-side YOLO26 ONNX inference, one model per facility
- resolves detected labels against a `LocationResolver` (optional Neo4j-backed) into a `ResolvedLocation` with `current_location` + optional `route`
- accepts both single-shot uploads (`POST /v1/detect`) and live streaming (`WS /ws/stream/{facility_id}`)
- serves platform-specific model artifacts so iOS/Android can run the same model on-device (CoreML `.mlpackage`, `.tflite`)

## Model artifacts

See [models/README.md](models/README.md) for the expected layout. Per facility:

- `models/{facility_id}/yolo26_indoor.onnx`     server-side inference
- `models/{facility_id}/yolo26_indoor.mlpackage` iOS on-device (CoreML)
- `models/{facility_id}/yolo26_indoor.tflite`   Android on-device
- `models/{facility_id}/metadata.json`          class names + version

On startup the container ensures `yolo26_indoor.onnx` exists for `FACILITY_ID`. If missing, it downloads `MODEL_ONNX_URL` into that location.

## Run with Docker

```bash
docker build -t aau-sw8-ml-vision:latest .

docker run --rm -p 8010:8000 \
  -e FACILITY_ID=default_facility \
  -e MODEL_ONNX_URL=https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx \
  -e MODEL_DIR=/app/models \
  aau-sw8-ml-vision:latest
```

Health: `GET http://localhost:8010/health`.

## API

All routes below are served directly on `:8010`. When the spatial backend's middleware is up, the same routes are reachable through the gateway under the `/api/v1/ml-vision` prefix (and require `X-Api-Key`):

| Direct                                | Through gateway                                  |
|---------------------------------------|--------------------------------------------------|
| `GET  /health`                        | `GET  /api/v1/ml-vision/health`                  |
| `POST /v1/detect`                     | `POST /api/v1/ml-vision/v1/detect`               |
| `GET  /v1/models/{facility_id}`       | `GET  /api/v1/ml-vision/v1/models/{facility_id}` |
| `GET  /v1/models/{facility_id}/{kind}`| `GET  /api/v1/ml-vision/v1/models/{facility_id}/{kind}` |
| `WS   /ws/stream/{facility_id}`       | `WS   /api/v1/ml-vision/ws/stream/{facility_id}` |

### Detect (single-shot)

`POST /v1/detect` — `multipart/form-data`:

| Field                  | Type   | Required | Description                                  |
|------------------------|--------|----------|----------------------------------------------|
| `facility_id`          | string | yes      | Picks which `models/{facility_id}/` to load  |
| `image`                | file   | yes      | JPEG or PNG                                  |
| `navigation_to`        | string | no       | Destination location id; populates `route`   |
| `request_id`           | string | no       | Echoed back for client correlation           |
| `confidence_threshold` | float  | no       | Per-request override, `0..1`                 |
| `iou_threshold`        | float  | no       | Per-request override, `0..1`                 |

Returns an `InferenceResponse` with detections + a `ResolvedLocation` (current location, optional route).

### Stream

`WS /ws/stream/{facility_id}`. Client sends raw JPEG bytes as binary messages; server responds with one JSON `StreamFrame` per frame:

```json
{
  "detections": [{"class": "door", "confidence": 0.91, "bbox": [..]}],
  "location": {
    "facility_id": "aau",
    "current_location": {"kind": "room", "id": "A101", "name": "A101", "confidence": 0.7},
    "route": null,
    "debug": {}
  }
}
```

Frames are decoded and run through the same detector + resolver as `/v1/detect`, off the event loop on a thread pool so per-frame latency doesn't block the websocket reader.

### Model download

`GET /v1/models/{facility_id}` lists what artifacts exist on disk for a facility.

`GET /v1/models/{facility_id}/{platform}` returns the file:

- `server` → `yolo26_indoor.onnx`
- `ios` / `coreml` → `.mlpackage` (zipped if it's a directory bundle)
- `android` / `tflite` → `.tflite`
- `metadata` → `metadata.json`

## Internals

```
serving/
├── main.py              FastAPI app, /health, /v1/detect, /v1/models/*, WS /ws/stream/{facility_id}
├── detector.py          ONNX session loading + cached per-facility detector instances
├── location_resolver.py LocationEntity / RouteStep / ResolvedLocation models, optional Neo4j-backed resolution
└── bootstrap_models.py  On-startup ensure-and-download for the configured facility's ONNX
```

`location_resolver.py` falls back to a stub resolver when Neo4j env vars are missing — the `/v1/detect` and stream endpoints still return a `ResolvedLocation` with a placeholder `current_location`, so clients can run end-to-end without Neo4j.

## Exposing for mobile

Either:

- bridge through the spatial backend's middleware (recommended — single gateway, single API key, single TLS termination), or
- expose `:8010` directly via something like `ngrok http 8010` for quick standalone tests from iOS.
