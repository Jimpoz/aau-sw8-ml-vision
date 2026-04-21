# AAU SW8 ML-Vision

FastAPI service that:

- runs server-side inference from an ONNX model (per facility)
- exposes a single `/v1/detect` endpoint for mobile clients
- serves platform-specific model artifacts (CoreML `.mlpackage` for iOS, `.tflite` for Android)

## Model artifacts

See `models/README.md` for the expected folder structure:

- `models/{facility_id}/yolo26_indoor.onnx`
- `models/{facility_id}/yolo26_indoor.mlpackage`
- `models/{facility_id}/yolo26_indoor.tflite`
- `models/{facility_id}/metadata.json`

## Run with Docker

Build:

```bash
docker build -t aau-sw8-ml-vision:latest .
```

Run (recommended host port 8010 to avoid clashing with the spatial backend):

```bash
docker run --rm -p 8010:8000 ^
  -e FACILITY_ID=default_facility ^
  -e MODEL_ONNX_URL=https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx ^
  -e MODEL_DIR=/app/models ^
  aau-sw8-ml-vision:latest
```

On startup, the container will ensure `models/{FACILITY_ID}/yolo26_indoor.onnx` exists.
If it does not, it will download `MODEL_ONNX_URL` into that location.

Health:

- `GET http://localhost:8010/health`

## API

### Detect

- `POST /v1/detect`
- `Content-Type: multipart/form-data`

Fields:

- `facility_id` (string, required)
- `image` (file, required; jpg/png)
- `navigation_to` (string, optional)
- `request_id` (string, optional)
- `confidence_threshold` (float 0..1, optional)
- `iou_threshold` (float 0..1, optional)

### List/download models

- `GET /v1/models/{facility_id}`
- `GET /v1/models/{facility_id}/{platform}` where `platform` is:
  - `server` (onnx)
  - `ios` / `coreml` (mlpackage, served as a zip if it’s a directory bundle)
  - `android` / `tflite`
  - `metadata` (json)

## ngrok

Expose the service:

```bash
ngrok http 8010
```

Use the resulting public HTTPS base URL from iOS.

