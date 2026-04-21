FROM python:3.11-slim

WORKDIR /app

# System deps for image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY serving/ /app/serving/
COPY models/ /app/models/

ENV PORT=8000 \
    MODEL_DIR=/app/models \
    FACILITY_ID=default_facility \
    MODEL_ONNX_URL=https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx \
    NEO4J_URI="" \
    NEO4J_USER="" \
    NEO4J_PASSWORD="" \
    SUPABASE_DATABASE_URL="" \
    SUPABASE_DB_PASSWORD="" \
    LOG_LEVEL=info

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
  CMD python -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=2).status_code)" || exit 1

CMD ["sh", "-c", "python -c \"from serving.bootstrap_models import ensure_server_model; print(ensure_server_model())\" && uvicorn serving.main:app --host=0.0.0.0 --port=8000"]