FROM python:3.12-slim

WORKDIR /app

# Install dependencies using CPU-only PyTorch wheel to prevent CUDA bloat
COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0 --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-api.txt

# Copy application code, static UI, and benchmark research artifacts
COPY src/ ./src/
COPY static/ ./static/
COPY outputs/tables/ ./outputs/tables/
COPY outputs/experiments/dialoguernn_unweighted/confusion_matrix.json ./outputs/experiments/dialoguernn_unweighted/confusion_matrix.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
