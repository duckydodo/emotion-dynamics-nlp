FROM python:3.12-slim

WORKDIR /app

# Install the project's existing, pinned dependencies.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Copy only the files needed to run the existing API and its local models.
COPY src/ ./src/
COPY static/ ./static/
COPY checkpoints/transformer_baseline/best/ ./checkpoints/transformer_baseline/best/
COPY checkpoints/transformer_context_1/best/ ./checkpoints/transformer_context_1/best/
COPY checkpoints/transformer_context_1_speaker/best/ ./checkpoints/transformer_context_1_speaker/best/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
