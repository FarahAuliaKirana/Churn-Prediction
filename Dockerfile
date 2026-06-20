FROM python:3.11-slim

WORKDIR /app

# Install dependencies dulu (layer terpisah agar cache efisien)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY outputs/model_xgb.pkl       ./outputs/model_xgb.pkl
COPY outputs/feature_cols.json   ./outputs/feature_cols.json
COPY outputs/metrics.json        ./outputs/metrics.json

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]