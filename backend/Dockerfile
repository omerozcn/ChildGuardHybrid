# Derleme bağlamı depo köküdür (docker-compose.yml: context: .)
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY childguard/ ./childguard/
COPY backend/api/ ./api/
ENV PYTHONPATH=/app MODEL_DIR=/app/final_models/v4
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
