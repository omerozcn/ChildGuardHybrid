FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ ./api/
RUN mkdir -p /app/final_models
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
