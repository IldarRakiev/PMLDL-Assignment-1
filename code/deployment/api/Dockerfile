# code/deployment/api/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

# copy model at build time if you want; usually keep it mounted or included in image
# COPY ../../models/wine_rf.joblib /app/models/wine_rf.joblib

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
