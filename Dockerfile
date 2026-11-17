FROM python:3.12-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model at build time so the first upload is fast.
COPY app/config.py app/__init__.py app/
RUN python -c "from fastembed import TextEmbedding; from app import config; TextEmbedding(model_name=config.EMBEDDING_MODEL)"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
