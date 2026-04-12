FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

RUN touch app/__init__.py

EXPOSE 8000

# Start env server on 8000, wait for it, then run inference
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    sleep 3 && \
    python inference.py
