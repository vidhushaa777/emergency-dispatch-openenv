FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Create __init__ files
RUN touch app/__init__.py

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Run the environment SERVER (needed for Phase 1 / HF Space)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
