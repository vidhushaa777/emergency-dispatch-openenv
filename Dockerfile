FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project definition and generate lockfile
COPY pyproject.toml .
RUN uv lock && uv sync --no-dev

# Copy source
COPY app/ ./app/
COPY server/ ./server/
COPY inference.py .

# Create __init__ files
RUN touch app/__init__.py server/__init__.py

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
