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
 
# Run inference.py (required by OpenEnv validator)
CMD ["python", "inference.py"]
