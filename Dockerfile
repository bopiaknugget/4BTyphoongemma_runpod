# Dockerfile for RunPod Serverless with performance_utils.py and token_utils.py
FROM python:3.10-slim

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Optimize CPU tokenization performance
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the serverless handler
CMD ["python", "main.py"]
