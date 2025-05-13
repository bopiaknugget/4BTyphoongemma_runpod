# Dockerfile for RunPod Serverless with app/main.py and token_utils.py
FROM python:3.10-slim

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# 1) Set a base work dir
WORKDIR /app

# 2) Optimize CPU-side tokenization
ENV TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# 3) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4) Copy your application code
#    Expecting:
#      app/main.py
#      app/performance_utils.py
#      app/token_utils.py
COPY app/ ./app

# 5) Switch into the app folder so `python main.py` works
WORKDIR /app/app

# 6) Launch your RunPod handler
CMD ["python", "main.py"]
