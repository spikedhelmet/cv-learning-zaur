FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for OpenCV and Ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Ensure huggingface cache goes to a writable directory
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV HF_HOME=/tmp/huggingface

EXPOSE 7860

# Run Uvicorn server on port 7860 (Hugging Face default)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
