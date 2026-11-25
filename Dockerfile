FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    DEPLOYMENT_ENV=cloud \
    OMP_NUM_THREADS=2

WORKDIR /app

# OpenCV / MediaPipe system runtime dependencies (no GUI X11 needed beyond GL/glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY web_app.py ./
COPY src/ ./src/
COPY web/ ./web/
COPY models/ ./models/
COPY demo_images/ ./demo_images/

RUN mkdir -p /app/calibration_data

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)" || exit 1

CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
