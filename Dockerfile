# ── Stage 1: Build Tailwind CSS ──────────────────────────────────────────────
FROM node:24-alpine AS css-builder
WORKDIR /build
RUN npm install -D tailwindcss@3
COPY tailwind.config.js .
COPY app/static/src/input.css ./app/static/src/input.css
COPY app/templates/ ./app/templates/
RUN npx tailwindcss -i ./app/static/src/input.css -o ./app/static/css/tailwind.min.css --minify

# ── Stage 2: Python application ─────────────────────────────────────────────
FROM python:3.12-slim

# Install build essentials + curl (untuk HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK tokenizer data into default search path
ENV NLTK_DATA=/root/nltk_data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/root/nltk_data'); nltk.download('punkt_tab', download_dir='/root/nltk_data')"

COPY . .

# Copy built Tailwind CSS from stage 1
COPY --from=css-builder /build/app/static/css/tailwind.min.css ./app/static/css/tailwind.min.css

# Ensure directories exist (uploads volume-mounted in docker-compose)
RUN mkdir -p /app/uploads /app/app/static/charts /app/app/data

# Verify NLP data files are present
RUN test -f /app/app/data/inset_negative.tsv && \
    test -f /app/app/data/inset_positive.tsv && \
    test -f /app/app/data/slang_words.json && \
    test -f /app/app/data/stop_words.txt && \
    echo "NLP data files OK" || echo "WARNING: NLP data files missing"

EXPOSE 5000

# Health check untuk Docker & orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Production WSGI server – optimized for AWS Free Tier (t3.micro / 1 GB RAM)
# 1 worker + 2 threads keeps memory under 512 MB; timeout 300s for scraping
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--worker-class", "gthread", "--timeout", "300", "--max-requests", "100", "--max-requests-jitter", "20", "--access-logfile", "-", "main:app"]