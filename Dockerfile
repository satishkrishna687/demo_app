# Dockerfile for Hugging Face Spaces (Docker SDK) — fallback if the Streamlit SDK
# option is not shown in the Space creation dialog.
# Runs the app in CLOUD_MODE with the API-only dependency set.

FROM python:3.12-slim

# Streamlit needs this; HF Spaces runs containers as a non-root user on port 7860.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install API-only dependencies (no torch / docling / sentence-transformers).
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# App code + pre-built indexes (data/indexes/pair2, pair3 are committed).
COPY config.py ./
COPY chain/ chain/
COPY graph/ graph/
COPY models/ models/
COPY retrieval/ retrieval/
COPY ui/ ui/
COPY vectorstore/ vectorstore/
COPY data/indexes/pair2 data/indexes/pair2
COPY data/indexes/pair3 data/indexes/pair3

# CLOUD_MODE is set in the Space's environment variables (recommended) —
# uncomment here as a fallback:
# ENV CLOUD_MODE=1

EXPOSE 7860

CMD ["streamlit", "run", "ui/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
