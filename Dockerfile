# ─────────────────────────────────────────────────────────────────
# ScholarAI — Hugging Face Spaces Dockerfile
#
# HF Spaces requirements:
#   - Must run as non-root user (uid 1000)
#   - App MUST listen on port 7860
#   - HOME must be writable for model cache
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps: ffmpeg for moviepy, espeak for pyttsx3, fonts for PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak \
    espeak-ng \
    libglib2.0-0 \
    libgl1 \
    fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000 — create it
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps as root first (faster layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLP models
RUN python -m spacy download en_core_web_sm
RUN python -c "\
import nltk; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('averaged_perceptron_tagger')"

# Pre-download models so first request isn't slow
# Use quality tier models (BGE-base, FLAN-T5-large, BART-large-CNN)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5')" || true

RUN python -c "\
from transformers import BartTokenizer, BartForConditionalGeneration; \
BartTokenizer.from_pretrained('facebook/bart-large-cnn'); \
BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')" || true

RUN python -c "\
from transformers import T5ForConditionalGeneration, T5Tokenizer; \
T5Tokenizer.from_pretrained('google/flan-t5-large'); \
T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')" || true

# Copy app code
COPY . .

# Create writable dirs and set ownership
RUN mkdir -p uploads outputs models \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Use quality tier models (BGE-base 768-dim, FLAN-T5-large, BART-large-CNN)
ENV RAG_EMBEDDING_TIER=quality
ENV RAG_GENERATION_TIER=quality

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
