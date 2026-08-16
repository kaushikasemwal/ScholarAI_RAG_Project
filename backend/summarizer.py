"""
summarizer.py — Summarization Pipeline
======================================
Pipeline:
  1. Split text into sentences (NLTK)
  2. Encode with Sentence-BERT (BAAI/bge-small-en-v1.5) → 384-dim embeddings
  3. Pass through Autoencoder → 128-dim compressed embeddings
  4. Select top-k representative sentences via cosine similarity to centroid
  5. Feed into Pegasus (google/pegasus-xsum) for abstractive summary

Why Pegasus over BART?
  Pegasus is pre-trained with a gap-sentence generation objective specifically
  designed for abstractive summarization. It produces more concise, fluent
  summaries than BART on document-level inputs.

Why BAAI/bge-small-en-v1.5 over MiniLM?
  BGE-small consistently outperforms all-MiniLM-L6-v2 on semantic similarity
  benchmarks (MTEB) while being only marginally larger.

Course: Advanced Topics in Machine Learning
"""

import logging
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk

log = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Import from centralized model manager
from .models import get_sbert, get_pegasus, get_autoencoder


# ─── PIPELINE STEPS ─────────────────────────────────────────────

def preprocess_text(text: str, max_sentences: int = 80) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 6]
    if len(sentences) > max_sentences:
        idx = np.linspace(0, len(sentences) - 1, max_sentences, dtype=int)
        sentences = [sentences[i] for i in idx]
    return sentences


def embed_sentences(sentences: List[str]) -> np.ndarray:
    model = get_sbert()
    # BGE models benefit from a query prefix for retrieval tasks
    embeddings = model.encode(
        sentences, batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True   # BGE recommendation
    )
    log.info(f"BGE-small: encoded {len(sentences)} sentences → {embeddings.shape}")
    return embeddings


def compress_embeddings(embeddings: np.ndarray) -> np.ndarray:
    ae = get_autoencoder(input_dim=embeddings.shape[1])
    compressed = ae.encode(embeddings)
    log.info(f"Autoencoder: compressed to {compressed.shape}")
    return compressed


def select_key_sentences(sentences: List[str],
                          compressed: np.ndarray,
                          top_k: int = 12) -> str:
    centroid = compressed.mean(axis=0, keepdims=True)
    sims     = cosine_similarity(compressed, centroid).flatten()
    top_idx  = sorted(np.argsort(sims)[::-1][:top_k].tolist())
    return " ".join(sentences[i] for i in top_idx)


def abstractive_summary(context: str,
                         max_length: int = 256,
                         min_length: int = 80) -> str:
    model, tok = get_pegasus()
    inputs = tok(context, return_tensors="pt",
                 max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=1.5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tok.decode(summary_ids[0], skip_special_tokens=True)


def generate_summary(text: str) -> tuple[str, dict]:
    """
    Generate summary with metadata about the generation process.
    
    Returns:
        tuple: (summary_text, metadata_dict)
        metadata_dict contains:
            - fallback_used: bool
            - fallback_reason: str (if fallback_used)
            - model_used: str
    """
    metadata = {"fallback_used": False, "fallback_reason": None, "model_used": "pegasus"}
    
    if not text or len(text.strip()) < 50:
        return "Insufficient text content found in document.", metadata
    try:
        sentences  = preprocess_text(text, max_sentences=80)
        if not sentences:
            return "Could not extract readable sentences from document.", metadata
        embeddings = embed_sentences(sentences)
        compressed = compress_embeddings(embeddings)
        context    = select_key_sentences(sentences, compressed, top_k=15)
        summary    = abstractive_summary(context)
        log.info(f"Summary generated: {len(summary)} chars")
        return summary, metadata
    except ImportError as e:
        log.warning(f"ML libraries missing, using extractive fallback: {e}")
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = f"ML libraries unavailable: {e}"
        metadata["model_used"] = "extractive_tfidf"
        return extractive_fallback(text), metadata
    except Exception as e:
        log.error(f"Summary pipeline error: {e}", exc_info=True)
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = f"Pipeline error: {e}"
        metadata["model_used"] = "extractive_tfidf"
        return extractive_fallback(text), metadata


def extractive_fallback(text: str, n_sentences: int = 5) -> str:
    try:
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return text[:500]
        words = nltk.word_tokenize(text.lower())
        freq: dict = {}
        for w in words:
            if w.isalpha():
                freq[w] = freq.get(w, 0) + 1
        scores = {}
        for i, sent in enumerate(sentences):
            scores[i] = sum(freq.get(w, 0) for w in nltk.word_tokenize(sent.lower()))
        top_idx = sorted(sorted(scores, key=scores.get, reverse=True)[:n_sentences])
        return " ".join(sentences[i] for i in top_idx)
    except Exception:
        return text[:800]