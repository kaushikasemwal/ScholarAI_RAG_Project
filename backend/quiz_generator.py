"""
quiz_generator.py — FLAN-T5-Based MCQ Quiz Generation
========================================================
Generates multiple-choice questions with:
  - 4 answer options (A/B/C/D)
  - 1 correct answer
  - Reasoning/explanation for the correct answer

Pipeline:
  1. Split text into meaningful chunks
  2. Use FLAN-T5 (google/flan-t5-large) to generate questions from each chunk
     using instruction-tuned prompting (no <hl> tags needed)
  3. Use the source chunk to derive distractors via keyword extraction
  4. Use BGE embeddings to select the most diverse final set of questions

Why FLAN-T5 over T5-base-qg-hl?
  FLAN-T5 is instruction-tuned on 1.8K tasks including question generation.
  It generalizes better to new domains and produces higher quality questions
  without requiring the specialized <hl> answer highlighting format.

Course: Advanced Topics in Machine Learning
"""

import logging
import random
import re

import nltk
import numpy as np

log = logging.getLogger(__name__)

for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

# Import from centralized model manager
from .models import get_sbert, get_t5

# ─── TEXT CHUNKING ───────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    """Split text into overlapping chunks suitable for T5 input."""
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent
    if current.strip():
        chunks.append(current.strip())
    return chunks[:20]  # cap at 20 chunks for speed


# ─── FLAN-T5 QUESTION GENERATION ─────────────────────────────────

def _generate_question_flan_t5(chunk: str, model, tok) -> str | None:
    """
    Generate a question using FLAN-T5's instruction-tuned capability.
    FLAN-T5 works best with direct instructions rather than <hl> highlighting.
    """
    # FLAN-T5 prompt: direct instruction for question generation
    prompt = (
        "Generate a clear, specific multiple-choice question based on the following text. "
        "The question should test understanding of a key concept. "
        "End with a question mark.\n\n"
        f"Text: {chunk}\n\nQuestion:"
    )
    
    inputs = tok(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        inputs["input_ids"],
        max_length=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        do_sample=True,
    )
    question = tok.decode(outputs[0], skip_special_tokens=True).strip()

    if not question or len(question.split()) < 4:
        return None
    if not question.endswith("?"):
        question += "?"
    return question


def _extract_keywords(text: str, n: int = 10) -> list[str]:
    """Extract top TF-IDF keywords from a chunk for distractor generation."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 2:
            sentences = [text]
        tfidf = TfidfVectorizer(max_features=n, stop_words="english", ngram_range=(1, 2))
        tfidf.fit(sentences)
        return list(tfidf.vocabulary_.keys())
    except Exception:
        words = [w for w in text.split() if len(w) > 4]
        return list(set(words))[:n]


def _generate_distractors(correct: str, context: str, n: int = 3) -> list[str]:
    """
    Generate plausible wrong answers from the same context chunk.
    Uses keyword extraction and BGE embeddings to find semantically related 
    but incorrect options (cosine similarity 0.3-0.7 = good distractors).
    """
    keywords = _extract_keywords(context, n=20)

    # Filter out keywords too similar to the correct answer
    candidate_distractors = [
        kw.title() for kw in keywords
        if kw.lower() not in correct.lower()
        and correct.lower() not in kw.lower()
        and len(kw) > 3
    ]

    # If we have SBERT available, use embeddings to find good distractors
    # Good distractors should be semantically related but not identical (cosine 0.3-0.7)
    try:
        model = get_sbert()
        correct_emb = model.encode([correct], convert_to_numpy=True, normalize_embeddings=True)
        
        scored_distractors = []
        for d in candidate_distractors:
            dist_emb = model.encode([d], convert_to_numpy=True, normalize_embeddings=True)
            cos_sim = float(np.dot(correct_emb[0], dist_emb[0]))
            # Good distractor: related but not too similar (0.3-0.7)
            if 0.3 <= cos_sim <= 0.7:
                scored_distractors.append((cos_sim, d))
        
        # Sort by similarity (closer to 0.5 is better)
        scored_distractors.sort(key=lambda x: abs(x[0] - 0.5))
        distractors = [d for _, d in scored_distractors[:n]]
    except Exception:
        # Fallback: random selection
        random.shuffle(candidate_distractors)
        distractors = candidate_distractors[:n]

    # Pad with generic placeholders if not enough distractors
    placeholders = [
        "None of the above",
        "All of the above",
        "Cannot be determined",
        "Not mentioned in the text"
    ]
    while len(distractors) < n:
        p = placeholders.pop(0) if placeholders else f"Option {len(distractors)+1}"
        distractors.append(p)

    return distractors[:n]


def _generate_reasoning(question: str, correct: str, context: str) -> str:
    """
    Generate a brief explanation for why the correct answer is right,
    grounded in the source context.
    """
    # Find the sentence in context that best supports the answer
    sentences = nltk.sent_tokenize(context)
    for sent in sentences:
        if correct.lower() in sent.lower():
            return f'"{sent.strip()}" — this directly supports the answer.'
    # Fallback: generic reasoning
    return f'According to the document, "{correct}" is the correct answer based on the provided context.'


def _flan_t5_generate_questions(chunks: list[str]) -> list[dict]:
    """Use FLAN-T5 to generate questions from text chunks."""
    try:
        model, tok = get_t5()
        qa_pairs = []

        for chunk in chunks:
            if len(chunk.split()) < 15:
                continue

            # Generate question using FLAN-T5 instruction format
            question = _generate_question_flan_t5(chunk, model, tok)
            if not question:
                continue

            # Extract answer from the chunk using keywords
            keywords = _extract_keywords(chunk, n=5)
            if not keywords:
                continue

            answer_span = keywords[0]

            # Check the answer span actually appears in the chunk
            if answer_span.lower() not in chunk.lower():
                for kw in keywords:
                    if kw.lower() in chunk.lower():
                        answer_span = kw
                        break
                else:
                    continue

            correct_answer = answer_span.title()
            distractors    = _generate_distractors(correct_answer, chunk, n=3)
            options        = [correct_answer] + distractors
            random.shuffle(options)
            correct_idx    = options.index(correct_answer)
            reasoning      = _generate_reasoning(question, correct_answer, chunk)

            qa_pairs.append({
                "question":     question,
                "options":      options,
                "correct":      correct_idx,   # index into options list
                "answer":       correct_answer,
                "reasoning":    reasoning
            })

        return qa_pairs

    except Exception as e:
        log.warning(f"FLAN-T5 question generation failed: {e}")
        return []

        scored_distractors = []
        for d in candidate_distractors:
            dist_emb = model.encode([d], convert_to_numpy=True, normalize_embeddings=True)
            cos_sim = float(np.dot(correct_emb[0], dist_emb[0]))
            # Good distractor: related but not too similar (0.3-0.7)
            if 0.3 <= cos_sim <= 0.7:
                scored_distractors.append((cos_sim, d))

        # Sort by similarity (closer to 0.5 is better)
        scored_distractors.sort(key=lambda x: abs(x[0] - 0.5))
        distractors = [d for _, d in scored_distractors[:n]]
    except Exception:
        # Fallback: random selection
        random.shuffle(candidate_distractors)
        distractors = candidate_distractors[:n]

    # Pad with generic placeholders if not enough distractors
    placeholders = [
        "None of the above",
        "All of the above",
        "Cannot be determined",
        "Not mentioned in the text"
    ]
    while len(distractors) < n:
        p = placeholders.pop(0) if placeholders else f"Option {len(distractors)+1}"
        distractors.append(p)

    return distractors[:n]


def _generate_reasoning(question: str, correct: str, context: str) -> str:
    """
    Generate a brief explanation for why the correct answer is right,
    grounded in the source context.
    """
    # Find the sentence in context that best supports the answer
    sentences = nltk.sent_tokenize(context)
    for sent in sentences:
        if correct.lower() in sent.lower():
            return f'"{sent.strip()}" — this directly supports the answer.'
    # Fallback: generic reasoning
    return f'According to the document, "{correct}" is the correct answer based on the provided context.'


# Old _t5_generate_questions removed - replaced by _flan_t5_generate_questions
# which uses FLAN-T5's instruction-tuned prompting (no <hl> tags needed)


# ─── FALLBACK: RULE-BASED MCQ ────────────────────────────────────

def _fallback_mcq(text: str, n: int = 10) -> list[dict]:
    """
    Rule-based MCQ fallback when T5 is unavailable.
    Uses definition patterns to extract Q&A pairs and wraps them as MCQ.
    """
    patterns = [
        r"(.+?)\s+is\s+((?:a|an|the)\s+.+?)[\.\,]",
        r"(.+?)\s+refers to\s+(.+?)[\.\,]",
        r"(.+?)\s+are\s+([\w\s]+?)[\.\,]",
    ]
    qa_pairs = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            subject = match.group(1).strip()
            defn    = match.group(2).strip()
            if 2 < len(subject.split()) < 8 and len(defn.split()) > 2:
                correct = defn.capitalize()
                distractors = _generate_distractors(correct, text[:1000], n=3)
                options = [correct] + distractors
                random.shuffle(options)
                correct_idx = options.index(correct)
                qa_pairs.append({
                    "question":  f"What is {subject}?",
                    "options":   options,
                    "correct":   correct_idx,
                    "answer":    correct,
                    "reasoning": f"{subject.capitalize()} is defined as: {correct}."
                })

    # Pad with generic questions if needed
    sentences = [s for s in nltk.sent_tokenize(text) if len(s.split()) > 10]
    for i, sent in enumerate(sentences[:n]):
        if len(qa_pairs) >= n:
            break
        keywords = _extract_keywords(sent, n=4)
        if not keywords:
            continue
        correct = keywords[0].title()
        distractors = _generate_distractors(correct, sent, n=3)
        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        qa_pairs.append({
            "question":  f'Fill in the blank: "{sent.replace(keywords[0], "______", 1)}"',
            "options":   options,
            "correct":   correct_idx,
            "answer":    correct,
            "reasoning": f'The correct term is "{correct}" based on the document context.'
        })

    return qa_pairs[:n]


# ─── DIVERSITY FILTER ────────────────────────────────────────────

def _diversify(qa_pairs: list[dict], n: int = 10) -> list[dict]:
    """Select n maximally diverse questions using BGE embeddings."""
    if len(qa_pairs) <= n:
        return qa_pairs
    try:
        model = get_sbert()  # Use shared SBERT instance
        qs     = [p["question"] for p in qa_pairs]
        embeds = model.encode(qs, convert_to_numpy=True, normalize_embeddings=True)

        from sklearn.metrics.pairwise import cosine_similarity
        selected = [0]
        while len(selected) < n and len(selected) < len(qa_pairs):
            remaining = [i for i in range(len(qa_pairs)) if i not in selected]
            if not remaining:
                break
            sel_embs = embeds[selected]
            scores   = [(cosine_similarity(embeds[i:i+1], sel_embs).max(), i)
                        for i in remaining]
            scores.sort()
            selected.append(scores[0][1])
        return [qa_pairs[i] for i in selected]
    except Exception:
        seen, unique = set(), []
        for p in qa_pairs:
            key = p["question"][:40].lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)
        random.shuffle(unique)
        return unique[:n]


# ─── MASTER FUNCTION ─────────────────────────────────────────────

def generate_quiz(text: str, n: int = 10) -> list[dict]:
    """
    Generate n MCQ questions from document text.

    Returns list of:
    {
      question:  str,
      options:   [str, str, str, str],   # 4 choices
      correct:   int,                    # index of correct option (0-3)
      answer:    str,                    # correct answer text
      reasoning: str                     # explanation shown on wrong answer
    }
    """
    if not text or len(text.strip()) < 100:
        return [{
            "question":  "Document too short to generate a meaningful quiz.",
            "options":   ["Upload a longer document", "N/A", "N/A", "N/A"],
            "correct":   0,
            "answer":    "Upload a longer document",
            "reasoning": "Please upload a document with more content."
        }]

    chunks   = _chunk_text(text, chunk_size=400)
    all_qa   = _flan_t5_generate_questions(chunks)
    log.info(f"FLAN-T5 generated {len(all_qa)} questions")

    # Fallback if T5 didn't produce enough
    if len(all_qa) < n:
        fallback = _fallback_mcq(text, n=n - len(all_qa) + 3)
        all_qa.extend(fallback)
        log.info(f"Fallback added {len(fallback)} questions, total: {len(all_qa)}")

    final = _diversify(all_qa, n=n)

    # Ensure exactly n questions
    while len(final) < n:
        final.append({
            "question":  f"Q{len(final)+1}: What is a key concept in this document?",
            "options":   ["Refer to the summary", "Not covered", "See document", "N/A"],
            "correct":   0,
            "answer":    "Refer to the summary",
            "reasoning": "Review the AI-generated summary for the main concepts."
        })

    log.info(f"Quiz ready: {len(final)} MCQ questions")
    return final[:n]
