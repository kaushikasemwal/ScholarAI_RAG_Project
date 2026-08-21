---
title: ScholarAI Backend
emoji: 🎓
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ScholarAI — AI-Powered Study Assistant

> Built by Kaushika Semwal

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=flat-square&logo=github)](https://kaushikasemwal.github.io/ScholarAI_Project/)
[![Backend API](https://img.shields.io/badge/Backend%20API-Hugging%20Face%20Spaces-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/kaushikasemwal/ScholarAI_Project/deploy.yml?style=flat-square&logo=github)](https://github.com/kaushikasemwal/ScholarAI_Project/actions)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)

---

## What is ScholarAI?

ScholarAI transforms any PDF or PowerPoint lecture notes into a complete study package — automatically. Upload a document and receive:

| Output | Technology |
|---|---|
| **AI Summary** | BGE Embeddings → Autoencoder → Google Pegasus |
| **10-Question MCQ Quiz** | T5 Question Generation (valhalla/t5-base-qg-hl) |
| **Audio Narration** | gTTS (Google Text-to-Speech) |
| **Explainer Video** | MoviePy + PIL (content-driven duration) |

All outputs are **saved to your account** via Firebase Firestore — no re-generating needed.

---

## ✨ New Features (v2.0)

### 🔐 Authentication & Security
- **Firebase ID Token Verification** on all backend endpoints
- **User-scoped data isolation** — each user only accesses their own files
- **WebSocket authentication** for real-time progress updates

### 🎨 UI/UX Enhancements
- **Dark/Light Theme** with system preference detection & manual toggle
- **Multi-file Upload Queue** with progress bars and file validation
- **Real-time Generation Progress** via WebSocket (ETA, cancel button)
- **Interactive Quiz Engine** with keyboard navigation, instant feedback & reasoning
- **Audio Player** with speed control (0.5x–2x), keyboard shortcuts
- **Video Player** with Picture-in-Picture, keyboard controls
- **My Notes Library**: Grid/List toggle, bulk selection & delete, storage badges
- **Keyboard Shortcuts** (U=upload, Enter=generate, T=theme, /=search, G=view toggle)
- **Loading Skeletons**, Empty States, Accessible Toast Notifications
- **WCAG 2.1 AA** — Semantic HTML, focus management, screen reader announcements, reduced motion

### ⚙️ Backend Architecture
- **Redis Rate Limiting** (distributed, fail-open) with in-memory fallback
- **Storage Abstraction** — Local/S3/GCS backends via unified interface
- **Model Preloading** — SBERT at startup for faster first request
- **ML Error Transparency** — Returns `fallback_used`, `fallback_reason`, `model_used`
- **Autoencoder Validation** — Weight quality checks on load
- **File Magic Byte Validation** — PDF (`%PDF`), PPTX (`PK\x03\x04`)

---

## Live Links

| Service | URL |
|---|---|
| **Frontend** | https://kaushikasemwal.github.io/ScholarAI_Project/ |
| **Backend API** | `https://your-hf-username-scholarai-backend.hf.space/docs` |
| **API Health** | `https://your-hf-username-scholarai-backend.hf.space/health` |
| **Metrics** | `https://your-hf-username-scholarai-backend.hf.space/metrics` |

---

## ML Pipeline

```
PDF / PPTX Upload
      │
      ▼ AES-256 Fernet Encryption
Text Extraction ── PyMuPDF / python-pptx / pdfplumber
      │
      ▼ NLTK Sentence Tokenization
BGE Embeddings ── BAAI/bge-base-en-v1.5 → 768-dim vectors
      │
      ▼ Advanced ML Component
Semantic Autoencoder ── 768 → 256-dim latent space (PyTorch)
      │
      ▼ Centroid-based Sentence Selection
      │
      ├── BART-large-CNN (facebook/bart-large-cnn) ─── AI Summary
      ├── FLAN-T5-large (google/flan-t5-large) ────── MCQ Quiz
      ├── gTTS ────────────────────────────────────── Audio MP3
      └── MoviePy + PIL ───────────────────────────── Video MP4
                │
                ▼
      Firebase Firestore ── Persist per user session
```

### Why BART-large-CNN over Pegasus?
BART-large-CNN is fine-tuned on CNN/DailyMail (300K news articles) for summarization,
providing stronger factual consistency and better generalization to diverse document types.
It also supports 1024-token context and is more widely benchmarked.

### Why FLAN-T5-large over T5-base-qg-hl?
FLAN-T5 is instruction-tuned on 1.8K tasks including question generation,
generalizing better to new domains without requiring specialized `<hl>` answer highlighting.
It produces higher quality, more diverse questions.

### Why BAAI/bge-base-en-v1.5 over BGE-small?
BGE-base (768-dim, 110M params) significantly outperforms BGE-small (384-dim, 33M params)
on MTEB benchmarks (avg +5-8% on semantic similarity tasks) while remaining efficient
for CPU inference (~2x slower, ~3x memory).

### Autoencoder Architecture

The autoencoder is the centrepiece of the ML contribution. It learns a compressed, denoised representation of sentence embeddings — removing redundant dimensions and surfacing core semantic signals before BART summarization.

```
Encoder                                Decoder
────────────────────────               ────────────────────────
Input:  768-dim (BGE-base)             Latent: 256-dim
Dense:  512  + ReLU + BatchNorm        Dense:  256  + ReLU + BatchNorm
Dense:  256  + ReLU + BatchNorm        Dense:  512  + ReLU + BatchNorm
Dense:  256  + Tanh  ← latent ──────►  Output: 768-dim
```

**Training objective:** MSE reconstruction loss  
**Optimizer:** Adam with gradient clipping  
**Compression ratio:** 3:1 (768 → 256 dimensions)

### Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| Reconstruction MSE | Autoencoder fidelity | < 0.01 |
| Cosine Similarity | Semantic preservation | > 0.95 |
| Compression Ratio | Dimensionality reduction | 3× |
| ROUGE-1 | Summary word overlap | > 0.40 |
| ROUGE-2 | Bigram overlap | > 0.20 |
| ROUGE-L | Longest common subsequence | > 0.35 |

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Authentication | Firebase Admin SDK (ID token verification) |
| Rate Limiting | Redis (distributed) + in-memory fallback |
| Summarization | facebook/bart-large-cnn |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Autoencoder | PyTorch (custom architecture) |
| Quiz Generation | google/flan-t5-large |
| Audio | gTTS (Google Text-to-Speech) |
| Video | MoviePy + Pillow |
| PDF Parsing | PyMuPDF (fitz) + pdfplumber |
| PPTX Parsing | python-pptx |
| NLP Utilities | NLTK, spaCy, scikit-learn |
| Encryption | cryptography (AES-256 Fernet) |
| Storage | Local / S3 / GCS (pluggable) |
| Observability | Prometheus metrics, structured JSON logging, health checks |
| Real-time | WebSocket (FastAPI native) |

### Frontend
| Component | Technology |
|---|---|
| Auth | Firebase Authentication (Google + Email) |
| Database | Firebase Firestore |
| Hosting | GitHub Pages |
| UI | Vanilla HTML / CSS / JavaScript (ES Modules) |
| Real-time | WebSocket (native) |
| Styling | CSS Custom Properties (themeable) |

### Infrastructure
| Component | Technology |
|---|---|
| Backend Hosting | Hugging Face Spaces (Docker, CPU Basic) |
| Frontend Hosting | GitHub Pages |
| CI/CD (Primary) | Jenkins → Azure Container Apps |
| CI/CD (Alt) | GitHub Actions → GitHub Pages + HF Spaces |
| Container Registry | Azure Container Registry (ACR) |

---

## Project Structure

```
ScholarAI_Project/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions: deploy to GH Pages + HF Spaces
│
├── backend/
│   ├── __init__.py
│   ├── app.py                  # FastAPI routes + lifespan + WebSocket
│   ├── auth.py                 # Firebase Admin SDK verification
│   ├── config.py               # Pydantic Settings (env-driven)
│   ├── schemas.py              # Pydantic models (request/response)
│   ├── summarizer.py           # BGE + Autoencoder + Pegasus pipeline
│   ├── autoencoder.py          # PyTorch autoencoder (core ML component)
│   ├── quiz_generator.py       # T5-based MCQ generation with distractors
│   ├── tts_generator.py        # gTTS audio (no ffmpeg required)
│   ├── video_generator.py      # MoviePy slide video (content-driven duration)
│   ├── utils.py                # AES-256 encryption, PDF/PPTX extraction
│   ├── rate_limit.py           # Redis distributed rate limiter
│   ├── websocket.py            # WebSocket connection manager
│   ├── models/                 # Model manager (lazy loading, preloading)
│   ├── storage/                # Storage abstraction (local/S3/GCS)
│   └── observability/          # Metrics, logging, health checks
│
├── frontend/
│   ├── login.html              # Firebase auth page
│   ├── index.html              # Upload + generate page
│   ├── my-notes.html           # Notes library (grid/list, bulk actions)
│   ├── notes.html              # Session view (tabs: summary/quiz/audio/video)
│   ├── script.js               # Upload, queue, WebSocket progress, generate
│   ├── notes.js                # Interactive quiz, audio/video players
│   ├── my-notes.js             # Search, filter, grid/list, bulk delete
│   ├── auth.js                 # Firebase auth handlers
│   ├── firebase-config.js      # Firebase config (gitignored — injected at deploy)
│   ├── firebase-config.example.js
│   └── shared/
│       ├── api.js              # API client + WebSocket helper
│       ├── auth-guard.js       # Auth state management
│       ├── toast.js            # Accessible toast notifications
│       └── utils.js            # Formatters, helpers
│   └── styles.css              # Themeable, accessible, responsive
│
├── tests/
│   ├── test_api.py             # pytest API tests
│   ├── test_model_accuracy.py  # ML evaluation tests
│   └── test_observability.py   # Metrics/logging/health tests
│
├── models/                     # Autoencoder weights (.pt) — auto-generated
├── outputs/                    # Generated audio/video files
├── uploads/                    # Encrypted uploaded files (auto-deleted 1hr)
│
├── Dockerfile                  # HF Spaces Docker config (port 7860, user 1000)
├── Jenkinsfile                 # Jenkins CI/CD pipeline (6 stages)
├── requirements.txt            # Python dependencies (CPU-only torch)
├── pyproject.toml              # Project config (ruff, mypy, pytest, coverage)
├── firebase.rules              # Firestore security rules
└── FIREBASE_SETUP.md           # Step-by-step Firebase setup guide
```

---

## Local Development

### Prerequisites

- Python 3.11+
- Redis (optional, for distributed rate limiting)
- A Firebase project (see [FIREBASE_SETUP.md](FIREBASE_SETUP.md))

### Backend Setup

```bash
# 1. Clone the repo
git clone https://github.com/kaushikasemwal/ScholarAI_Project.git
cd ScholarAI_Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 6. (Optional) Start Redis for rate limiting
redis-server

# 7. Start the backend
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs: `http://localhost:8000/docs`  
Health: `http://localhost:8000/health`  
Metrics: `http://localhost:8000/metrics`

### Frontend Setup

```bash
# Copy and configure Firebase credentials
cp frontend/firebase-config.example.js frontend/firebase-config.js
# Edit firebase-config.js with your Firebase project values

# Serve the frontend (must use a server, not file://)
python -m http.server 3000 --directory frontend
```

Open: `http://localhost:3000/login.html`

### Running Tests

```bash
pip install pytest httpx
pytest tests/ -v

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

### Code Quality

```bash
# Lint
ruff check backend/ frontend/

# Type check
mypy backend/

# Format
ruff format backend/ frontend/
```

---

## Firebase Setup

See **[FIREBASE_SETUP.md](FIREBASE_SETUP.md)** for the full step-by-step guide.

Quick summary:
1. Create a project at [console.firebase.google.com](https://console.firebase.google.com)
2. Register a Web app → copy the config into `frontend/firebase-config.js`
3. Enable **Authentication** → Google + Email/Password
4. Create **Firestore Database** in test mode
5. Add a composite index: `sessions` collection → `uid` (Asc) + `createdAt` (Desc)

---

## Deployment

### Automated via GitHub Actions

Every push to `main` automatically:
1. Deploys the frontend to **GitHub Pages**
2. Pushes the backend to **Hugging Face Spaces**

Required GitHub Secrets:

| Secret | Description |
|---|---|
| `HF_TOKEN` | Hugging Face write token (Settings → Access Tokens) |
| `HF_SPACE_ID` | `your-username/ScholarAI-backend` |
| `HF_SPACE_URL` | `https://your-username-scholarai-backend.hf.space` |
| `FIREBASE_CONFIG_JSON` | Firebase config as a single-line JSON string |

The workflow:
- Builds frontend and injects `HF_SPACE_URL` into `<meta name="api-base">` tags
- Generates `frontend/firebase-config.js` from `FIREBASE_CONFIG_JSON`
- Deploys frontend to `gh-pages` branch (GitHub Pages)
- Pushes backend to Hugging Face Spaces

### Jenkins (Alternative CI/CD)

```bash
# Configure Jenkins with credentials:
# AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AES_KEY
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/health` | Full health check |
| `GET` | `/health/live` | Kubernetes liveness probe |
| `GET` | `/health/ready` | Kubernetes readiness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/upload` | Upload PDF/PPTX (AES-256 encrypted) |
| `POST` | `/generate/summary` | Run BGE → Autoencoder → Pegasus pipeline |
| `POST` | `/generate/quiz` | Generate 10 MCQ questions via T5 |
| `POST` | `/generate/audio` | Generate MP3 narration via gTTS |
| `POST` | `/generate/video` | Generate MP4 explainer video |
| `GET` | `/media/{filename}` | Serve generated media files |
| `GET` | `/media-check/{filename}` | Debug: verify file exists + size |
| `DELETE` | `/cleanup/{file_id}` | Delete uploaded file and outputs |
| `POST` | `/jobs` | Create async generation job |
| `GET` | `/jobs/{job_id}` | Poll job status (with progress) |
| `WS` | `/ws/jobs/{job_id}` | Real-time progress WebSocket |

### Response Format (Summary)

```json
{
  "file_id": "uuid",
  "summary": "Generated text...",
  "status": "ok",
  "fallback_used": false,
  "fallback_reason": null,
  "model_used": "pegasus"
}
```

---

## Security

- **AES-256 Encryption** (Fernet) — files encrypted before disk write
- **UUID-only filenames** — original names never stored
- **Auto-deletion** — files removed after 1 hour via background thread
- **Firebase Auth** — ID token verification on all endpoints
- **User Isolation** — Firestore rules + backend ownership checks
- **Secrets Management** — API keys injected at deploy time via GitHub Secrets
- **Non-root Docker** — runs as uid 1000
- **Rate Limiting** — Redis-backed, per-IP, per-endpoint
- **File Validation** — Magic byte checks (PDF: `%PDF`, PPTX: `PK\x03\x04`)

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `U` | Open file upload |
| `Enter` | Generate All |
| `T` | Toggle theme |
| `?` / `/` | Show help / focus search |
| `Esc` | Close modals / clear selection |
| `G` | Toggle grid/list view (My Notes) |
| `Ctrl+A` | Select all (My Notes) |
| `Space` / `K` | Play/Pause (Audio/Video) |
| `←` / `→` | Seek ±10s |
| `↑` / `↓` | Volume ±10% |
| `M` | Mute |
| `F` | Fullscreen (Video) |

---

## Author

**Kaushika Semwal**  
Advanced Topics in Machine Learning (HTML Course Project)

---

*Stop Googling. Start ScholarAI-ing.*