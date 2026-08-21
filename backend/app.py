"""
ScholarAI — Backend API
========================
FastAPI application that orchestrates the full ML pipeline.

Endpoints:
  POST /upload               → Upload & encrypt PDF/PPTX
  POST /generate/summary     → Run summarization pipeline
  POST /generate/quiz        → Run quiz generation
  POST /generate/audio       → Run text-to-speech
  POST /generate/video       → Run video generation
  GET  /media/{filename}     → Served automatically via StaticFiles mount
  GET  /media-check/{filename} → Debug: verify file exists + size
  DELETE /cleanup/{file_id}  → Manual cleanup

Author : ScholarAI Project
Course : Advanced Topics in Machine Learning (HTML)
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .auth import CurrentUser, require_auth
from .config import get_settings
from .models import register_default_models
from .observability import (
    RequestIDMiddleware,
    get_health_checker,
    get_logger,
    get_metrics_collector,
    record_generation,
    record_upload,
    setup_logging,
)
from .observability.health import HealthStatus
from .quiz_generator import generate_quiz
from .rate_limit import get_rate_limiter
from .schemas import (
    AudioResponse,
    CleanupResponse,
    ErrorResponse,
    GenerateRequest,
    GenerationStatus,
    HealthResponse,
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    MediaCheckResponse,
    QuizResponse,
    SummaryResponse,
    Tags,
    UploadResponse,
    VideoResponse,
)
from .storage import get_storage
from .summarizer import generate_summary
from .tts_generator import generate_audio
from .utils import decrypt_file, encrypt_file
from .video_generator import generate_video
from .websocket import get_ws_manager

# ─── SETTINGS ──────────────────────────────────────────────────────
settings = get_settings()

# ─── LOGGING ─────────────────────────────────────────────────────
setup_logging(
    level=getattr(logging, settings.LOG_LEVEL),
    json_format=settings.LOG_JSON,
    service_name=settings.LOG_SERVICE_NAME
)
log = get_logger(__name__)

# ─── RATE LIMITER ────────────────────────────────────────────────
# Redis-based rate limiter (falls back to in-memory if Redis unavailable)
_rate_limiter_instance = None

def get_app_rate_limiter():
    """Get or create the Redis rate limiter instance."""
    global _rate_limiter_instance
    if _rate_limiter_instance is None and settings.REDIS_ENABLED:
        try:
            _rate_limiter_instance = get_rate_limiter()
        except Exception as e:
            log.warning(f"Failed to initialize Redis rate limiter, using in-memory fallback: {e}")
    return _rate_limiter_instance

# Initialize rate limiter
get_app_rate_limiter()

# ─── DIRECTORIES ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / settings.UPLOAD_DIR
OUTPUT_DIR = BASE_DIR / settings.OUTPUT_DIR
MODELS_DIR = BASE_DIR / settings.MODELS_DIR

# Always create dirs before anything else — StaticFiles mount will fail
# silently if OUTPUT_DIR doesn't exist at startup
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

log.info(f"Upload directory : {UPLOAD_DIR} (exists={UPLOAD_DIR.exists()})")
log.info(f"Output directory : {OUTPUT_DIR} (exists={OUTPUT_DIR.exists()})")

# Storage backend
_storage = None

def get_storage_backend():
    """Get the storage backend instance."""
    global _storage
    if _storage is None:
        _storage = get_storage()
    return _storage

# In-memory store: file_id → { storage_key, created_at, extracted_text, user_id }
FILE_STORE: dict[str, dict] = {}
FILE_STORE_LOCK = threading.Lock()
AUTO_DELETE_SECONDS = settings.AUTO_DELETE_HOURS * 3600
MAX_UPLOAD_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ─── CORS CONFIG ─────────────────────────────────────────────────
ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
if ALLOWED_ORIGINS == ["*"]:
    log.warning("CORS allowing all origins (set ALLOWED_ORIGINS env var for production)")

# ─── RATE LIMITER DEPENDENCY ──────────────────────────────────────
def create_rate_limit_dependency(limit: str):
    """Create a rate limit dependency with a specific limit."""
    async def rate_limit_dependency(request: Request):
        # Parse limit string like "100/minute" -> (100, 60)
        try:
            parts = limit.split("/")
            max_requests = int(parts[0])
            window_str = parts[1] if len(parts) > 1 else "minute"
            window_map = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
            window_seconds = window_map.get(window_str, 60)
        except Exception:
            max_requests, window_seconds = 100, 60

        # Use client IP as identifier
        identifier = request.client.host if request.client else "unknown"
        endpoint = request.url.path

        limiter = get_app_rate_limiter()
        if limiter:
            result = await limiter.check_limit(identifier, endpoint, max_requests, window_seconds)
            if not result.allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {int(result.retry_after or 0)} seconds.",
                    headers={"Retry-After": str(int(result.retry_after or window_seconds))}
                )
        # If Redis unavailable, allow request (fail open)
    return rate_limit_dependency


# Pre-configured rate limit dependencies
rate_limit_default = create_rate_limit_dependency("100/minute")
rate_limit_upload = create_rate_limit_dependency("10/minute")
rate_limit_cleanup = create_rate_limit_dependency("30/minute")
rate_limit_jobs = create_rate_limit_dependency("10/minute")

# ─── THREAD POOL FOR CPU-BOUND TASKS ─────────────────────────────
# Separate thread pool for ML generation to avoid blocking event loop
_generation_executor = ThreadPoolExecutor(
    max_workers=settings.GENERATION_WORKERS,
    thread_name_prefix="generation"
)

# ─── JOB QUEUE FOR ASYNC GENERATION ──────────────────────────────
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationJob(BaseModel):
    job_id: str
    file_id: str
    generation_type: str  # summary, quiz, audio, video
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict | None = None
    error: str | None = None
    user_id: str = ""  # Owner of this job


JOB_STORE: dict[str, GenerationJob] = {}
JOB_STORE_LOCK = threading.Lock()
MAX_JOB_AGE_SECONDS = 86400  # 24 hours


def _cleanup_old_jobs():
    """Clean up old completed/failed jobs."""
    now = time.time()
    with JOB_STORE_LOCK:
        expired = [
            jid for jid, job in list(JOB_STORE.items())
            if now - job.created_at > MAX_JOB_AGE_SECONDS
        ]
        for jid in expired:
            del JOB_STORE[jid]
        if expired:
            log.info(f"Cleaned {len(expired)} old jobs")


def create_job(file_id: str, generation_type: str, user_id: str) -> GenerationJob:
    """Create a new generation job."""
    job_id = str(uuid.uuid4())
    job = GenerationJob(
        job_id=job_id,
        file_id=file_id,
        generation_type=generation_type,
        user_id=user_id,
    )
    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = job
    return job


def get_job(job_id: str) -> GenerationJob | None:
    """Get job by ID."""
    with JOB_STORE_LOCK:
        return JOB_STORE.get(job_id)


def update_job_status(job_id: str, status: JobStatus, result: dict = None, error: str = None, progress: int = None):
    """Update job status and result."""
    with JOB_STORE_LOCK:
        if job_id in JOB_STORE:
            job = JOB_STORE[job_id]
            job.status = status
            if progress is not None:
                job.progress = progress
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = time.time()
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = time.time()
                job.progress = 100
            if result:
                job.result = result
            if error:
                job.error = error


def run_generation_job(job_id: str):
    """Background task to run generation job with WebSocket progress updates."""
    import asyncio

    job = get_job(job_id)
    if not job:
        return

    update_job_status(job_id, JobStatus.RUNNING)
    ws_manager = get_ws_manager()

    async def _send_progress(progress: int, status: str, result: dict = None, error: str = None):
        """Send progress update via WebSocket and update job status."""
        await ws_manager.send_progress(job_id, progress, status, result, error)
        if status == "running":
            update_job_status(job_id, JobStatus.RUNNING, progress=progress)
        elif status == "completed":
            update_job_status(job_id, JobStatus.COMPLETED, result=result, progress=progress)
        elif status == "failed":
            update_job_status(job_id, JobStatus.FAILED, error=error, progress=progress)

    async def _run():
        try:
            await _send_progress(5, "running", None, None)
            text = await _get_text(job.file_id, job.user_id)
            await _send_progress(10, "running", None, None)

            if job.generation_type == "summary":
                await _send_progress(20, "running", None, None)
                with record_generation("summary", success=True):
                    summary = await run_in_threadpool(generate_summary, text)
                await _send_progress(90, "running", None, None)
                result = {"file_id": job.file_id, "summary": summary, "status": "ok"}

            elif job.generation_type == "quiz":
                await _send_progress(20, "running", None, None)
                with record_generation("quiz", success=True):
                    questions = await run_in_threadpool(generate_quiz, text, 10)
                await _send_progress(90, "running", None, None)
                result = {"file_id": job.file_id, "questions": questions, "status": "ok"}

            elif job.generation_type == "audio":
                await _send_progress(20, "running", None, None)
                summary = await run_in_threadpool(generate_summary, text)
                await _send_progress(50, "running", None, None)
                out_path = OUTPUT_DIR / f"{job.file_id}_audio.mp3"
                success = await run_in_threadpool(generate_audio, summary, str(out_path))
                if success:
                    result = {"file_id": job.file_id, "audio_url": f"/media/{job.file_id}_audio.mp3", "status": "ok"}
                else:
                    await _send_progress(100, "failed", None, "TTS generation failed")
                    return

            elif job.generation_type == "video":
                await _send_progress(15, "running", None, None)
                summary = await run_in_threadpool(generate_summary, text)
                await _send_progress(35, "running", None, None)
                out_path = OUTPUT_DIR / f"{job.file_id}_video.mp4"
                await run_in_threadpool(generate_video, summary, str(out_path))
                await _send_progress(85, "running", None, None)
                if out_path.exists() and out_path.stat().st_size > 10_000:
                    result = {"file_id": job.file_id, "video_url": f"/media/{job.file_id}_video.mp4", "status": "ok"}
                else:
                    zip_path = OUTPUT_DIR / f"{job.file_id}_video_slides.zip"
                    if zip_path.exists():
                        result = {"file_id": job.file_id, "video_url": None, "status": "slides_only",
                                 "slides_url": f"/media/{job.file_id}_video_slides.zip",
                                 "message": "MoviePy unavailable, slides provided"}
                    else:
                        await _send_progress(100, "failed", None, "Video generation failed")
                        return

            else:
                raise ValueError(f"Unknown generation type: {job.generation_type}")

            await _send_progress(100, "completed", result, None)

        except Exception as e:
            log.error(f"Generation job {job_id} failed: {e}", exc_info=True)
            await _send_progress(100, "failed", None, str(e))

    # Run async function in new event loop
    asyncio.run(_run())


# ─── APP ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks."""
    # Startup
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"[startup] uploads dir OK  : {UPLOAD_DIR}")
    log.info(f"[startup] outputs dir OK  : {OUTPUT_DIR}")
    log.info("[startup] StaticFiles mount for /media will serve from outputs/")

    # Clean up orphaned files on startup
    _cleanup_orphaned_uploads()

    # Register and optionally preload ML models
    register_default_models()
    # Preload models at startup (reduces first-request latency)
    # Only preload SBERT by default to save memory; Pegasus/T5 loaded on-demand
    try:
        from .models import get_model_manager
        get_model_manager().preload(["sbert"])
        log.info("[startup] Preloaded SBERT model")
    except Exception as e:
        log.warning(f"[startup] Model preload failed (will load on-demand): {e}")

    yield

    # Shutdown: unload models to free memory
    from .models import get_model_manager
    unloaded = get_model_manager().unload_all()
    log.info(f"[shutdown] Unloaded {unloaded} models")
    log.info("[shutdown] Application shutting down")

app = FastAPI(
    title="ScholarAI API",
    description="""
**AI-powered Study Assistant — ML Pipeline Backend**

Transform PDF/PPTX lecture notes into complete study packages:

| Output | Technology |
|--------|------------|
| **AI Summary** | BGE Embeddings → Autoencoder → Google Pegasus |
| **10-Question MCQ Quiz** | T5 Question Generation (valhalla/t5-base-qg-hl) |
| **Audio Narration** | gTTS (Google Text-to-Speech) |
| **Explainer Video** | MoviePy + PIL (content-driven duration) |

## Authentication
Currently uses Firebase Authentication on the frontend. Backend endpoints are public but rate-limited.

## Rate Limits
- **Default**: 100 requests/minute per IP
- **Upload**: 10 requests/minute
- **Cleanup**: 30 requests/minute

## File Handling
- Uploaded files are **AES-256 encrypted** (Fernet) before being written to disk
- Files are assigned a **UUID** — original filenames are never stored on disk
- Files are **auto-deleted after 1 hour** via a background cleanup thread
- Maximum upload size: **50 MB**

## Media URLs
All generated media (audio, video, slides) are served from `/media/{filename}`.
Use the returned URLs directly in `<audio>`, `<video>`, or `<a download>` tags.
""",
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "ScholarAI Project",
        "url": "https://github.com/kaushikasemwal/ScholarAI_Project",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {"name": Tags.UPLOAD, "description": "File upload and encryption"},
        {"name": Tags.GENERATION, "description": "AI content generation (summary, quiz, audio, video)"},
        {"name": Tags.MEDIA, "description": "Media file serving and debugging"},
        {"name": Tags.UTILITY, "description": "Health checks and maintenance"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ─── OBSERVABILITY MIDDLEWARE ────────────────────────────────────
app.add_middleware(RequestIDMiddleware)

# ─── HEALTH & METRICS ENDPOINTS ──────────────────────────────────

@app.get(
    "/health/live",
    response_model=HealthResponse,
    tags=[Tags.UTILITY],
    summary="Liveness probe",
    description="Kubernetes liveness probe - returns 200 if process is alive.",
    include_in_schema=False,
)
async def liveness() -> HealthResponse:
    checker = get_health_checker()
    report = checker.liveness()
    status_code = 200 if report.status == HealthStatus.HEALTHY else 503
    return HealthResponse(
        message=f"Liveness: {report.status.value}",
    )

@app.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=[Tags.UTILITY],
    summary="Readiness probe",
    description="Kubernetes readiness probe - returns 200 if service can handle requests.",
    include_in_schema=False,
)
async def readiness() -> HealthResponse:
    checker = get_health_checker()
    report = checker.readiness()
    status_code = 200 if report.status == HealthStatus.HEALTHY else 503
    return HealthResponse(
        message=f"Readiness: {report.status.value}",
    )

@app.get(
    "/health/startup",
    response_model=HealthResponse,
    tags=[Tags.UTILITY],
    summary="Startup probe",
    description="Kubernetes startup probe - returns 200 when initialization is complete.",
    include_in_schema=False,
)
async def startup_probe() -> HealthResponse:
    checker = get_health_checker()
    report = checker.startup()
    status_code = 200 if report.status == HealthStatus.HEALTHY else 503
    return HealthResponse(
        message=f"Startup: {report.status.value}",
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[Tags.UTILITY],
    summary="Full health check",
    description="Comprehensive health report with all checks.",
)
async def health() -> HealthResponse:
    checker = get_health_checker()
    report = checker.readiness()
    details = {c.name: {"status": c.status.value, "message": c.message} for c in report.checks}
    return HealthResponse(
        message=f"Health: {report.status.value}",
    )

@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    tags=[Tags.UTILITY],
    summary="Prometheus metrics",
    description="Prometheus-formatted metrics exposition endpoint.",
    include_in_schema=False,
)
async def metrics() -> PlainTextResponse:
    collector = get_metrics_collector()
    output = collector.generate_prometheus_output()
    return PlainTextResponse(output, media_type="text/plain; version=0.0.4")

# ─── STARTUP CLEANUP ─────────────────────────────────────────────
def _cleanup_orphaned_uploads():
    """Clean up orphaned encrypted files on startup."""
    try:
        count = 0
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.suffix == ".enc":
                try:
                    f.unlink()
                    count += 1
                except Exception as e:
                    log.warning(f"Could not delete orphaned file {f}: {e}")
        if count:
            log.info(f"[startup] Cleaned {count} orphaned encrypted files")
    except Exception as e:
        log.warning(f"Startup cleanup failed: {e}")


# ─── BACKGROUND AUTO-CLEANUP ─────────────────────────────────────
def auto_cleanup_daemon():
    """Background thread that periodically deletes old uploaded files."""
    while True:
        time.sleep(300)
        now = time.time()
        with FILE_STORE_LOCK:
            expired = [
                fid for fid, meta in list(FILE_STORE.items())
                if now - meta["created_at"] > AUTO_DELETE_SECONDS
            ]
            for fid in expired:
                try:
                    p = FILE_STORE[fid].get("path")
                    if p and Path(p).exists():
                        Path(p).unlink()
                        log.info(f"Auto-deleted file {fid}")
                    del FILE_STORE[fid]
                except Exception as e:
                    log.warning(f"Cleanup error for {fid}: {e}")

threading.Thread(target=auto_cleanup_daemon, daemon=True).start()

# ─── ENDPOINTS ───────────────────────────────────────────────────

@app.get(
    "/",
    response_model=HealthResponse,
    tags=[Tags.UTILITY],
    summary="Health check",
    description="Returns a simple status message confirming the API is running.",
    response_description="API status message",
)
async def root() -> HealthResponse:
    return HealthResponse(message="ScholarAI API is running. See /docs for endpoints.")


@app.get(
    "/media-check/{filename}",
    response_model=MediaCheckResponse,
    tags=[Tags.MEDIA],
    summary="Verify media file",
    description="Debug endpoint to verify a generated media file exists on disk and return its size. Use to diagnose playback issues.",
    response_description="File existence and metadata",
    responses={
        200: {"description": "File found", "model": MediaCheckResponse},
        404: {"description": "File not found", "model": MediaCheckResponse},
    },
)
async def media_check(filename: str) -> MediaCheckResponse:
    p = OUTPUT_DIR / filename
    if p.exists():
        size = p.stat().st_size
        is_likely_real = size > 500
        return MediaCheckResponse(
            exists=True,
            size_bytes=size,
            is_likely_real_audio=is_likely_real,
            path=str(p),
        )
    return MediaCheckResponse(exists=False, filename=filename)


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=[Tags.UPLOAD],
    summary="Upload PDF/PPTX",
    description="""
Upload a PDF or PowerPoint file for processing.

**Security:**
- File is **AES-256 encrypted** (Fernet) before being written to disk
- Original filename is never stored — only a UUID is used
- Files auto-delete after **1 hour**

**Limits:**
- Max file size: **50 MB**
- Allowed types: `application/pdf`, `application/vnd.openxmlformats-officedocument.presentationml.presentation`, `application/vnd.ms-powerpoint`
- Rate limited: **10 requests/minute**

**Returns:** A `file_id` to use with generation endpoints.
""",
    response_description="Upload confirmation with file_id",
    responses={
        200: {"description": "Upload successful", "model": UploadResponse},
        400: {"description": "Invalid file type", "model": ErrorResponse},
        413: {"description": "File too large", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
    },
)
async def upload_file(request: Request, file: UploadFile = File(...), current_user: CurrentUser = Depends(require_auth), _rate_limit: None = Depends(rate_limit_upload)) -> UploadResponse:
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint"
    }
    allowed_exts = {".pdf", ".pptx", ".ppt"}

    ext = Path(file.filename).suffix.lower()
    if file.content_type not in allowed_types and ext not in allowed_exts:
        raise HTTPException(400, "Only PDF and PPTX files are accepted.")

    # Check file size before reading entirely into memory
    if file.size and file.size > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)} MB")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)} MB")

    # Validate file content (magic bytes)
    if ext == ".pdf" and not raw_bytes.startswith(b"%PDF"):
        raise HTTPException(400, "Invalid PDF file: missing %PDF header")
    elif ext in (".pptx", ".ppt") and not raw_bytes.startswith(b"PK\x03\x04"):
        raise HTTPException(400, "Invalid PPTX/PPT file: missing ZIP/PK header")

    file_id   = str(uuid.uuid4())
    encrypted = encrypt_file(raw_bytes)
    storage_key = f"uploads/{file_id}{ext}.enc"

    # Store in storage backend
    storage = get_storage_backend()
    await storage.put_object(storage_key, encrypted, content_type="application/octet-stream")

    with FILE_STORE_LOCK:
        FILE_STORE[file_id] = {
            "storage_key": storage_key,
            "ext":         ext,
            "filename":    file.filename,
            "created_at":  time.time(),
            "text":        None,
            "user_id":     current_user.uid,
        }

    log.info(f"Uploaded & encrypted: {file.filename} → {file_id} (user: {current_user.uid})")
    record_upload(ext.lstrip("."), len(raw_bytes), success=True)
    return UploadResponse(file_id=file_id, filename=file.filename, status="uploaded")


async def _get_text(file_id: str, user_id: str) -> str:
    """Decrypt file and extract text (cached after first call). Verifies user ownership."""
    with FILE_STORE_LOCK:
        if file_id not in FILE_STORE:
            raise HTTPException(404, "File not found. Please re-upload.")
        meta = FILE_STORE[file_id]

    # Verify user owns this file
    if meta.get("user_id") != user_id:
        raise HTTPException(403, "Access denied: file belongs to another user.")

    if meta["text"]:
        return meta["text"]

    # Retrieve from storage backend
    storage = get_storage_backend()
    storage_key = meta["storage_key"]
    try:
        encrypted_data = await storage.get_object(storage_key)
    except KeyError:
        raise HTTPException(404, "Encrypted file missing from storage.")

    raw_bytes = decrypt_file(encrypted_data)
    ext = meta["ext"]

    from .utils import extract_text_pdf, extract_text_pptx

    if ext == ".pdf":
        text = extract_text_pdf(raw_bytes)
    elif ext in (".pptx", ".ppt"):
        text = extract_text_pptx(raw_bytes)
    else:
        raise HTTPException(
            400,
            f"Unsupported file extension: {ext}. Please upload a PDF or PPTX."
        )

    with FILE_STORE_LOCK:
        if file_id in FILE_STORE:
            FILE_STORE[file_id]["text"] = text
    log.info(f"Extracted {len(text)} chars from {file_id}")
    return text


@app.post(
    "/generate/summary",
    response_model=SummaryResponse,
    tags=[Tags.GENERATION],
    summary="Generate AI summary",
    description="""
Generate an abstractive summary using the full ML pipeline:

1. **Text Extraction** → PyMuPDF / python-pptx
2. **Sentence Tokenization** → NLTK
3. **Embeddings** → BAAI/bge-small-en-v1.5 (384-dim)
4. **Compression** → Semantic Autoencoder (384 → 128-dim latent)
5. **Key Sentence Selection** → Centroid-based cosine similarity
6. **Abstractive Summarization** → Google Pegasus (pegasus-xsum)

**Pipeline Details:**
- Uses BGE embeddings with query prefix for retrieval tasks
- Autoencoder denoises embeddings, preserving core semantic signals
- Pegasus chosen over BART for its gap-sentence generation pretraining objective
- Falls back to extractive TF-IDF summarization if ML libraries unavailable
""",
    response_description="Generated summary with metadata",
    responses={
        200: {"description": "Summary generated", "model": SummaryResponse},
        404: {"description": "File not found", "model": ErrorResponse},
        500: {"description": "Pipeline error (falls back to extractive)", "model": SummaryResponse},
    },
)
async def api_summary(req: GenerateRequest, current_user: CurrentUser = Depends(require_auth)) -> SummaryResponse:
    text = await _get_text(req.file_id, current_user.uid)
    log.info(f"Generating summary for {req.file_id} (user: {current_user.uid})")
    with record_generation("summary", success=True):
        summary, metadata = generate_summary(text)
    return SummaryResponse(
        file_id=req.file_id,
        summary=summary,
        status=GenerationStatus.OK,
        fallback_used=metadata.get("fallback_used", False),
        fallback_reason=metadata.get("fallback_reason"),
        model_used=metadata.get("model_used", "pegasus")
    )


@app.post(
    "/generate/quiz",
    response_model=QuizResponse,
    tags=[Tags.GENERATION],
    summary="Generate MCQ quiz",
    description="""
Generate a 10-question multiple-choice quiz using T5 question generation:

1. **Chunking** → Split text into ~400-char overlapping chunks
2. **Question Generation** → valhalla/t5-base-qg-hl with `<hl>` answer highlighting
3. **Distractor Generation** → TF-IDF keyword extraction from source chunk
4. **Diversity Filtering** → BGE embeddings + Max-Min diversity selection
5. **Fallback** → Rule-based pattern matching if T5 unavailable

**Question Format:**
- 4 options (A, B, C, D)
- 1 correct answer with index (0-3)
- Reasoning/explanation grounded in source text
- Minimum 10 questions guaranteed (padded with generic if needed)
""",
    response_description="10 MCQ questions with answers and reasoning",
    responses={
        200: {"description": "Quiz generated", "model": QuizResponse},
        404: {"description": "File not found", "model": ErrorResponse},
        500: {"description": "Generation error", "model": ErrorResponse},
    },
)
async def api_quiz(req: GenerateRequest, current_user: CurrentUser = Depends(require_auth)) -> QuizResponse:
    text = await _get_text(req.file_id, current_user.uid)
    log.info(f"Generating quiz for {req.file_id} (user: {current_user.uid})")
    with record_generation("quiz", success=True):
        questions = generate_quiz(text, n=10)
    return QuizResponse(file_id=req.file_id, questions=questions, status=GenerationStatus.OK)


@app.post(
    "/generate/audio",
    response_model=AudioResponse,
    tags=[Tags.GENERATION],
    summary="Generate audio narration",
    description="""
Convert the AI-generated summary to speech using gTTS (primary) or pyttsx3 (fallback).

**TTS Engines (tried in order):**
1. **gTTS** — Google Text-to-Speech (requires internet, high quality)
2. **pyttsx3** — Offline system TTS (Windows only, SAPI5)

**Technical Details:**
- Text split into 500-char sentence-aware chunks
- gTTS MP3 chunks concatenated at binary level (no ffmpeg needed)
- pyttsx3 saves WAV, copied to MP3 path for consistent serving
- Minimum 500 bytes required for valid audio file
""",
    response_description="Audio URL or failure details",
    responses={
        200: {
            "description": "Audio generated or failed",
            "model": AudioResponse,
            "content": {
                "application/json": {
                    "examples": {
                        "success": {
                            "summary": "Audio ready",
                            "value": {"file_id": "...", "audio_url": "/media/..._audio.mp3", "status": "ok"}
                        },
                        "failed": {
                            "summary": "TTS unavailable",
                            "value": {"file_id": "...", "audio_url": None, "status": "failed", "message": "..."}
                        }
                    }
                }
            }
        },
        404: {"description": "File not found", "model": ErrorResponse},
    },
)
async def api_audio(req: GenerateRequest, current_user: CurrentUser = Depends(require_auth)) -> AudioResponse:
    text = await _get_text(req.file_id, current_user.uid)
    log.info(f"Generating audio for {req.file_id} (user: {current_user.uid})")

    summary, _ = generate_summary(text)
    out_path = OUTPUT_DIR / f"{req.file_id}_audio.mp3"

    success = False
    with record_generation("audio", success=True):
        success = generate_audio(summary, str(out_path))

    if success:
        size = out_path.stat().st_size
        log.info(f"Audio ready: {out_path} ({size} bytes)")
        return AudioResponse(
            file_id=req.file_id,
            audio_url=f"/media/{req.file_id}_audio.mp3",
            status=GenerationStatus.OK,
        )

    log.warning(f"Audio generation failed for {req.file_id}")
    return AudioResponse(
        file_id=req.file_id,
        audio_url=None,
        status=GenerationStatus.FAILED,
        message=(
            "No TTS engine produced a valid audio file. "
            "Make sure gTTS is installed (pip install gtts) "
            "and that you have an active internet connection, "
            "or install pyttsx3 for offline TTS."
        ),
    )


@app.post(
    "/generate/video",
    response_model=VideoResponse,
    tags=[Tags.GENERATION],
    summary="Generate explainer video",
    description="""
Generate a slide-style explainer video using MoviePy + PIL.

**Video Generation:**
- Duration derived from content (reading speed: 150 wpm)
- Each slide duration = reading time + 1s padding (clamped 5-20s)
- Slides rendered with PIL (grid background, accent bar, branding)
- TTS narration embedded if audio generation succeeds
- Exported as H.264 MP4 (CRF 35, ultrafast preset)

**Fallbacks:**
1. **Slides ZIP** — If MoviePy unavailable, returns PNG slides in ZIP
2. **Placeholder** — If PIL unavailable, returns text placeholder

**Rate Limited:** No explicit limit (depends on generation time)
""",
    response_description="Video URL, slides ZIP, or failure details",
    responses={
        200: {
            "description": "Video generated, slides only, or failed",
            "model": VideoResponse,
        },
        404: {"description": "File not found", "model": ErrorResponse},
    },
)
async def api_video(req: GenerateRequest, current_user: CurrentUser = Depends(require_auth)) -> VideoResponse:
    text = await _get_text(req.file_id, current_user.uid)
    log.info(f"Generating video for {req.file_id} (user: {current_user.uid})")

    summary, _ = generate_summary(text)
    out_path = OUTPUT_DIR / f"{req.file_id}_video.mp4"

    with record_generation("video", success=True):
        video_path, metadata = generate_video(summary, str(out_path))

    status_map = {
        "video_ok": GenerationStatus.OK,
        "slides_only": GenerationStatus.SLIDES_ONLY,
        "placeholder": GenerationStatus.FAILED,
    }

    video_status = status_map.get(metadata.get("status", "placeholder"), GenerationStatus.FAILED)

    if video_status == GenerationStatus.OK:
        size = out_path.stat().st_size
        log.info(f"Video ready: {out_path} ({size} bytes)")
        return VideoResponse(
            file_id=req.file_id,
            video_url=f"/media/{req.file_id}_video.mp4",
            status=video_status,
            message=metadata.get("message"),
        )

    zip_path = OUTPUT_DIR / f"{req.file_id}_video_slides.zip"
    if zip_path.exists() or metadata.get("slides_url"):
        log.info(f"Video not generated; slides ZIP available: {zip_path}")
        return VideoResponse(
            file_id=req.file_id,
            video_url=None,
            status=GenerationStatus.SLIDES_ONLY,
            slides_url=f"/media/{req.file_id}_video_slides.zip",
            message=metadata.get("message", "MoviePy/ffmpeg not available. Download slide images instead."),
        )

    log.warning(f"Video generation produced no usable output for {req.file_id}")
    return VideoResponse(
        file_id=req.file_id,
        video_url=None,
        status=GenerationStatus.FAILED,
        message=metadata.get("message", "Video generation failed."),
    )


@app.delete(
    "/cleanup/{file_id}",
    response_model=CleanupResponse,
    tags=[Tags.UTILITY],
    summary="Delete file and outputs",
    description="""
Manually delete an uploaded file and all its generated outputs.

**Deletes:**
- Encrypted upload file (`uploads/{file_id}.enc`)
- Audio (`outputs/{file_id}_audio.mp3`)
- Video (`outputs/{file_id}_video.mp4`)
- Slides ZIP (`outputs/{file_id}_video_slides.zip`)

**Rate Limited:** 30 requests/minute
""",
    response_description="Deletion confirmation",
    responses={
        200: {"description": "Deleted", "model": CleanupResponse},
        404: {"description": "File not found", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
    },
)
async def manual_cleanup(request: Request, file_id: str, current_user: CurrentUser = Depends(require_auth), _rate_limit: None = Depends(rate_limit_cleanup)) -> CleanupResponse:
    with FILE_STORE_LOCK:
        if file_id not in FILE_STORE:
            raise HTTPException(404, "File ID not found.")
        meta = FILE_STORE[file_id]

        # Verify user owns this file
        if meta.get("user_id") != current_user.uid:
            raise HTTPException(403, "Access denied: file belongs to another user.")

        FILE_STORE.pop(file_id)

    # Delete generated output files
    for suffix in ["_audio.mp3", "_video.mp4", "_video_slides.zip"]:
        out = OUTPUT_DIR / f"{file_id}{suffix}"
        if out.exists():
            out.unlink()

    # Also delete from storage backend
    storage = get_storage_backend()
    try:
        await storage.delete_object(meta["storage_key"])
    except Exception as e:
        log.warning(f"Failed to delete from storage: {e}")

    return CleanupResponse(status="deleted", file_id=file_id)


# ─── ASYNC GENERATION ENDPOINTS ──────────────────────────────────

@app.post(
    "/jobs",
    response_model=JobCreateResponse,
    tags=[Tags.ASYNC_GENERATION],
    summary="Create async generation job",
    description="""
Start an asynchronous generation job for summary, quiz, audio, or video.

Returns a job_id immediately. Poll `/jobs/{job_id}` for status updates.

**Generation Types:**
- `summary` — Abstractive summarization (typically 10-30s)
- `quiz` — 10 MCQ questions via T5 (typically 15-45s)
- `audio` — TTS narration via gTTS (typically 5-20s)
- `video` — Slide-style explainer video (typically 30-120s)

**Rate Limited:** 10 requests/minute per IP
""",
    responses={
        200: {"description": "Job created", "model": JobCreateResponse},
        404: {"description": "File not found", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse},
    },
)
async def create_generation_job(request: Request, job_req: JobCreateRequest, current_user: CurrentUser = Depends(require_auth), _rate_limit: None = Depends(rate_limit_jobs)) -> JobCreateResponse:
    # Verify file exists and user owns it
    with FILE_STORE_LOCK:
        if job_req.file_id not in FILE_STORE:
            raise HTTPException(404, "File not found. Please re-upload.")
        meta = FILE_STORE[job_req.file_id]
        if meta.get("user_id") != current_user.uid:
            raise HTTPException(403, "Access denied: file belongs to another user.")

    job = create_job(job_req.file_id, job_req.generation_type, current_user.uid)

    # Submit to thread pool for background execution
    _generation_executor.submit(run_generation_job, job.job_id)

    return JobCreateResponse(
        job_id=job.job_id,
        status=JobStatus.PENDING,
        progress=0,
        message="Job created. Poll /jobs/{job_id} for status."
    )


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=[Tags.ASYNC_GENERATION],
    summary="Get job status",
    description="""
Poll this endpoint to check the status of an async generation job.

**Progress Values:**
- `0` — Pending/queued
- `1-99` — Running (approximate)
- `100` — Completed or failed

**Typical Flow:**
1. POST `/jobs` → get `job_id`
2. Poll GET `/jobs/{job_id}` every 2-5 seconds
3. When `status` is `completed`, use `result` field
4. When `status` is `failed`, check `error` field
""",
    responses={
        200: {"description": "Job status", "model": JobStatusResponse},
        404: {"description": "Job not found", "model": ErrorResponse},
    },
)
async def get_job_status(job_id: str, current_user: CurrentUser = Depends(require_auth)) -> JobStatusResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Verify user owns this job
    if job.user_id != current_user.uid:
        raise HTTPException(403, "Access denied: job belongs to another user.")

    return JobStatusResponse(
        job_id=job.job_id,
        file_id=job.file_id,
        generation_type=job.generation_type,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
    )


@app.get(
    "/jobs",
    response_model=list[JobStatusResponse],
    tags=[Tags.ASYNC_GENERATION],
    summary="List user's jobs",
    description="List all generation jobs for the current user.",
)
async def list_jobs(current_user: CurrentUser = Depends(require_auth)) -> list[JobStatusResponse]:
    with JOB_STORE_LOCK:
        jobs = [j for j in JOB_STORE.values() if j.user_id == current_user.uid]

    result = []
    for job in jobs:
        result.append(JobStatusResponse(
            job_id=job.job_id,
            file_id=job.file_id,
            generation_type=job.generation_type,
            status=job.status,
            progress=job.progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result=job.result,
            error=job.error,
        ))

    # Sort by creation time (newest first)
    result.sort(key=lambda j: j.created_at, reverse=True)
    return result


# ─── WEBSOCKET ENDPOINT ───────────────────────────────────────────

@app.websocket(
    "/ws/jobs/{job_id}",
    name="job_progress_ws",
)
async def job_progress_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates.
    
    Connect to this endpoint after creating a job via POST /jobs.
    The WebSocket will send progress updates until the job completes or fails.
    
    **Authentication:** Pass Firebase ID token as query parameter `token` or `Authorization` header.
    
    **Message Format:**
    ```json
    {
        "job_id": "uuid",
        "progress": 50,
        "status": "running",
        "result": {...},
        "error": null
    }
    ```
    
    **Status Values:**
    - `pending` — Job queued
    - `running` — Job in progress
    - `completed` — Job finished successfully
    - `failed` — Job failed
    """
    # Extract token from query params or headers
    token = websocket.query_params.get("token")
    if not token:
        # Check Authorization header
        auth_header = websocket.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    # Verify token
    from .auth import get_firebase_auth
    auth = get_firebase_auth()
    if auth is None:
        await websocket.close(code=4001, reason="Authentication not configured")
        return

    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token.get("uid")
        if not user_id:
            await websocket.close(code=4001, reason="Invalid token")
            return
    except Exception:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # Verify job exists and user owns it
    job = get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    if job.user_id != user_id:
        await websocket.close(code=4003, reason="Access denied: job belongs to another user")
        return

    ws_manager = get_ws_manager()
    await ws_manager.connect(websocket, job_id)

    try:
        # Send initial status
        await ws_manager.send_progress(
            job_id,
            progress=job.progress,
            status=job.status.value,
            result=job.result,
            error=job.error,
        )

        # Keep connection alive, listen for client messages (ping/pong)
        while True:
            try:
                # Wait for client message (e.g., ping) with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back or handle client commands
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "keepalive"}))
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        ws_manager.disconnect(websocket)


# ─── STATIC FILES (must be LAST) ─────────────────────────────────
# Serves /media/xxx_audio.mp3 and /media/xxx_video.mp4
# IMPORTANT: mount AFTER all route definitions so /media-check route
# is registered first and is not shadowed by the static mount.
app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR)), name="media")
