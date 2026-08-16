"""
schemas.py — Pydantic Models for Request/Response Validation & OpenAPI Documentation
======================================================================================
Centralized type-safe schemas for all API endpoints.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ─── ENUMS ────────────────────────────────────────────────────────

class GenerationStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SLIDES_ONLY = "slides_only"


class FileType(str, Enum):
    PDF = "pdf"
    PPTX = "pptx"
    PPT = "ppt"


class QuizOption(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# ─── REQUEST MODELS ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request to generate AI output from an uploaded file."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"file_id": "550e8400-e29b-41d4-a716-446655440000"}
        }
    )
    file_id: str = Field(
        ...,
        description="Unique identifier returned from /upload endpoint",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )


class UploadResponse(BaseModel):
    """Response after successful file upload."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "lecture_notes.pdf",
                "status": "uploaded"
            }
        }
    )
    file_id: str = Field(..., description="Unique identifier for this upload")
    filename: str = Field(..., description="Original filename")
    status: Literal["uploaded"] = Field(..., description="Upload status")


# ─── RESPONSE MODELS ───────────────────────────────────────────────

class QuizQuestion(BaseModel):
    """Single MCQ question with options and metadata."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the primary purpose of a convolutional neural network?",
                "options": [
                    "Image classification and feature extraction",
                    "Text generation",
                    "Time series forecasting",
                    "Data compression"
                ],
                "correct": 0,
                "answer": "Image classification and feature extraction",
                "reasoning": "CNNs are designed for spatial data like images, using convolutional layers to extract hierarchical features."
            }
        }
    )
    question: str = Field(..., description="The question text", min_length=5)
    options: List[str] = Field(
        ..., 
        description="Exactly 4 answer options (A, B, C, D)",
        min_length=4,
        max_length=4
    )
    correct: int = Field(
        ..., 
        description="Index of correct option (0-3)",
        ge=0,
        le=3
    )
    answer: str = Field(..., description="Text of the correct answer")
    reasoning: str = Field(..., description="Explanation for why the answer is correct")


class SummaryResponse(BaseModel):
    """Response from summary generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "summary": "Machine learning is a subset of AI that enables systems to learn from data...",
                "status": "ok",
                "fallback_used": False,
                "fallback_reason": None,
                "model_used": "pegasus"
            }
        }
    )
    file_id: str = Field(..., description="Source file identifier")
    summary: str = Field(..., description="Generated abstractive summary", min_length=1)
    status: GenerationStatus = Field(..., description="Generation status")
    fallback_used: bool = Field(default=False, description="Whether extractive fallback was used")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback if used")
    model_used: str = Field(default="pegasus", description="Model used for generation")


class QuizResponse(BaseModel):
    """Response from quiz generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_id": "550e8400-e29b-41d4-a716-446655440000",
                "questions": [
                    {
                        "question": "What is supervised learning?",
                        "options": [
                            "Learning from labeled data",
                            "Learning without labels",
                            "Reinforcement learning",
                            "Unsupervised clustering"
                        ],
                        "correct": 0,
                        "answer": "Learning from labeled data",
                        "reasoning": "Supervised learning requires labeled training examples to learn a mapping from inputs to outputs."
                    }
                ],
                "status": "ok"
            }
        }
    )
    file_id: str = Field(..., description="Source file identifier")
    questions: List[QuizQuestion] = Field(..., description="List of MCQ questions", min_length=1)
    status: GenerationStatus = Field(..., description="Generation status")


class AudioResponse(BaseModel):
    """Response from audio generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": {
                "success": {
                    "summary": "Audio generated successfully",
                    "value": {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "audio_url": "/media/550e8400-e29b-41d4-a716-446655440000_audio.mp3",
                        "status": "ok"
                    }
                },
                "failed": {
                    "summary": "Audio generation failed",
                    "value": {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "audio_url": None,
                        "status": "failed",
                        "message": "No TTS engine produced a valid audio file. Check internet connection or install pyttsx3."
                    }
                }
            }
        }
    )
    file_id: str = Field(..., description="Source file identifier")
    audio_url: Optional[str] = Field(None, description="Playable audio URL (relative to API base)")
    status: GenerationStatus = Field(..., description="Generation status")
    message: Optional[str] = Field(None, description="Error details if failed")


class VideoResponse(BaseModel):
    """Response from video generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": {
                "success": {
                    "summary": "Video generated successfully",
                    "value": {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_url": "/media/550e8400-e29b-41d4-a716-446655440000_video.mp4",
                        "status": "ok"
                    }
                },
                "slides_only": {
                    "summary": "MoviePy unavailable, slides provided",
                    "value": {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_url": None,
                        "status": "slides_only",
                        "slides_url": "/media/550e8400-e29b-41d4-a716-446655440000_video_slides.zip",
                        "message": "MoviePy/ffmpeg not available. Download slide images instead."
                    }
                },
                "failed": {
                    "summary": "Video generation failed",
                    "value": {
                        "file_id": "550e8400-e29b-41d4-a716-446655440000",
                        "video_url": None,
                        "status": "failed",
                        "message": "Video generation failed. Install MoviePy and ffmpeg."
                    }
                }
            }
        }
    )
    file_id: str = Field(..., description="Source file identifier")
    video_url: Optional[str] = Field(None, description="Playable video URL (relative to API base)")
    status: GenerationStatus = Field(..., description="Generation status")
    slides_url: Optional[str] = Field(None, description="Slide images ZIP if video unavailable")
    message: Optional[str] = Field(None, description="Error details or fallback info")


class MediaCheckResponse(BaseModel):
    """Debug response for media file verification."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "exists": True,
                "size_bytes": 1234567,
                "is_likely_real_audio": True,
                "path": "/app/outputs/abc123_audio.mp3"
            }
        }
    )
    exists: bool = Field(..., description="Whether file exists on disk")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    is_likely_real_audio: Optional[bool] = Field(None, description="Heuristic: file > 500 bytes")
    path: Optional[str] = Field(None, description="Absolute server path")
    filename: Optional[str] = Field(None, description="Requested filename (if not found)")


class CleanupResponse(BaseModel):
    """Response from manual cleanup."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"status": "deleted", "file_id": "550e8400-e29b-41d4-a716-446655440000"}
        }
    )
    status: Literal["deleted"] = Field(..., description="Cleanup status")
    file_id: str = Field(..., description="Deleted file identifier")


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "ScholarAI API is running. See /docs for endpoints."}
        }
    )
    message: str = Field(..., description="Status message")


# ─── ERROR MODELS ──────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error response format."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "File not found. Please re-upload."
            }
        }
    )
    detail: str = Field(..., description="Human-readable error message")


class ValidationErrorResponse(BaseModel):
    """Validation error with field details."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "loc": ["body", "file_id"],
                        "msg": "Field required",
                        "type": "value_error.missing"
                    }
                ]
            }
        }
    )
    detail: List[dict] = Field(..., description="Validation error details")


# ─── UNION RESPONSES (for endpoints with multiple success shapes) ──

AudioResult = Union[
    AudioResponse,  # status=ok or status=failed
]

VideoResult = Union[
    VideoResponse,  # status=ok, status=slides_only, or status=failed
]


# ─── ASYNC JOB MODELS ──────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreateRequest(BaseModel):
    """Request to start an async generation job."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"file_id": "550e8400-e29b-41d4-a716-446655440000", "generation_type": "summary"}
        }
    )
    file_id: str = Field(..., description="File ID from upload")
    generation_type: Literal["summary", "quiz", "audio", "video"] = Field(
        ..., description="Type of generation to perform"
    )


class JobCreateResponse(BaseModel):
    """Response after creating an async job."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "progress": 0,
                "message": "Job created. Poll /jobs/{job_id} for status."
            }
        }
    )
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Initial job status")
    progress: int = Field(default=0, description="Initial progress percentage (0-100)", ge=0, le=100)
    message: str = Field(..., description="Human-readable message")


class JobStatusResponse(BaseModel):
    """Response for job status polling."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": {
                "pending": {
                    "summary": "Job queued",
                    "value": {"job_id": "...", "status": "pending", "progress": 0}
                },
                "running": {
                    "summary": "Job in progress",
                    "value": {"job_id": "...", "status": "running", "progress": 50}
                },
                "completed": {
                    "summary": "Job completed",
                    "value": {"job_id": "...", "status": "completed", "result": {"file_id": "...", "summary": "..."}}
                },
                "failed": {
                    "summary": "Job failed",
                    "value": {"job_id": "...", "status": "failed", "error": "Model loading failed"}
                }
            }
        }
    )
    job_id: str = Field(..., description="Job identifier")
    file_id: str = Field(..., description="Source file identifier")
    generation_type: str = Field(..., description="Type of generation")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(default=0, description="Progress percentage (0-100)", ge=0, le=100)
    created_at: float = Field(..., description="Job creation timestamp")
    started_at: Optional[float] = Field(None, description="Job start timestamp")
    completed_at: Optional[float] = Field(None, description="Job completion timestamp")
    result: Optional[dict] = Field(None, description="Generation result (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")


# ─── TAGS FOR OPENAPI GROUPING ────────────────────────────────────

class Tags:
    UPLOAD = "Upload"
    GENERATION = "Generation"
    ASYNC_GENERATION = "Async Generation"
    MEDIA = "Media"
    UTILITY = "Utility"