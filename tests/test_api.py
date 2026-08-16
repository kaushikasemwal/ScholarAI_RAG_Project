"""
test_api.py — API Integration Tests
===================================
Tests for all API endpoints including observability features.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


class TestHealthEndpoints:
    """Health check endpoint tests."""

    def test_liveness(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Liveness" in data["message"]

    def test_readiness(self):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Readiness" in data["message"]

    def test_startup(self):
        response = client.get("/health/startup")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Startup" in data["message"]

    def test_full_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        text = response.text
        assert "# HELP" in text
        assert "# TYPE" in text


class TestRequestIDMiddleware:
    """Request ID middleware tests."""

    def test_request_id_header_returned(self):
        response = client.get("/health/live", headers={"X-Request-ID": "test-123"})
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == "test-123"

    def test_request_id_generated_if_missing(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0


class TestRootEndpoint:
    """Root endpoint tests."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "ScholarAI" in data["message"]


class TestMediaCheckEndpoint:
    """Media check endpoint tests."""

    def test_media_check_not_found(self):
        response = client.get("/media-check/nonexistent.mp3")
        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is False
        assert data["filename"] == "nonexistent.mp3"

    def test_media_check_existing(self):
        # Create a dummy file
        import tempfile
        from pathlib import Path
        from backend.app import OUTPUT_DIR
        
        test_file = OUTPUT_DIR / "test_audio.mp3"
        test_file.write_text("dummy content")
        
        try:
            response = client.get("/media-check/test_audio.mp3")
            assert response.status_code == 200
            data = response.json()
            assert data["exists"] is True
            assert data["size_bytes"] > 0
        finally:
            test_file.unlink(missing_ok=True)


class TestUploadEndpoint:
    """File upload endpoint tests."""

    def test_upload_invalid_type(self):
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"text content", "text/plain")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_upload_pdf(self):
        # Minimal valid PDF
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n193\n%%EOF"
        
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test.pdf"
        assert data["status"] == "uploaded"

    def test_upload_too_large(self):
        # Create content larger than 50MB
        large_content = b"x" * (51 * 1024 * 1024)
        response = client.post(
            "/upload",
            files={"file": ("large.pdf", large_content, "application/pdf")}
        )
        assert response.status_code == 413


class TestGenerationEndpoints:
    """Generation endpoint tests (require uploaded file)."""

    @pytest.fixture
    def uploaded_file_id(self):
        """Upload a test PDF and return file_id."""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n193\n%%EOF"
        
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )
        assert response.status_code == 200
        return response.json()["file_id"]

    def test_generate_summary(self, uploaded_file_id):
        response = client.post(
            "/generate/summary",
            json={"file_id": uploaded_file_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "summary" in data
        assert "status" in data
        assert data["status"] == "ok"
        assert len(data["summary"]) > 0

    def test_generate_quiz(self, uploaded_file_id):
        response = client.post(
            "/generate/quiz",
            json={"file_id": uploaded_file_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "questions" in data
        assert "status" in data
        assert data["status"] == "ok"
        assert isinstance(data["questions"], list)
        assert len(data["questions"]) > 0
        
        # Check question structure
        q = data["questions"][0]
        assert "question" in q
        assert "options" in q
        assert "correct" in q
        assert "answer" in q
        assert "reasoning" in q
        assert len(q["options"]) == 4
        assert 0 <= q["correct"] <= 3

    def test_generate_audio(self, uploaded_file_id):
        response = client.post(
            "/generate/audio",
            json={"file_id": uploaded_file_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "status" in data
        # Audio may fail if no internet/gTTS, but should return proper structure
        assert data["status"] in ["ok", "failed"]
        if data["status"] == "ok":
            assert data["audio_url"] is not None
            assert data["audio_url"].endswith("_audio.mp3")

    def test_generate_video(self, uploaded_file_id):
        response = client.post(
            "/generate/video",
            json={"file_id": uploaded_file_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "status" in data
        # Video may fail if MoviePy not available
        assert data["status"] in ["ok", "slides_only", "failed"]
        if data["status"] == "ok":
            assert data["video_url"] is not None
            assert data["video_url"].endswith("_video.mp4")

    def test_generate_nonexistent_file(self):
        response = client.post(
            "/generate/summary",
            json={"file_id": "nonexistent-file-id"}
        )
        assert response.status_code == 404


class TestCleanupEndpoint:
    """Cleanup endpoint tests."""

    def test_cleanup(self, uploaded_file_id):
        response = client.delete(f"/cleanup/{uploaded_file_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["file_id"] == uploaded_file_id

    def test_cleanup_nonexistent(self):
        response = client.delete("/cleanup/nonexistent-file-id")
        assert response.status_code == 404


# Use the fixture from TestGenerationEndpoints
@pytest.fixture
def uploaded_file_id():
    """Upload a test PDF and return file_id."""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n193\n%%EOF"
    
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", pdf_content, "application/pdf")}
    )
    assert response.status_code == 200
    return response.json()["file_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])