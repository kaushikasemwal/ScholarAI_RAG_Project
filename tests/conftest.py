"""
conftest.py — Shared Test Configuration
========================================
Common fixtures and configuration for all tests.
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure test environment
os.environ["LOG_JSON"] = "false"
os.environ["ALLOWED_ORIGINS"] = "*"


@pytest.fixture(scope="session")
def sample_pdf():
    """Minimal valid PDF for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n193\n%%EOF"


@pytest.fixture
def uploaded_file_id(client, sample_pdf):
    """Upload a test PDF and return file_id."""
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", sample_pdf, "application/pdf")}
    )
    assert response.status_code == 200
    return response.json()["file_id"]