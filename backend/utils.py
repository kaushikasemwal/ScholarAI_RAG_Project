"""
utils.py — Utility Functions
==============================
Provides:
  - AES-256 file encryption / decryption (using Fernet / PyCryptodome)
  - PDF text extraction (PyMuPDF / pdfplumber)
  - PPTX text extraction (python-pptx)
  - Cleanup utilities

Security Notes:
  - AES-256 in CBC mode with random IV per file
  - Key derived from environment variable or auto-generated on startup
  - Auto-delete temporary files after configurable TTL

Course: Advanced Topics in Machine Learning (HTML)
"""

import io
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# ─── ENCRYPTION KEY MANAGEMENT ──────────────────────────────────
# Supports key rotation: maintains current + previous keys for decryption
# File format: [key_version:1 byte][encrypted_data]
# Key files: models/encryption.key (current), models/encryption.key.v{N} (previous)

import base64
import logging

log = logging.getLogger(__name__)

# Key management
_ENCRYPTION_KEYS: dict[int, bytes] = {}  # version -> key
_CURRENT_KEY_VERSION: int = 1
_KEY_DIR = Path(__file__).parent.parent / "models"
_KEY_LOCK = threading.Lock()
_MAX_KEY_VERSIONS = 5  # Keep last 5 key versions for decryption


def _get_key_file(version: int) -> Path:
    """Get path for key file of given version."""
    if version == 1:
        return _KEY_DIR / "encryption.key"
    return _KEY_DIR / f"encryption.key.v{version}"


def _load_all_keys() -> dict[int, bytes]:
    """Load all available key versions from disk."""
    keys = {}
    # Check current key
    current_file = _get_key_file(1)
    if current_file.exists():
        try:
            key_data = current_file.read_bytes()
            if len(key_data) == 32:
                keys[1] = key_data
        except Exception as e:
            log.warning(f"Failed to load current key: {e}")

    # Check previous versions
    for v in range(2, _MAX_KEY_VERSIONS + 1):
        key_file = _get_key_file(v)
        if key_file.exists():
            try:
                key_data = key_file.read_bytes()
                if len(key_data) == 32:
                    keys[v] = key_data
            except Exception as e:
                log.warning(f"Failed to load key v{v}: {e}")

    return keys


def _get_key(version: int | None = None) -> bytes:
    """
    Get encryption key for given version (or current version).
    Priority:
    1. AES_KEY environment variable (base64 32-byte key) - version 1 only
    2. Persisted key files
    3. Generate new key and persist it (version 1 only)
    """
    global _ENCRYPTION_KEYS, _CURRENT_KEY_VERSION

    target_version = version or _CURRENT_KEY_VERSION

    with _KEY_LOCK:
        # Load keys if not already loaded
        if not _ENCRYPTION_KEYS:
            _ENCRYPTION_KEYS = _load_all_keys()
            if _ENCRYPTION_KEYS:
                _CURRENT_KEY_VERSION = max(_ENCRYPTION_KEYS.keys())
                log.info(f"Loaded {len(_ENCRYPTION_KEYS)} encryption key versions (current: v{_CURRENT_KEY_VERSION})")

        # Return requested version if available
        if target_version in _ENCRYPTION_KEYS:
            return _ENCRYPTION_KEYS[target_version]

        # For current version, try env var or generate new
        if target_version == _CURRENT_KEY_VERSION or target_version == 1:
            env_key = os.environ.get("AES_KEY")
            if env_key:
                key = base64.b64decode(env_key)
                if len(key) == 32:
                    _ENCRYPTION_KEYS[1] = key
                    _persist_key(1, key)
                    log.info("Loaded AES-256 key from environment variable.")
                    return key

            # Generate new key
            key = os.urandom(32)
            _ENCRYPTION_KEYS[1] = key
            _CURRENT_KEY_VERSION = 1
            _persist_key(1, key)
            log.info("Generated new AES-256 key (v1).")
            return key

        raise ValueError(f"Encryption key version {target_version} not available")


def _persist_key(version: int, key: bytes):
    """Persist key to disk with proper permissions."""
    try:
        _KEY_DIR.mkdir(parents=True, exist_ok=True)
        key_file = _get_key_file(version)
        key_file.write_bytes(key)
        key_file.chmod(0o600)
    except Exception as e:
        log.warning(f"Could not persist key v{version}: {e}")


def rotate_encryption_key() -> int:
    """
    Rotate encryption key: current becomes previous, new key becomes current.
    Returns new key version number.
    
    Old keys are retained for decryption of existing files.
    """
    global _ENCRYPTION_KEYS, _CURRENT_KEY_VERSION

    with _KEY_LOCK:
        # Load existing keys if needed
        if not _ENCRYPTION_KEYS:
            _ENCRYPTION_KEYS = _load_all_keys()
            if _ENCRYPTION_KEYS:
                _CURRENT_KEY_VERSION = max(_ENCRYPTION_KEYS.keys())

        # Archive current key as previous version
        new_version = _CURRENT_KEY_VERSION + 1
        if new_version > _MAX_KEY_VERSIONS:
            # Remove oldest version
            oldest = new_version - _MAX_KEY_VERSIONS
            old_file = _get_key_file(oldest)
            if old_file.exists():
                old_file.unlink()
                log.info(f"Removed old key version v{oldest}")
            _ENCRYPTION_KEYS.pop(oldest, None)

        # Move current to previous version
        if _CURRENT_KEY_VERSION in _ENCRYPTION_KEYS:
            _persist_key(new_version, _ENCRYPTION_KEYS[_CURRENT_KEY_VERSION])
            _ENCRYPTION_KEYS[new_version] = _ENCRYPTION_KEYS[_CURRENT_KEY_VERSION]

        # Generate new current key
        new_key = os.urandom(32)
        _ENCRYPTION_KEYS[1] = new_key
        _CURRENT_KEY_VERSION = 1
        _persist_key(1, new_key)

        log.info(f"Rotated encryption key: v{new_version} archived, new current key v1")
        return 1


def get_key_versions() -> list[int]:
    """Get list of available key versions."""
    if not _ENCRYPTION_KEYS:
        _load_all_keys()
    return sorted(_ENCRYPTION_KEYS.keys())


def get_current_key_version() -> int:
    """Get current key version."""
    if not _ENCRYPTION_KEYS:
        _load_all_keys()
    return _CURRENT_KEY_VERSION


# ─── AES ENCRYPTION ─────────────────────────────────────────────

def encrypt_file(data: bytes) -> bytes:
    """
    Encrypt raw bytes with AES-256 (Fernet or PyCryptodome CBC).
    Returns encrypted bytes with key version prefix: [version:1 byte][encrypted_data]
    """
    current_version = get_current_key_version()
    version_byte = current_version.to_bytes(1, "big")

    try:
        import base64
        import hashlib

        from cryptography.fernet import Fernet

        # Derive a valid Fernet key (32 bytes → URL-safe base64)
        raw_key = _get_key()
        fernet_key = base64.urlsafe_b64encode(hashlib.sha256(raw_key).digest())
        f = Fernet(fernet_key)
        encrypted = f.encrypt(data)
        log.debug(f"Encrypted {len(data)} bytes → {len(encrypted)} bytes (Fernet, v{current_version})")
        return version_byte + encrypted

    except ImportError:
        pass  # Try PyCryptodome

    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad

        key = _get_key()[:32]
        iv  = os.urandom(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ct = cipher.encrypt(pad(data, AES.block_size))
        result = iv + ct
        log.debug(f"Encrypted {len(data)} bytes → {len(result)} bytes (AES-CBC, v{current_version})")
        return version_byte + result

    except ImportError:
        log.warning("No encryption library available (install cryptography or pycryptodome). Storing unencrypted.")
        return version_byte + data  # Include version byte even for unencrypted fallback


def decrypt_file(data: bytes) -> bytes:
    """
    Decrypt bytes previously encrypted with encrypt_file().
    Handles key version prefix: [version:1 byte][encrypted_data]
    """
    if len(data) < 2:
        raise ValueError("Invalid encrypted data: too short")

    version = data[0]
    encrypted_data = data[1:]

    try:
        import base64
        import hashlib

        from cryptography.fernet import Fernet

        raw_key = _get_key(version)
        fernet_key = base64.urlsafe_b64encode(hashlib.sha256(raw_key).digest())
        f = Fernet(fernet_key)
        return f.decrypt(encrypted_data)

    except ImportError:
        pass

    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad

        key = _get_key(version)[:32]
        iv  = encrypted_data[:16]
        ct  = encrypted_data[16:]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size)

    except ImportError:
        return encrypted_data  # No decryption (matches encrypt fallback)

    except Exception as e:
        log.error(f"Decryption failed (v{version}): {e}")
        raise ValueError(f"Could not decrypt file with key v{version}. Key may have been rotated.") from e


# ─── TEXT EXTRACTION (LEGACY — backward compatible) ───────────────

def extract_text_pdf(raw_bytes: bytes) -> str:
    """
    Extract text from a PDF file given as raw bytes.
    Tries PyMuPDF (fitz) first, then pdfplumber as fallback.
    """
    text = ""

    # ── Method 1: PyMuPDF (fastest, best layout preservation)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pages_text.append(page.get_text("text"))
        text = "\n\n".join(pages_text)
        doc.close()
        log.info(f"PyMuPDF extracted {len(text)} chars from PDF ({len(doc)} pages)")
        return _clean_text(text)

    except ImportError:
        log.info("PyMuPDF not available, trying pdfplumber…")
    except Exception as e:
        log.warning(f"PyMuPDF failed: {e}, trying pdfplumber…")

    # ── Method 2: pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
            text = "\n\n".join(pages_text)
        log.info(f"pdfplumber extracted {len(text)} chars")
        return _clean_text(text)

    except ImportError:
        log.warning("pdfplumber not available. Install: pip install pdfplumber")
    except Exception as e:
        log.error(f"pdfplumber failed: {e}")

    # ── Method 3: PyPDF2 (basic fallback)
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages_text)
        log.info(f"PyPDF2 extracted {len(text)} chars")
        return _clean_text(text)

    except ImportError:
        log.warning("No PDF library available. Install: pip install pymupdf")
        return "Could not extract PDF text. Please install PyMuPDF: pip install pymupdf"

    except Exception as e:
        log.error(f"All PDF extraction methods failed: {e}")
        return ""


def extract_text_pptx(raw_bytes: bytes) -> str:
    """
    Extract text from a PPTX file given as raw bytes.
    Uses python-pptx library.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches

        prs = Presentation(io.BytesIO(raw_bytes))
        slide_texts = []

        for slide_num, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slide_texts.append(f"[Slide {slide_num + 1}]\n" + "\n".join(texts))

        text = "\n\n".join(slide_texts)
        log.info(f"python-pptx extracted {len(text)} chars from {len(prs.slides)} slides")
        return _clean_text(text)

    except ImportError:
        log.warning("python-pptx not available. Install: pip install python-pptx")
        return "Could not extract PPTX text. Please install python-pptx."
    except Exception as e:
        log.error(f"PPTX extraction failed: {e}")
        return ""


def _clean_text(text: str) -> str:
    """
    Remove excessive whitespace, repeated newlines, and non-UTF-8 chars.
    """
    import re

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove lone special characters that aren't content
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text.strip()


# ─── STRUCTURED DOCUMENT EXTRACTION (RAG Phase 1) ─────────────────
# These functions return structured data with page/slide metadata
# while preserving backward compatibility with existing extract_text_* functions.

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DocumentPage:
    """Represents a single page/slide with its text content."""
    page_number: int
    text: str
    metadata: dict


@dataclass
class StructuredDocument:
    """Structured document representation preserving page/slide boundaries."""
    file_id: str
    filename: str
    file_type: str  # "pdf" or "pptx"
    pages: List[DocumentPage]
    total_pages: int
    metadata: dict


def extract_pdf_pages(raw_bytes: bytes) -> List[DocumentPage]:
    """
    Extract text from PDF preserving page boundaries.
    Returns list of DocumentPage objects with page numbers.
    """
    pages = []

    # Method 1: PyMuPDF
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            cleaned = _clean_text(text)
            pages.append(DocumentPage(
                page_number=page_num + 1,
                text=cleaned,
                metadata={"source": "pymupdf", "page_index": page_num}
            ))
        doc.close()
        log.info(f"PyMuPDF extracted {len(pages)} pages from PDF")
        return pages

    except ImportError:
        log.info("PyMuPDF not available, trying pdfplumber…")
    except Exception as e:
        log.warning(f"PyMuPDF failed: {e}, trying pdfplumber…")

    # Method 2: pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    cleaned = _clean_text(t)
                    pages.append(DocumentPage(
                        page_number=page_num + 1,
                        text=cleaned,
                        metadata={"source": "pdfplumber", "page_index": page_num}
                    ))
        log.info(f"pdfplumber extracted {len(pages)} pages from PDF")
        return pages

    except ImportError:
        log.warning("pdfplumber not available. Install: pip install pdfplumber")
    except Exception as e:
        log.error(f"pdfplumber failed: {e}")

    # Method 3: PyPDF2
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        for page_num, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            cleaned = _clean_text(t)
            if cleaned:
                pages.append(DocumentPage(
                    page_number=page_num + 1,
                    text=cleaned,
                    metadata={"source": "pypdf2", "page_index": page_num}
                ))
        log.info(f"PyPDF2 extracted {len(pages)} pages from PDF")
        return pages

    except ImportError:
        log.warning("No PDF library available. Install: pip install pymupdf")
        return [DocumentPage(page_number=1, text="Could not extract PDF text. Please install PyMuPDF: pip install pymupdf", metadata={"error": True})]
    except Exception as e:
        log.error(f"All PDF extraction methods failed: {e}")
        return []


def extract_pptx_slides(raw_bytes: bytes) -> List[DocumentPage]:
    """
    Extract text from PPTX preserving slide boundaries.
    Returns list of DocumentPage objects with slide numbers.
    """
    try:
        from pptx import Presentation

        prs = Presentation(io.BytesIO(raw_bytes))
        pages = []

        for slide_num, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                cleaned = _clean_text("\n".join(texts))
                pages.append(DocumentPage(
                    page_number=slide_num + 1,
                    text=cleaned,
                    metadata={"source": "python-pptx", "slide_index": slide_num}
                ))

        log.info(f"python-pptx extracted {len(pages)} slides from PPTX")
        return pages

    except ImportError:
        log.warning("python-pptx not available. Install: pip install python-pptx")
        return [DocumentPage(page_number=1, text="Could not extract PPTX text. Please install python-pptx.", metadata={"error": True})]
    except Exception as e:
        log.error(f"PPTX extraction failed: {e}")
        return []


def extract_structured_document(
    raw_bytes: bytes,
    file_id: str,
    filename: str,
    file_type: str
) -> StructuredDocument:
    """
    Extract structured document with page/slide metadata.
    This is the main entry point for RAG document ingestion.
    """
    if file_type.lower() == "pdf":
        pages = extract_pdf_pages(raw_bytes)
    elif file_type.lower() in ("pptx", "ppt"):
        pages = extract_pptx_slides(raw_bytes)
    elif file_type.lower() in ("txt", "text"):
        # Plain text - treat as single page
        text = _clean_text(raw_bytes.decode("utf-8", errors="ignore"))
        pages = [DocumentPage(page_number=1, text=text, metadata={"source": "text"})]
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return StructuredDocument(
        file_id=file_id,
        filename=filename,
        file_type=file_type.lower(),
        pages=pages,
        total_pages=len(pages),
        metadata={"extraction_timestamp": time.time()}
    )


# ─── FILE CLEANUP ───────────────────────────────────────────────

def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    """
    Delete files older than max_age_seconds from a directory.
    Called periodically by the background cleanup thread.
    """
    now   = time.time()
    count = 0
    for f in Path(directory).iterdir():
        if f.is_file():
            age = now - f.stat().st_mtime
            if age > max_age_seconds:
                try:
                    f.unlink()
                    count += 1
                except Exception as e:
                    log.warning(f"Could not delete {f}: {e}")
    if count > 0:
        log.info(f"Cleaned {count} old files from {directory}")


# ─── KEY ROTATION CLI ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Encryption key management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Rotate key
    rotate_parser = subparsers.add_parser("rotate-key", help="Rotate encryption key (archive current, generate new)")
    rotate_parser.add_argument("--force", action="store_true", help="Force rotation even if key is recent")

    # List keys
    list_parser = subparsers.add_parser("list-keys", help="List available key versions")

    # Show current key version
    current_parser = subparsers.add_parser("current-key", help="Show current key version")

    args = parser.parse_args()

    if args.command == "rotate-key":
        new_version = rotate_encryption_key()
        print(f"Key rotated successfully. New current key: v{new_version}")
        print(f"Available versions: {get_key_versions()}")
    elif args.command == "list-keys":
        versions = get_key_versions()
        current = get_current_key_version()
        for v in versions:
            marker = " (current)" if v == current else ""
            print(f"  v{v}{marker}")
    elif args.command == "current-key":
        print(f"Current key version: v{get_current_key_version()}")
    else:
        parser.print_help()
        sys.exit(1)
