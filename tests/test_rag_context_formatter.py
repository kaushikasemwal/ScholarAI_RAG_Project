"""
test_rag_context_formatter.py — Context Formatter Tests
========================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.context_formatter import (
    ContextFormatConfig,
    format_chunk,
    format_retrieved_context,
    extract_provenance,
    format_context_for_prompt,
)


class MockDoc:
    """Mock LangChain Document for testing."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class TestContextFormatConfig:
    def test_default_config(self):
        config = ContextFormatConfig()
        assert config.include_page is True
        assert config.include_chunk_id is False  # Changed default to reduce prompt noise
        assert config.include_score is False
        assert config.max_chars_per_chunk == 1000
        assert "---" in config.separator

    def test_custom_config(self):
        config = ContextFormatConfig(
            include_page=False,
            include_score=True,
            max_chars_per_chunk=500
        )
        assert config.include_page is False
        assert config.include_score is True
        assert config.max_chars_per_chunk == 500


class TestFormatChunk:
    def test_basic_formatting(self):
        doc = MockDoc(
            page_content="Machine learning is a subset of AI.",
            metadata={
                "page_number": 1,
                "chunk_id": "doc1_p1_c0_abc123",
                "similarity_score": 0.95,
                "source_filename": "test.pdf"
            }
        )
        # Default config doesn't include chunk_id anymore
        config = ContextFormatConfig()
        formatted = format_chunk(doc, 1, config)
        
        assert "[Source 1]" in formatted
        assert "Page/Slide: 1" in formatted
        assert "Chunk ID:" not in formatted  # Not included by default
        assert "Machine learning is a subset of AI." in formatted

    def test_formatting_with_score(self):
        doc = MockDoc(
            page_content="Content here",
            metadata={"page_number": 2, "chunk_id": "c2", "similarity_score": 0.87}
        )
        config = ContextFormatConfig(include_score=True)
        formatted = format_chunk(doc, 1, config)
        assert "Relevance: 0.870" in formatted

    def test_formatting_without_page(self):
        doc = MockDoc(
            page_content="Content",
            metadata={"chunk_id": "c1"}
        )
        config = ContextFormatConfig(include_page=False)
        formatted = format_chunk(doc, 1, config)
        assert "Page/Slide:" not in formatted
        # When page is disabled, it shouldn't show "Page/Slide: Unknown"

    def test_formatting_without_chunk_id(self):
        doc = MockDoc(
            page_content="Content",
            metadata={"page_number": 1}
        )
        config = ContextFormatConfig(include_chunk_id=False)
        formatted = format_chunk(doc, 1, config)
        assert "Chunk ID:" not in formatted

    def test_content_truncation(self):
        long_content = "A" * 1500
        doc = MockDoc(
            page_content=long_content,
            metadata={"page_number": 1, "chunk_id": "c1"}
        )
        config = ContextFormatConfig(max_chars_per_chunk=500)
        formatted = format_chunk(doc, 1, config)
        assert len(formatted) < 1600  # Content should be truncated
        assert "[truncated]" in formatted


class TestFormatRetrievedContext:
    def test_multiple_documents(self):
        docs = [
            MockDoc("Content 1", {"page_number": 1, "chunk_id": "c1"}),
            MockDoc("Content 2", {"page_number": 2, "chunk_id": "c2"}),
            MockDoc("Content 3", {"page_number": 1, "chunk_id": "c3"}),
        ]
        config = ContextFormatConfig()
        formatted = format_retrieved_context(docs, config)
        
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted
        assert "[Source 3]" in formatted
        assert "---" in formatted  # separator

    def test_empty_documents(self):
        formatted = format_retrieved_context([], ContextFormatConfig())
        assert formatted == "[No relevant context retrieved]"

    def test_custom_separator(self):
        docs = [
            MockDoc("Content 1", {"page_number": 1, "chunk_id": "c1"}),
            MockDoc("Content 2", {"page_number": 2, "chunk_id": "c2"}),
        ]
        config = ContextFormatConfig(separator="\n|||\n")
        formatted = format_retrieved_context(docs, config)
        assert "|||" in formatted


class TestExtractProvenance:
    def test_extract_provenance(self):
        docs = [
            MockDoc(
                "Content 1",
                {
                    "document_id": "doc-1",
                    "source_filename": "test.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                    "chunk_id": "doc-1_p1_c0_abc",
                    "chunk_type": "chunk",
                    "similarity_score": 0.9,
                    "rank": 1
                }
            ),
            MockDoc(
                "Content 2",
                {
                    "document_id": "doc-2",
                    "source_filename": "test2.pdf",
                    "file_type": "pptx",
                    "page_number": 2,
                    "chunk_id": "doc-2_p2_c0_def",
                    "chunk_type": "chunk",
                    "similarity_score": 0.8,
                    "rank": 2
                }
            )
        ]
        
        provenance = extract_provenance(docs)
        
        assert len(provenance) == 2
        assert provenance[0]["document_id"] == "doc-1"
        assert provenance[0]["source_filename"] == "test.pdf"
        assert provenance[0]["page_or_slide"] == 1
        assert provenance[0]["chunk_id"] == "doc-1_p1_c0_abc"
        assert provenance[0]["similarity_score"] == 0.9
        assert provenance[1]["document_id"] == "doc-2"
        assert provenance[1]["page_or_slide"] == 2


class TestFormatContextForPrompt:
    def test_limit_chunks(self):
        docs = [
            MockDoc(f"Content {i}", {"page_number": i, "chunk_id": f"c{i}"})
            for i in range(10)
        ]
        formatted = format_context_for_prompt(docs, max_chunks=3)
        
        # Should only have 3 sources
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted
        assert "[Source 3]" in formatted
        assert "[Source 4]" not in formatted

    def test_no_limit(self):
        docs = [
            MockDoc(f"Content {i}", {"page_number": i, "chunk_id": f"c{i}"})
            for i in range(3)
        ]
        formatted = format_context_for_prompt(docs, max_chunks=None)
        
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted
        assert "[Source 3]" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])