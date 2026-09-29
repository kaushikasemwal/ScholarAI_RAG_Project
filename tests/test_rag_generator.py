"""
test_rag_generator.py — Grounded Generator Tests
================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock

from backend.rag.generator import (
    GeneratorConfig,
    GroundedQuestionResult,
    FLANT5Generator,
    GroundedQuizGenerator,
    create_grounded_generator,
    generate_grounded_question,
)
from backend.rag.output_parser import GroundedQuizQuestion
from backend.rag.context_formatter import format_context_for_prompt


class MockDoc:
    """Mock LangChain Document for testing."""
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class TestGeneratorConfig:
    def test_default_config(self):
        config = GeneratorConfig()
        assert config.max_context_chunks == 5
        assert config.temperature == 0.3
        assert config.difficulty == "medium"
        assert config.topic_focus == "the key concepts"
        assert config.max_new_tokens == 64
        assert config.num_beams == 2
        assert config.do_sample == False
        assert config.max_retries == 2

    def test_custom_config(self):
        config = GeneratorConfig(
            max_context_chunks=3,
            temperature=0.5,
            difficulty="hard",
            topic_focus="specific topic"
        )
        assert config.max_context_chunks == 3
        assert config.temperature == 0.5
        assert config.difficulty == "hard"
        assert config.topic_focus == "specific topic"


class TestGroundedQuestionResult:
    def test_success_result(self):
        q = GroundedQuizQuestion(
            question="What is ML?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="Test"
        )
        result = GroundedQuestionResult(
            success=True,
            question=q,
            provenance=[{"doc": "test"}],
            retrieval_metadata={"count": 1}
        )
        assert result.success is True
        assert result.question == q

    def test_failure_result(self):
        result = GroundedQuestionResult(
            success=False,
            error="Generation failed",
            provenance=[],
            retrieval_metadata={}
        )
        assert result.success is False
        assert result.error == "Generation failed"


class TestFLANT5Generator:
    @pytest.fixture
    def mock_t5(self):
        """Mock the get_t5 function to return a mock model/tokenizer."""
        with patch('backend.rag.generator.get_t5') as mock_get_t5:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_get_t5.return_value = (mock_model, mock_tokenizer)
            
            # Setup tokenizer mock
            mock_tokenizer.return_value = {
                "input_ids": Mock()
            }
            # Return a valid question that passes validation (>= 10 words, not generic)
            mock_tokenizer.decode.return_value = "How do generative models learn to create new content effectively?"
            
            # Setup model mock
            mock_model.generate.return_value = [Mock()]
            
            yield mock_model, mock_tokenizer

    def test_generator_initialization(self, mock_t5):
        config = GeneratorConfig()
        gen = FLANT5Generator(config)
        
        # Access model to trigger lazy loading
        _ = gen.model
        _ = gen.tokenizer
        
        assert gen.config == config

    def test_generate_calls_model(self, mock_t5):
        mock_model, mock_tokenizer = mock_t5
        config = GeneratorConfig()
        gen = FLANT5Generator(config)
        
        prompt = "Test prompt"
        result = gen.generate(prompt)
        
        mock_tokenizer.assert_called_once()
        mock_model.generate.assert_called_once()
        assert result == "How do generative models learn to create new content effectively?"


class TestGroundedQuizGenerator:
    @pytest.fixture
    def sample_docs(self):
        return [
            MockDoc(
                "Machine learning is a subset of artificial intelligence.",
                {
                    "document_id": "doc-1",
                    "source_filename": "ml.pdf",
                    "page_number": 1,
                    "chunk_id": "doc-1_p1_c0_abc",
                    "similarity_score": 0.9
                }
            ),
            MockDoc(
                "Supervised learning uses labeled data for training.",
                {
                    "document_id": "doc-1",
                    "source_filename": "ml.pdf",
                    "page_number": 2,
                    "chunk_id": "doc-1_p2_c0_def",
                    "similarity_score": 0.85
                }
            )
        ]

    def test_generate_with_empty_documents(self):
        config = GeneratorConfig()
        generator = GroundedQuizGenerator(config)
        
        result = generator.generate_grounded_question([])
        
        assert result.success is False
        assert "No retrieved documents" in result.error

    def test_generate_with_empty_context(self):
        config = GeneratorConfig()
        generator = GroundedQuizGenerator(config)
        
        # Document with empty content
        empty_doc = MockDoc("", {"page_number": 1, "chunk_id": "c1"})
        result = generator.generate_grounded_question([empty_doc])
        
        assert result.success is False
        assert "empty" in result.error.lower()

    @patch('backend.rag.generator.get_t5')
    def test_generate_success(self, mock_get_t5, sample_docs):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_t5.return_value = (mock_model, mock_tokenizer)
        
        # Mock tokenizer for question generation
        mock_tokenizer.return_value = {"input_ids": Mock()}
        # Return a valid question (>= 10 words)
        mock_tokenizer.decode.return_value = "How do generative models learn to create new content effectively?"
        mock_model.generate.return_value = [Mock()]
        
        # Mock the answer validation to pass
        with patch('backend.rag.generator._validate_answer_support', return_value=(True, 0.8, "Machine learning is a subset of artificial intelligence.")):
            with patch('backend.rag.generator._extract_answer_from_context', return_value="By learning patterns in data"):
                with patch('backend.rag.generator._generate_distractors', return_value=["By memorizing training data", "By random guessing", "By copying existing content"]):
                    config = GeneratorConfig()
                    generator = GroundedQuizGenerator(config)
                    
                    result = generator.generate_grounded_question(
                        retrieved_documents=sample_docs,
                        topic_focus="machine learning",
                        difficulty="medium"
                    )
                    
                    assert result.success is True
                    assert result.question is not None
                    assert result.question.question == "How do generative models learn to create new content effectively?"
                    assert result.question.correct_answer == "By learning patterns in data"
                    assert len(result.provenance) == 2
                    assert result.provenance[0]["document_id"] == "doc-1"

    @patch('backend.rag.generator.get_t5')
    def test_generate_parsing_failure(self, mock_get_t5, sample_docs):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_t5.return_value = (mock_model, mock_tokenizer)
        
        mock_tokenizer.return_value = {"input_ids": Mock()}
        # Return a generic question that fails validation
        mock_tokenizer.decode.return_value = "Which of the following is true according to the passage?"
        mock_model.generate.return_value = [Mock()]
        
        config = GeneratorConfig()
        generator = GroundedQuizGenerator(config)
        
        result = generator.generate_grounded_question(
            retrieved_documents=sample_docs,
            topic_focus="machine learning"
        )
        
        assert result.success is False
        assert "generic" in result.error.lower() or "validation" in result.error.lower()

    @patch('backend.rag.generator.get_t5')
    def test_generate_model_failure(self, mock_get_t5, sample_docs):
        mock_get_t5.side_effect = Exception("Model loading failed")
        
        config = GeneratorConfig()
        generator = GroundedQuizGenerator(config)
        
        result = generator.generate_grounded_question(
            retrieved_documents=sample_docs
        )
        
        assert result.success is False
        assert "Model generation failed" in result.error

    @patch('backend.rag.generator.get_t5')
    def test_generate_multiple(self, mock_get_t5, sample_docs):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_t5.return_value = (mock_model, mock_tokenizer)
        
        mock_tokenizer.return_value = {"input_ids": Mock()}
        mock_tokenizer.decode.return_value = "How do generative models learn to create new content effectively?"
        mock_model.generate.return_value = [Mock()]
        
        with patch('backend.rag.generator._validate_answer_support', return_value=(True, 0.8, "Machine learning is a subset of artificial intelligence.")):
            with patch('backend.rag.generator._extract_answer_from_context', return_value="By learning patterns in data"):
                with patch('backend.rag.generator._generate_distractors', return_value=["By memorizing training data", "By random guessing", "By copying existing content"]):
                    config = GeneratorConfig()
                    generator = GroundedQuizGenerator(config)
                    
                    results = generator.generate_multiple(
                        retrieved_documents=sample_docs,
                        count=3,
                        topic_focus="machine learning"
                    )
                    
                    assert len(results) == 3
                    assert all(r.success for r in results)


class TestConvenienceFunctions:
    @patch('backend.rag.generator.get_t5')
    def test_create_grounded_generator(self, mock_get_t5):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_t5.return_value = (mock_model, mock_tokenizer)
        
        config = GeneratorConfig(max_context_chunks=3)
        generator = create_grounded_generator(config)
        
        assert isinstance(generator, GroundedQuizGenerator)
        assert generator.config.max_context_chunks == 3

    @patch('backend.rag.generator.get_t5')
    def test_generate_grounded_question_function(self, mock_get_t5):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_get_t5.return_value = (mock_model, mock_tokenizer)
        
        mock_tokenizer.return_value = {"input_ids": Mock()}
        mock_tokenizer.decode.return_value = "How do generative models learn to create new content effectively?"
        mock_model.generate.return_value = [Mock()]
        
        with patch('backend.rag.generator._validate_answer_support', return_value=(True, 0.8, "Machine learning is a subset of artificial intelligence.")):
            with patch('backend.rag.generator._extract_answer_from_context', return_value="By learning patterns in data"):
                with patch('backend.rag.generator._generate_distractors', return_value=["By memorizing training data", "By random guessing", "By copying existing content"]):
                    docs = [MockDoc("ML is a subset of AI", {"page_number": 1, "chunk_id": "c1"})]
                    
                    result = generate_grounded_question(
                        retrieved_documents=docs,
                        topic_focus="machine learning",
                        difficulty="easy",
                        max_context_chunks=3
                    )
                    
                    assert result.success is True
                    assert result.question.question == "How do generative models learn to create new content effectively?"


class TestContextFormattingInGenerator:
    def test_format_context_for_prompt(self):
        docs = [
            MockDoc(f"Content {i}", {"page_number": i, "chunk_id": f"c{i}"})
            for i in range(10)
        ]
        
        # Test with limit
        formatted = format_context_for_prompt(docs, max_chunks=3)
        assert "[Source 1]" in formatted
        assert "[Source 3]" in formatted
        assert "[Source 4]" not in formatted
        
        # Test without limit
        formatted_all = format_context_for_prompt(docs, max_chunks=None)
        assert "[Source 10]" in formatted_all


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])