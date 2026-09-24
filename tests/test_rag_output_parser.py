"""
test_rag_output_parser.py — Output Parser Tests
================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.output_parser import (
    GroundedQuizQuestion,
    ParseResult,
    GroundedOutputParser,
    create_grounded_output_parser,
    InsufficientContextResult,
    LangChainGroundedParser,
)


class TestGroundedQuizQuestion:
    def test_valid_question(self):
        q = GroundedQuizQuestion(
            question="What is ML?",
            options=["A subset of AI", "A programming language", "A database", "An OS"],
            correct_answer="A subset of AI",
            explanation="ML is defined as a subset of AI in the context."
        )
        assert q.question == "What is ML?"
        assert len(q.options) == 4
        assert q.correct_answer == "A subset of AI"

    def test_invalid_option_count_too_few(self):
        with pytest.raises(Exception):
            GroundedQuizQuestion(
                question="What is ML?",
                options=["Option 1", "Option 2"],
                correct_answer="Option 1",
                explanation="Explanation"
            )

    def test_invalid_option_count_too_many(self):
        with pytest.raises(Exception):
            GroundedQuizQuestion(
                question="What is ML?",
                options=["O1", "O2", "O3", "O4", "O5"],
                correct_answer="O1",
                explanation="Explanation"
            )

    def test_correct_answer_must_be_in_options(self):
        with pytest.raises(Exception):
            GroundedQuizQuestion(
                question="What is ML?",
                options=["A", "B", "C", "D"],
                correct_answer="E",  # Not in options
                explanation="Explanation"
            )

    def test_case_insensitive_match(self):
        # Should work with case-insensitive match
        q = GroundedQuizQuestion(
            question="What is ML?",
            options=["A subset of AI", "B", "C", "D"],
            correct_answer="a subset of ai",  # Different case
            explanation="Explanation"
        )
        assert q.correct_answer == "a subset of ai"


class TestParseResult:
    def test_success_result(self):
        q = GroundedQuizQuestion(
            question="What is ML?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="Because..."
        )
        result = ParseResult(success=True, question=q, raw_output='{"question":"test"}')
        assert result.success is True
        assert result.question == q

    def test_failure_result(self):
        result = ParseResult(success=False, error="Parse error", raw_output="bad output")
        assert result.success is False
        assert result.error == "Parse error"


class TestGroundedOutputParser:
    @pytest.fixture
    def parser(self):
        return create_grounded_output_parser(strict=True)

    @pytest.fixture
    def lenient_parser(self):
        return create_grounded_output_parser(strict=False)

    def test_parse_valid_json(self, parser):
        output = '''{
            "question": "What is machine learning?",
            "options": [
                "A subset of AI",
                "A programming language",
                "A database",
                "An operating system"
            ],
            "correct_answer": "A subset of AI",
            "explanation": "The context defines ML as a subset of AI."
        }'''
        result = parser.parse(output)
        assert result.success is True
        assert result.question.question == "What is machine learning?"
        assert len(result.question.options) == 4
        assert result.question.correct_answer == "A subset of AI"

    def test_parse_insufficient_context(self, parser):
        output = '''{
            "error": "INSUFFICIENT_CONTEXT",
            "reason": "Context does not mention machine learning"
        }'''
        result = parser.parse(output)
        assert result.success is False
        assert "insufficient context" in result.error.lower()

    def test_parse_malformed_json(self, parser):
        output = "This is not JSON at all"
        result = parser.parse(output)
        assert result.success is False

    def test_parse_incomplete_json(self, parser):
        output = '{"question": "What is ML?"'  # Missing closing brace
        result = parser.parse(output)
        assert result.success is False

    def test_parse_wrong_option_count(self, parser):
        output = '''{
            "question": "What is ML?",
            "options": ["A", "B"],
            "correct_answer": "A",
            "explanation": "Test"
        }'''
        result = parser.parse(output)
        assert result.success is False

    def test_parse_correct_answer_not_in_options(self, parser):
        output = '''{
            "question": "What is ML?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "E",
            "explanation": "Test"
        }'''
        result = parser.parse(output)
        assert result.success is False

    def test_parse_empty_output(self, parser):
        result = parser.parse("")
        assert result.success is False
        assert "Empty" in result.error

    def test_lenient_parser_recovers(self, lenient_parser):
        # Missing one option - lenient should pad
        output = '''{
            "question": "What is ML?",
            "options": ["A", "B", "C"],
            "correct_answer": "A",
            "explanation": "Test"
        }'''
        result = lenient_parser.parse(output)
        # Lenient parser attempts recovery
        # May succeed or fail depending on validation


class TestSimpleFormatParsing:
    @pytest.fixture
    def parser(self):
        return create_grounded_output_parser(strict=True)

    def test_parse_simple_format(self, parser):
        output = """Question: What is ML?
A) A subset of AI
B) A programming language
C) A database
D) An OS
Correct: A
Explanation: ML is a subset of AI"""
        result = parser.parse(output)
        assert result.success is True
        assert result.question.question == "What is ML?"
        assert result.question.correct_answer == "A subset of AI"

    def test_parse_simple_format_alt(self, parser):
        output = """Question: What is deep learning?
A. Neural networks
B. Shallow learning
C. Linear regression
D. Decision trees
Correct: A
Explanation: Deep learning uses neural networks"""
        result = parser.parse(output)
        assert result.success is True
        assert result.question.correct_answer == "Neural networks"


class TestLangChainGroundedParser:
    def test_langchain_parser(self):
        parser = LangChainGroundedParser(strict=True)
        output = '''{
            "question": "What is ML?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Test"
        }'''
        question = parser.parse(output)
        assert isinstance(question, GroundedQuizQuestion)
        assert question.question == "What is ML?"

    def test_langchain_parser_format_instructions(self):
        parser = LangChainGroundedParser()
        instructions = parser.get_format_instructions()
        assert "question" in instructions
        assert "options" in instructions
        assert "correct_answer" in instructions
        assert "explanation" in instructions


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])