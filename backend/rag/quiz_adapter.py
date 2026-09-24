"""
quiz_adapter.py — RAG Quiz Adapter
===================================
Converts RAG GroundedQuizQuestion to existing ScholarAI QuizQuestion API format.
Preserves provenance internally for potential future use.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .output_parser import GroundedQuizQuestion, ParseResult
from ..schemas import QuizQuestion

log = logging.getLogger(__name__)


@dataclass
class RAGQuizQuestion:
    """Internal representation of a RAG-generated quiz question with provenance."""
    question: QuizQuestion
    provenance: List[Dict[str, Any]]
    retrieval_metadata: Dict[str, Any]
    raw_model_output: Optional[str] = None


def convert_grounded_to_quiz_question(
    grounded: GroundedQuizQuestion,
    provenance: List[Dict[str, Any]],
    retrieval_metadata: Dict[str, Any],
    raw_model_output: Optional[str] = None
) -> RAGQuizQuestion:
    """
    Convert a GroundedQuizQuestion to the existing QuizQuestion API format.
    
    Preserves provenance and retrieval metadata internally for potential
    future use (e.g., debugging, evaluation, citation display).
    
    Args:
        grounded: Parsed grounded question from RAG pipeline
        provenance: Source document provenance from retrieval
        retrieval_metadata: Metadata about the retrieval/generation
        raw_model_output: Raw model output for debugging
        
    Returns:
        RAGQuizQuestion with QuizQuestion and internal metadata
    """
    # Map correct_answer text to correct index in options
    correct_idx = 0
    for i, opt in enumerate(grounded.options):
        if opt.lower().strip() == grounded.correct_answer.lower().strip():
            correct_idx = i
            break
    
    # Ensure we have exactly 4 options
    options = grounded.options[:4]
    while len(options) < 4:
        options.append(f"Option {len(options) + 1}")
    
    quiz_question = QuizQuestion(
        question=grounded.question,
        options=options,
        correct=correct_idx,
        answer=grounded.correct_answer,
        reasoning=grounded.explanation
    )
    
    return RAGQuizQuestion(
        question=quiz_question,
        provenance=provenance,
        retrieval_metadata=retrieval_metadata,
        raw_model_output=raw_model_output
    )


def convert_multiple_grounded_to_quiz(
    results: List["GroundedQuestionResult"]
) -> List[RAGQuizQuestion]:
    """
    Convert multiple GroundedQuestionResult to RAGQuizQuestion list.
    
    Filters out failed generations and converts successful ones.
    """
    rag_questions = []
    for result in results:
        if result.success and result.question:
            rag_q = convert_grounded_to_quiz_question(
                grounded=result.question,
                provenance=result.provenance or [],
                retrieval_metadata=result.retrieval_metadata or {},
                raw_model_output=result.raw_model_output
            )
            rag_questions.append(rag_q)
    return rag_questions


def extract_quiz_questions(
    rag_questions: List[RAGQuizQuestion]
) -> List[QuizQuestion]:
    """
    Extract just the QuizQuestion objects for API response.
    
    This is the public interface - internal provenance is not exposed
    in the standard API response.
    """
    return [rq.question for rq in rag_questions]


def extract_provenance_map(
    rag_questions: List[RAGQuizQuestion]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Extract provenance map for internal use (logging, evaluation, etc.).
    
    Returns a mapping from question index to its provenance list.
    """
    return {i: rq.provenance for i, rq in enumerate(rag_questions)}


def rag_questions_to_api_format(
    rag_questions: List[RAGQuizQuestion],
    file_id: str,
    status: str = "ok"
) -> Dict[str, Any]:
    """
    Convert RAG quiz questions to the full API response format.
    
    Matches the QuizResponse schema expected by the frontend.
    """
    return {
        "file_id": file_id,
        "questions": extract_quiz_questions(rag_questions),
        "status": status
    }


# Export
__all__ = [
    "RAGQuizQuestion",
    "convert_grounded_to_quiz_question",
    "convert_multiple_grounded_to_quiz",
    "extract_quiz_questions",
    "extract_provenance_map",
    "rag_questions_to_api_format",
]