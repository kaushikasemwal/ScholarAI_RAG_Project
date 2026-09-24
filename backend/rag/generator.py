"""
generator.py — Grounded Quiz Generation for RAG
================================================
Generates quiz questions grounded in retrieved evidence using FLAN-T5.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence
except ImportError:
    PromptTemplate = None
    RunnableSequence = None

from ..models import get_t5
from .prompts import format_prompt, GROUNDED_QG_SYSTEM_PROMPT
from .context_formatter import format_context_for_prompt, extract_provenance, ContextFormatConfig, create_generation_context_config
from .output_parser import (
    GroundedQuizQuestion,
    ParseResult,
    GroundedOutputParser,
    create_grounded_output_parser,
)

log = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for grounded generation."""
    max_context_chunks: int = 5
    temperature: float = 0.7
    max_output_tokens: int = 256
    num_beams: int = 4
    do_sample: bool = True
    top_p: float = 0.9
    no_repeat_ngram_size: int = 3
    context_format: Optional[ContextFormatConfig] = None
    prompt_template: str = "grounded_qg"
    difficulty: str = "medium"
    topic_focus: str = "the key concepts"


@dataclass
class GroundedQuestionResult:
    """Result of grounded question generation."""
    success: bool
    question: Optional[GroundedQuizQuestion] = None
    error: Optional[str] = None
    raw_model_output: Optional[str] = None
    prompt_used: Optional[str] = None
    provenance: List[dict] = None
    retrieval_metadata: Dict[str, Any] = None


class FLANT5Generator:
    """
    FLAN-T5 based grounded question generator.
    Reuses the shared model from ModelManager.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._model = None
        self._tokenizer = None
        self._parser = create_grounded_output_parser(strict=True)
    
    @property
    def model(self):
        """Lazy-load the shared FLAN-T5 model and tokenizer."""
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = get_t5()
        return self._model
    
    @property
    def tokenizer(self):
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = get_t5()
        return self._tokenizer
    
    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generate text using FLAN-T5.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.config.max_output_tokens,
            num_beams=self.config.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            top_p=self.config.top_p,
        )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return generated
    
    def generate_with_prompt_template(
        self,
        prompt_template: PromptTemplate,
        **kwargs
    ) -> str:
        """Generate using a LangChain PromptTemplate."""
        formatted_prompt = prompt_template.format(**kwargs)
        return self.generate(formatted_prompt)


class GroundedQuizGenerator:
    """
    Main class for generating grounded quiz questions from retrieved evidence.
    """
    
    def __init__(
        self,
        generator_config: Optional[GeneratorConfig] = None,
        t5_generator: Optional[FLANT5Generator] = None,
    ):
        self.config = generator_config or GeneratorConfig()
        # Use generation-optimized context format by default
        if self.config.context_format is None:
            self.config.context_format = create_generation_context_config()
        self.t5_generator = t5_generator or FLANT5Generator(self.config)
        self.parser = create_grounded_output_parser(strict=True)
    
    def generate_grounded_question(
        self,
        retrieved_documents: List[Any],  # LangChain Documents
        question_spec: Optional[str] = None,
        topic_focus: Optional[str] = None,
        difficulty: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> GroundedQuestionResult:
        """
        Generate a grounded quiz question from retrieved evidence.
        
        Args:
            retrieved_documents: List of retrieved LangChain Documents
            question_spec: Optional specific question direction
            topic_focus: What the question should focus on
            difficulty: Question difficulty (easy/medium/hard)
            user_id: User ID for tracking
            document_id: Document ID for tracking
            
        Returns:
            GroundedQuestionResult with question or error
        """
        # Handle empty retrieval
        if not retrieved_documents:
            return GroundedQuestionResult(
                success=False,
                error="No retrieved documents provided",
                provenance=[],
                retrieval_metadata={"retrieved_count": 0}
            )
        
        # Format context for prompt
        context = format_context_for_prompt(
            retrieved_documents,
            max_chunks=self.config.max_context_chunks,
            config=self.config.context_format
        )
        
        # Check if context is effectively empty
        # This catches: no documents, all empty content, or only metadata
        has_content = any(
            doc.page_content and doc.page_content.strip() 
            for doc in retrieved_documents[:self.config.max_context_chunks]
        )
        if not has_content:
            return GroundedQuestionResult(
                success=False,
                error="Retrieved context is empty (no extractable content)",
                provenance=extract_provenance(retrieved_documents),
                retrieval_metadata={
                    "retrieved_count": len(retrieved_documents),
                    "context_empty": True
                }
            )
        
        # Build prompt
        prompt = format_prompt(
            context=context,
            topic_focus=topic_focus or self.config.topic_focus,
            difficulty=difficulty or self.config.difficulty,
            template_name=self.config.prompt_template,
            system_prompt=GROUNDED_QG_SYSTEM_PROMPT
        )
        
        # Generate with FLAN-T5
        try:
            raw_output = self.t5_generator.generate(prompt)
        except Exception as e:
            log.error(f"FLAN-T5 generation failed: {e}")
            return GroundedQuestionResult(
                success=False,
                error=f"Model generation failed: {e}",
                raw_model_output=None,
                prompt_used=prompt,
                provenance=extract_provenance(retrieved_documents),
                retrieval_metadata={
                    "retrieved_count": len(retrieved_documents),
                    "generation_error": str(e)
                }
            )
        
        # Parse output
        parse_result = self.parser.parse(raw_output)
        
        if not parse_result.success:
            # Try with non-strict parser as fallback
            fallback_parser = create_grounded_output_parser(strict=False)
            fallback_result = fallback_parser.parse(raw_output)
            
            if not fallback_result.success:
                return GroundedQuestionResult(
                    success=False,
                    error=f"Output parsing failed: {parse_result.error}",
                    raw_model_output=raw_output,
                    prompt_used=prompt,
                    provenance=extract_provenance(retrieved_documents),
                    retrieval_metadata={
                        "retrieved_count": len(retrieved_documents),
                        "parse_error": parse_result.error
                    }
                )
            parse_result = fallback_result
        
        # Build provenance from retrieved documents
        provenance = extract_provenance(retrieved_documents)
        
        # Add retrieval metadata
        retrieval_metadata = {
            "retrieved_count": len(retrieved_documents),
            "context_chunks_used": min(len(retrieved_documents), self.config.max_context_chunks),
            "topic_focus": topic_focus or self.config.topic_focus,
            "difficulty": difficulty or self.config.difficulty,
            "user_id": user_id,
            "document_id": document_id,
        }
        
        return GroundedQuestionResult(
            success=True,
            question=parse_result.question,
            raw_model_output=raw_output,
            prompt_used=prompt,
            provenance=provenance,
            retrieval_metadata=retrieval_metadata
        )
    
    def generate_multiple(
        self,
        retrieved_documents: List[Any],
        count: int = 3,
        topic_focus: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[GroundedQuestionResult]:
        """
        Generate multiple questions from the same retrieved context.
        Note: This reuses the same context for all questions.
        """
        results = []
        for i in range(count):
            # Vary the topic focus slightly for diversity
            varied_focus = topic_focus
            if topic_focus and count > 1:
                varied_focus = f"{topic_focus} (aspect {i+1})"
            
            result = self.generate_grounded_question(
                retrieved_documents=retrieved_documents,
                topic_focus=varied_focus,
                difficulty=difficulty,
            )
            results.append(result)
        return results


def create_grounded_generator(
    config: Optional[GeneratorConfig] = None
) -> GroundedQuizGenerator:
    """Factory function to create a grounded quiz generator."""
    return GroundedQuizGenerator(generator_config=config)


def generate_grounded_question(
    retrieved_documents: List[Any],
    topic_focus: str = "the key concepts",
    difficulty: str = "medium",
    max_context_chunks: int = 5,
) -> GroundedQuestionResult:
    """
    Convenience function for one-off grounded question generation.
    
    Args:
        retrieved_documents: List of retrieved LangChain Documents
        topic_focus: What the question should focus on
        difficulty: Question difficulty
        max_context_chunks: Maximum chunks to include in context
        
    Returns:
        GroundedQuestionResult
    """
    config = GeneratorConfig(
        max_context_chunks=max_context_chunks,
        topic_focus=topic_focus,
        difficulty=difficulty
    )
    generator = create_grounded_generator(config)
    return generator.generate_grounded_question(
        retrieved_documents=retrieved_documents,
        topic_focus=topic_focus,
        difficulty=difficulty
    )


# Export
__all__ = [
    "GeneratorConfig",
    "GroundedQuestionResult",
    "FLANT5Generator",
    "GroundedQuizGenerator",
    "create_grounded_generator",
    "generate_grounded_question",
]