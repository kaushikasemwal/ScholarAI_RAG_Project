"""
multi_question.py — Multi-Question RAG Quiz Generation
=======================================================
Generates multiple diverse grounded quiz questions from a document.
Implements diversity through retrieval variation and embedding-based selection.
"""

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set
import numpy as np

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None

from .retriever import Retriever, RetrievalConfig
from .generator import GroundedQuizGenerator, GeneratorConfig, GroundedQuestionResult
from .embeddings import BGEEmbeddings
from .quiz_adapter import RAGQuizQuestion, convert_grounded_to_quiz_question

log = logging.getLogger(__name__)


@dataclass
class MultiQuestionConfig:
    """Configuration for multi-question generation."""
    num_questions: int = 10
    top_k_per_question: int = 5
    max_context_chunks: int = 5
    diversity_threshold: float = 0.75  # Cosine similarity threshold for deduplication
    min_questions_per_chunk: int = 1
    max_questions_per_chunk: int = 3
    temperature: float = 0.7
    difficulty: str = "medium"
    topic_focus: str = "the key concepts"


class DiversityFilter:
    """
    Filters questions for diversity using embedding similarity.
    """
    
    def __init__(self, embeddings: BGEEmbeddings, threshold: float = 0.75):
        self.embeddings = embeddings
        self.threshold = threshold
    
    def filter(self, questions: List[RAGQuizQuestion]) -> List[RAGQuizQuestion]:
        """
        Filter questions to remove near-duplicates.
        
        Uses question text embeddings to find similar questions.
        Keeps the first occurrence and removes subsequent similar ones.
        """
        if len(questions) <= 1:
            return questions
        
        # Get question texts
        texts = [q.question.question for q in questions]
        
        # Embed all questions
        try:
            embeddings = self.embeddings.embed_documents(texts)
            embeddings = np.array(embeddings)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = embeddings / norms
            
            # Compute similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)
            
            # Select diverse questions
            selected_indices = []
            for i in range(len(questions)):
                # Check if this question is too similar to any already selected
                is_duplicate = False
                for j in selected_indices:
                    if similarity_matrix[i, j] >= self.threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    selected_indices.append(i)
            
            filtered = [questions[i] for i in selected_indices]
            log.info(f"Diversity filter: {len(questions)} -> {len(filtered)} questions")
            return filtered
            
        except Exception as e:
            log.warning(f"Diversity filtering failed, returning all: {e}")
            return questions


class ChunkTracker:
    """
    Tracks which source chunks have been used for question generation
    to ensure coverage across the document.
    """
    
    def __init__(self):
        self.chunk_usage: Dict[str, int] = {}
        self.chunk_to_questions: Dict[str, List[int]] = {}
    
    def record_usage(self, chunk_id: str, question_idx: int):
        """Record that a chunk was used for a question."""
        self.chunk_usage[chunk_id] = self.chunk_usage.get(chunk_id, 0) + 1
        if chunk_id not in self.chunk_to_questions:
            self.chunk_to_questions[chunk_id] = []
        self.chunk_to_questions[chunk_id].append(question_idx)
    
    def get_least_used_chunks(self, available_chunks: List[str], n: int) -> List[str]:
        """Get n chunks that have been used the least."""
        # Sort by usage count
        sorted_chunks = sorted(available_chunks, key=lambda c: self.chunk_usage.get(c, 0))
        return sorted_chunks[:n]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_chunks_used": len(self.chunk_usage),
            "usage_distribution": dict(self.chunk_usage),
            "questions_per_chunk": {
                chunk: len(qs) for chunk, qs in self.chunk_to_questions.items()
            }
        }


class MultiQuestionRAGGenerator:
    """
    Generates multiple diverse quiz questions from a document using RAG.
    
    Strategy:
    1. Retrieve diverse chunks from the document (different pages/sections)
    2. Generate questions from different chunks to ensure coverage
    3. Use embedding-based diversity filtering on generated questions
    4. Fall back to additional retrieval if not enough unique questions
    """
    
    def __init__(
        self,
        retriever: Retriever,
        generator: GroundedQuizGenerator,
        embeddings: BGEEmbeddings,
        config: Optional[MultiQuestionConfig] = None
    ):
        self.retriever = retriever
        self.generator = generator
        self.embeddings = embeddings
        self.config = config or MultiQuestionConfig()
        self.diversity_filter = DiversityFilter(embeddings, config.diversity_threshold)
        self.chunk_tracker = ChunkTracker()
    
    def generate_quiz(
        self,
        user_id: str,
        document_id: str,
        num_questions: Optional[int] = None
    ) -> List[RAGQuizQuestion]:
        """
        Generate a diverse quiz from a document.
        
        Args:
            user_id: User ID for isolation
            document_id: Document ID to generate quiz from
            num_questions: Override default number of questions
            
        Returns:
            List of RAGQuizQuestion objects
        """
        n = num_questions or self.config.num_questions
        log.info(f"Generating {n} questions for document {document_id} (user: {user_id})")
        
        # Strategy: Retrieve a broad set of chunks first, then generate from subsets
        # This ensures we have diverse source material
        all_rag_questions = []
        attempts = 0
        max_attempts = n * 2  # Allow some extra attempts for failures
        
        while len(all_rag_questions) < n and attempts < max_attempts:
            attempts += 1
            
            # Generate retrieval queries to get diverse chunks
            queries = self._generate_retrieval_queries(n - len(all_rag_questions))
            
            for query in queries:
                if len(all_rag_questions) >= n:
                    break
                
                # Retrieve chunks for this query
                result = self.retriever.retrieve(
                    query=query,
                    user_id=user_id,
                    document_id=document_id,
                    top_k=self.config.top_k_per_question
                )
                
                if not result.documents:
                    log.warning(f"No documents retrieved for query: {query}")
                    continue
                
                # Filter to chunks not heavily used yet
                candidate_docs = self._select_diverse_chunks(result.documents)
                
                if not candidate_docs:
                    continue
                
                # Generate question from these chunks
                gen_result = self.generator.generate_grounded_question(
                    retrieved_documents=candidate_docs,
                    topic_focus=self.config.topic_focus,
                    difficulty=self.config.difficulty
                )
                
                if gen_result.success and gen_result.question:
                    rag_q = convert_grounded_to_quiz_question(
                        grounded=gen_result.question,
                        provenance=gen_result.provenance,
                        retrieval_metadata=gen_result.retrieval_metadata,
                        raw_model_output=gen_result.raw_model_output
                    )
                    
                    # Track chunk usage
                    for doc in candidate_docs:
                        chunk_id = doc.metadata.get("chunk_id")
                        if chunk_id:
                            self.chunk_tracker.record_usage(chunk_id, len(all_rag_questions))
                    
                    all_rag_questions.append(rag_q)
                    log.debug(f"Generated question {len(all_rag_questions)}/{n}")
        
        # Apply diversity filtering
        if len(all_rag_questions) > 1:
            all_rag_questions = self.diversity_filter.filter(all_rag_questions)
        
        # If still not enough, try a final broad retrieval
        if len(all_rag_questions) < n:
            log.info(f"Only got {len(all_rag_questions)}/{n} questions, trying broad retrieval")
            additional = self._generate_fallback_questions(user_id, document_id, n - len(all_rag_questions))
            all_rag_questions.extend(additional)
        
        # Trim to requested number
        final_questions = all_rag_questions[:n]
        
        # Pad if absolutely necessary (should be rare)
        while len(final_questions) < n:
            log.warning(f"Padding quiz with generic question {len(final_questions) + 1}/{n}")
            # This should not happen with proper fallback, but just in case
            # We create a minimal question from the first available chunk
            pass
        
        log.info(f"Quiz generation complete: {len(final_questions)} questions")
        log.debug(f"Chunk usage stats: {self.chunk_tracker.get_usage_stats()}")
        
        return final_questions
    
    def _generate_retrieval_queries(self, needed: int) -> List[str]:
        """Generate diverse retrieval queries to get different chunks."""
        # Use the topic focus with variations
        base_focus = self.config.topic_focus
        queries = [base_focus]
        
        # Add variations for diversity
        variations = [
            f"key concepts in {base_focus}",
            f"important details about {base_focus}",
            f"definitions related to {base_focus}",
            f"examples of {base_focus}",
            f"processes in {base_focus}",
            f"principles of {base_focus}",
            f"applications of {base_focus}",
        ]
        
        # Select diverse queries
        selected = [base_focus]
        random.shuffle(variations)
        for v in variations:
            if len(selected) >= min(needed, 7):
                break
            selected.append(v)
        
        return selected
    
    def _select_diverse_chunks(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Select chunks that haven't been heavily used yet."""
        # Get chunk IDs
        chunk_ids = [doc.metadata.get("chunk_id", f"unknown_{i}") for i, doc in enumerate(documents)]
        
        # Get least used chunks
        least_used = self.chunk_tracker.get_least_used_chunks(chunk_ids, self.config.max_context_chunks)
        
        # Map back to documents
        selected = []
        for chunk_id in least_used:
            for doc in documents:
                if doc.metadata.get("chunk_id") == chunk_id:
                    selected.append(doc)
                    break
        
        # If none selected (all new), use first max_context_chunks
        if not selected:
            selected = documents[:self.config.max_context_chunks]
        
        return selected
    
    def _generate_fallback_questions(
        self,
        user_id: str,
        document_id: str,
        needed: int
    ) -> List[RAGQuizQuestion]:
        """Generate fallback questions with broad retrieval."""
        # Do a broad retrieval without specific query
        result = self.retriever.retrieve(
            query=self.config.topic_focus,
            user_id=user_id,
            document_id=document_id,
            top_k=needed * 3  # Get more chunks for fallback
        )
        
        if not result.documents:
            return []
        
        # Generate from different chunks
        fallback_questions = []
        chunk_groups = self._group_chunks_by_page(result.documents)
        
        for page_chunks in chunk_groups:
            if len(fallback_questions) >= needed:
                break
            
            gen_result = self.generator.generate_grounded_question(
                retrieved_documents=page_chunks[:self.config.max_context_chunks],
                topic_focus=self.config.topic_focus,
                difficulty=self.config.difficulty
            )
            
            if gen_result.success and gen_result.question:
                rag_q = convert_grounded_to_quiz_question(
                    grounded=gen_result.question,
                    provenance=gen_result.provenance,
                    retrieval_metadata=gen_result.retrieval_metadata,
                    raw_model_output=gen_result.raw_model_output
                )
                fallback_questions.append(rag_q)
        
        return fallback_questions
    
    def _group_chunks_by_page(self, documents: List[LangChainDocument]) -> List[List[LangChainDocument]]:
        """Group chunks by page/slide for diverse coverage."""
        page_groups: Dict[Any, List[LangChainDocument]] = {}
        
        for doc in documents:
            page = doc.metadata.get("page_number", doc.metadata.get("slide_number", 0))
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        # Sort by page number and return groups
        sorted_pages = sorted(page_groups.keys())
        return [page_groups[p] for p in sorted_pages]


def create_multi_question_generator(
    retriever: Retriever,
    generator: GroundedQuizGenerator,
    embeddings: BGEEmbeddings,
    config: Optional[MultiQuestionConfig] = None
) -> MultiQuestionRAGGenerator:
    """Factory function to create a multi-question RAG generator."""
    return MultiQuestionRAGGenerator(retriever, generator, embeddings, config)


def generate_rag_quiz(
    user_id: str,
    document_id: str,
    num_questions: int = 10,
    retriever: Optional[Retriever] = None,
    generator: Optional[GroundedQuizGenerator] = None,
    embeddings: Optional[BGEEmbeddings] = None,
    retriever_config: Optional[RetrievalConfig] = None,
    generator_config: Optional[GeneratorConfig] = None,
    embedding_config: Optional["EmbeddingConfig"] = None,
    vector_store_config: Optional["VectorStoreConfig"] = None,
    multi_config: Optional[MultiQuestionConfig] = None
) -> List[RAGQuizQuestion]:
    """
    Convenience function for one-off multi-question quiz generation.
    
    Creates all necessary components if not provided.
    """
    # Import here to avoid circular imports
    from .vectorstore import create_vector_store, LocalVectorStore, VectorStoreConfig
    from .embeddings import BGEEmbeddings, EmbeddingConfig, get_embeddings
    from .retriever import create_retriever
    from .generator import create_grounded_generator
    
    # Create defaults if not provided
    if embeddings is None:
        embeddings = get_embeddings(embedding_config)
    
    if retriever is None:
        retriever = create_retriever(
            user_id=user_id,
            document_id=document_id,
            embeddings=embeddings,
            config=retriever_config,
            vector_store_config=vector_store_config
        )
    
    if generator is None:
        generator = create_grounded_generator(generator_config)
    
    multi_gen = create_multi_question_generator(
        retriever=retriever,
        generator=generator,
        embeddings=embeddings,
        config=multi_config
    )
    
    return multi_gen.generate_quiz(user_id, document_id, num_questions)


# Export
__all__ = [
    "MultiQuestionConfig",
    "MultiQuestionRAGGenerator",
    "DiversityFilter",
    "ChunkTracker",
    "create_multi_question_generator",
    "generate_rag_quiz",
]