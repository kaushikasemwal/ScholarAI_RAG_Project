"""
multi_question.py — Multi-Question RAG Quiz Generation
=======================================================
Generates multiple diverse grounded quiz questions from a document.
Implements diversity through retrieval variation and embedding-based selection.
"""

import logging
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set
import numpy as np

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None

from .retriever import Retriever, RetrievalConfig
from .generator import GroundedQuizGenerator, GeneratorConfig, GroundedQuestionResult, _extract_answer_candidates
from .embeddings import BGEEmbeddings
from .quiz_adapter import RAGQuizQuestion, convert_grounded_to_quiz_question

log = logging.getLogger(__name__)


@dataclass
class MultiQuestionConfig:
    """Configuration for multi-question generation."""
    num_questions: int = 10
    top_k_per_question: int = 5
    max_context_chunks: int = 5
    diversity_threshold: float = 0.65  # Cosine similarity threshold for deduplication (lowered from 0.75)
    diversity_use_answer: bool = True  # Include answer text in diversity embeddings
    min_questions_per_chunk: int = 1
    max_questions_per_chunk: int = 3
    temperature: float = 0.7
    difficulty: str = "medium"
    topic_focus: str = "the key concepts"


class DiversityFilter:
    """
    Filters questions for diversity using embedding similarity.
    
    Supports multiple diversity strategies:
    - Question + Answer embeddings (more discriminative)
    - Source chunk diversity (encourage questions from different pages)
    - Configurable similarity threshold
    """
    
    def __init__(
        self, 
        embeddings: BGEEmbeddings, 
        threshold: float = 0.65,
        use_answer: bool = True
    ):
        self.embeddings = embeddings
        self.threshold = threshold
        self.use_answer = use_answer
    
    def filter(self, questions: List[RAGQuizQuestion]) -> List[RAGQuizQuestion]:
        """
        Filter questions to remove near-duplicates.
        
        Uses question (+ answer) text embeddings to find similar questions.
        Keeps the first occurrence and removes subsequent similar ones.
        Also considers source chunk diversity as a secondary criterion.
        """
        if len(questions) <= 1:
            return questions
        
        # Get texts for embedding: question (+ answer if enabled)
        if self.use_answer:
            texts = [
                f"{q.question.question} | {q.question.answer}" 
                for q in questions
            ]
        else:
            texts = [q.question.question for q in questions]
        
        # Embed all texts
        try:
            embeddings = self.embeddings.embed_documents(texts)
            embeddings = np.array(embeddings)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = embeddings / norms
            
            # Compute similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)
            
            # Select diverse questions using greedy algorithm with tie-breaking
            # Prefer questions from less-used source chunks
            selected_indices = []
            
            # Get source chunk info for tie-breaking
            chunk_usage = {}
            for i, q in enumerate(questions):
                # Get provenance from retrieval metadata
                chunk_ids = []
                for prov in q.provenance:
                    if "chunk_id" in prov:
                        chunk_ids.append(prov["chunk_id"])
                chunk_usage[i] = chunk_ids
            
            # Score each question: lower chunk usage = higher priority
            chunk_usage_counts = {}
            for i, chunk_ids in chunk_usage.items():
                for cid in chunk_ids:
                    chunk_usage_counts[cid] = chunk_usage_counts.get(cid, 0) + 1
            
            question_scores = []
            for i, chunk_ids in chunk_usage.items():
                if chunk_ids:
                    avg_usage = sum(chunk_usage_counts.get(cid, 0) for cid in chunk_ids) / len(chunk_ids)
                else:
                    avg_usage = 0
                question_scores.append((i, avg_usage))
            
            # Sort by chunk usage (ascending - prefer less-used chunks)
            question_scores.sort(key=lambda x: x[1])
            
            # Greedy selection with diversity constraint
            selected_indices = []
            for idx, _ in question_scores:
                # Check if this question is too similar to any already selected
                is_duplicate = False
                for j in selected_indices:
                    if similarity_matrix[idx, j] >= self.threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    selected_indices.append(idx)
            
            filtered = [questions[i] for i in selected_indices]
            log.info(f"Diversity filter: {len(questions)} -> {len(filtered)} questions "
                     f"(threshold={self.threshold}, use_answer={self.use_answer})")
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
    1. Do initial broad retrieval to understand document structure
    2. Extract answer candidates to generate content-aware retrieval queries
    3. Retrieve diverse chunks from different pages/sections
    4. Generate questions using ANSWER-FIRST approach (guarantees grounding)
    5. Use embedding-based diversity filtering on generated questions
    6. Fall back to additional retrieval if not enough unique questions
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
        self.diversity_filter = DiversityFilter(
            embeddings, 
            self.config.diversity_threshold,
            self.config.diversity_use_answer
        )
        self.chunk_tracker = ChunkTracker()
        self._document_structure: Optional[Dict[str, Any]] = None
    
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
        
        # Step 1: Analyze document structure to generate better queries
        self._analyze_document_structure(user_id, document_id)
        
        # Step 2: Generate content-aware retrieval queries
        queries = self._generate_content_aware_queries(n)
        
        all_rag_questions = []
        attempts = 0
        max_attempts = n * 3  # Allow more attempts with answer-first approach
        
        # Try each query to generate questions
        for query_idx, query in enumerate(queries):
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
            
            # Filter out non-educational pages from retrieved documents
            def is_educational_doc(doc):
                page = doc.metadata.get("page_number", doc.metadata.get("slide_number", 0))
                content = doc.page_content.lower()
                skip_patterns = [
                    'some interesting examples',
                    'identify the context of the image',
                    'response from chatgpt',
                    'response from claude',
                    'response from gemini',
                    'prompt used',
                    'quote attributed',
                    'j.r.r. tolkien',
                    'gandalf',
                    'campus tower',
                    'university landmarks',
                    'chess',
                    'alan turing',
                    'mygreatlearning',
                    'simulation or modeling',
                    'simulation of physical',
                    'image, the intended audience',
                    'research question being addressed',
                    'challenges to be considered',
                    'essence of the quote',
                    'time is a precious',
                    'paralyzed by our circumstances',
                    'well-known line from j.r.r.',
                    'fellowship of the ring',
                    'lord of the rings',
                ]
                for pattern in skip_patterns:
                    if pattern in content:
                        return False
                return True
            
            filtered_docs = [d for d in result.documents if is_educational_doc(d)]
            if not filtered_docs:
                log.warning(f"All retrieved documents filtered out for query: {query}")
                continue
            
            # Filter to chunks not heavily used yet
            candidate_docs = self._select_diverse_chunks(filtered_docs)
            
            if not candidate_docs:
                continue
            
            # Generate question using ANSWER-FIRST approach
            gen_result = self.generator.generate_grounded_question_answer_first(
                retrieved_documents=candidate_docs,
                topic_focus=self.config.topic_focus,
                difficulty=self.config.difficulty,
                user_id=user_id,
                document_id=document_id
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
                log.debug(f"Generated question {len(all_rag_questions)}/{n} from query: {query[:50]}")
            
            attempts += 1
            if attempts >= max_attempts:
                break
        
        # Apply diversity filtering
        if len(all_rag_questions) > 1:
            all_rag_questions = self.diversity_filter.filter(all_rag_questions)
        
        # If still not enough, try fallback with broad retrieval
        if len(all_rag_questions) < n:
            log.info(f"Only got {len(all_rag_questions)}/{n} questions, trying fallback retrieval")
            additional = self._generate_fallback_questions(user_id, document_id, n - len(all_rag_questions))
            all_rag_questions.extend(additional)
            
            # Re-apply diversity filtering after fallback
            if len(all_rag_questions) > 1:
                all_rag_questions = self.diversity_filter.filter(all_rag_questions)
        
        # Trim to requested number
        final_questions = all_rag_questions[:n]
        
        log.info(f"Quiz generation complete: {len(final_questions)} questions")
        log.debug(f"Chunk usage stats: {self.chunk_tracker.get_usage_stats()}")
        
        return final_questions

    def _analyze_document_structure(self, user_id: str, document_id: str):
        """Analyze document structure by doing broad retrieval to understand content."""
        if self._document_structure is not None:
            return
        
        # Do a broad retrieval to get chunks from across the document
        result = self.retriever.retrieve(
            query=self.config.topic_focus,
            user_id=user_id,
            document_id=document_id,
            top_k=100  # Get many chunks to understand structure
        )
        
        if not result.documents:
            self._document_structure = {"pages": [], "page_summaries": {}}
            return
        
        # Group by page and extract key terms from each page
        page_groups = {}
        for doc in result.documents:
            page = doc.metadata.get("page_number", doc.metadata.get("slide_number", 0))
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(doc)
        
        # Determine educational pages by checking ALL retrieved pages for skip patterns
        def is_educational_page(page_num, docs):
            combined_text = " ".join(d.page_content for d in docs[:3]).lower()
            skip_patterns = [
                'some interesting examples',
                'identify the context of the image',
                'response from chatgpt',
                'response from claude',
                'response from gemini',
                'prompt used',
                'false non-match rate',
                'biometric',
                'fnmr',
                'quote attributed',
                'j.r.r. tolkien',
                'gandalf',
                'campus tower',
                'university landmarks',
                'chess',
                'alan turing',
                'mygreatlearning',
                'simulation or modeling',
                'simulation of physical',
                'image, the intended audience',
                'research question being addressed',
                'challenges to be considered',
                'essence of the quote',
                'time is a precious',
                'paralyzed by our circumstances',
                'well-known line from j.r.r.',
                'fellowship of the ring',
                'lord of the rings',
            ]
            for pattern in skip_patterns:
                if pattern in combined_text:
                    return False
            return True
        
        # Extract key terms from each educational page
        page_summaries = {}
        for page, docs in page_groups.items():
            if not is_educational_page(page, docs):
                continue
            combined_text = " ".join(d.page_content for d in docs[:3])
            keywords = self._extract_key_terms(combined_text, n=10)
            page_summaries[page] = {
                "keywords": keywords,
                "chunk_count": len(docs),
                "sample_content": combined_text[:200]
}
        
        self._document_structure = {
            "pages": sorted(page_summaries.keys()),
            "page_summaries": page_summaries,
            "total_chunks": len(result.documents)
        }
        log.info(f"Document structure: {len(page_summaries)} educational pages, {len(result.documents)} total chunks")

    def _extract_key_terms(self, text: str, n: int = 10) -> List[str]:
        """Extract key terms from text using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import nltk
            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 2:
                sentences = [text]
            tfidf = TfidfVectorizer(max_features=n, stop_words="english", ngram_range=(1, 2))
            tfidf.fit(sentences)
            return list(tfidf.vocabulary_.keys())
        except Exception:
            words = [w for w in text.split() if len(w) > 4]
            return list(set(words))[:n]
    
    def _is_generic_query(self, query: str) -> bool:
        """Check if a query is too generic/abstract to be useful for retrieval."""
        query_lower = query.lower().strip()
        
        # Generic patterns that indicate poor retrieval queries
        generic_patterns = [
            # Abstract single-word + generic suffix
            r'^(generative|learning|foundation|model|large|language|ai)\s+(explanation|foundation|learning|model|concept|introduction)$',
            r'^(explanation|foundation|learning|model|concept|introduction)\s+(generative|learning|foundation|model|large|language|ai)$',
            
            # Generic page queries
            r'^page\s+\d+\s+the\s+key\s+concepts$',
            
            # Too short/abstract
            r'^(generative|learning|foundation|model|large|language|ai)\s+(explanation|foundation|learning)$',
        ]
        
        import re
        for pattern in generic_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check if query consists mostly of generic words
        words = query_lower.split()
        generic_words = {'generative', 'learning', 'foundation', 'model', 'large', 'language', 'ai', 'explanation', 'concept', 'introduction', 'key', 'concepts', 'details', 'definitions', 'examples', 'processes', 'principles', 'applications', 'comparison', 'advantages', 'disadvantages', 'history', 'evolution'}
        if len(words) <= 3:
            generic_count = sum(1 for w in words if w in generic_words)
            if generic_count >= len(words) - 1:  # All but one word are generic
                return True
        
        return False
    
    def _normalize_query_intent(self, query: str) -> str:
        """Normalize query for near-duplicate detection."""
        query_lower = query.lower().strip()
        # Remove common prefixes/suffixes
        query_lower = re.sub(r'^(what is|what are|how does|how do|why does|why do|explain|describe)\s+', '', query_lower)
        query_lower = re.sub(r'\s+(explanation|definition|concept|model|work|works)$', '', query_lower)
        # Sort words for order-independent comparison
        words = sorted(query_lower.split())
        return ' '.join(words)
    
    def _generate_content_aware_queries(self, needed: int) -> List[str]:
        """Generate retrieval queries based on actual document content."""
        if not self._document_structure or not self._document_structure.get("page_summaries"):
            # Fallback to generic queries
            return self._generate_generic_queries(needed)
        
        queries = []
        page_summaries = self._document_structure["page_summaries"]
        
        # Generate queries from each page's keywords - prioritize meaningful multi-word terms
        for page in sorted(page_summaries.keys()):
            summary = page_summaries[page]
            keywords = summary["keywords"]
            
            if keywords:
                # Separate single-word and multi-word keywords
                single_words = [kw for kw in keywords if ' ' not in kw and len(kw) > 3]
                multi_words = [kw for kw in keywords if ' ' in kw]
                
                # Prioritize multi-word keywords - they are more specific and retrieval-friendly
                for kw in multi_words[:3]:
                    queries.append(kw)
                    queries.append(f"{kw} explanation")
                    queries.append(f"what is {kw}")
                    queries.append(f"how does {kw} work")
                
                # Use single words only in meaningful combinations with multi-word terms
                for kw in single_words[:3]:
                    # Only create combinations with multi-word terms
                    for mw in multi_words[:2]:
                        queries.append(f"{kw} {mw}")
                        queries.append(f"{mw} {kw}")
                    # Don't create single-word + single-word or single-word + generic suffix queries
                
                # Also use single words directly if they're technical terms (contain special chars)
                for kw in single_words[:2]:
                    if any(c in kw for c in '-./'):  # Technical terms like GPT-2, BERT, etc.
                        queries.append(kw)
                        queries.append(f"{kw} explanation")
                        queries.append(f"what is {kw}")
        
        # Add explicit queries for known educational topics (proven effective)
        topic_queries = [
            "discriminative model vs generative model",
            "generative AI definition",
            "large language models LLM",
            "transformer architecture",
            "GPT BERT",
            "diffusion models",
            "generative AI evaluation metrics",
            "RAG retrieval augmented generation",
            "fine tuning PEFT LoRA",
            "generative AI applications",
            "generative AI history evolution",
            "GAN VAE generative modeling",
            "content creation generative AI",
            "versatility interactivity generative AI",
            "ChatGPT GPT-3 GPT-4",
        ]
        queries.extend(topic_queries)
        
        # Filter out generic queries
        filtered_queries = [q for q in queries if not self._is_generic_query(q)]
        
        # Near-deduplicate: remove queries with same normalized intent
        seen_intents = set()
        unique_queries = []
        for q in filtered_queries:
            norm = self._normalize_query_intent(q)
            if norm not in seen_intents:
                seen_intents.add(norm)
                unique_queries.append(q)
        
        # Also remove exact duplicates (preserve order)
        seen = set()
        final_queries = []
        for q in unique_queries:
            if q not in seen:
                seen.add(q)
                final_queries.append(q)
        
        # Ensure we have enough queries - if content-aware queries are too few, pad with topic queries
        if len(final_queries) < max(needed * 2, 15):
            # Add back topic queries that were filtered out
            for q in topic_queries:
                if len(final_queries) >= max(needed * 4, 20):
                    break
                if q not in seen:
                    seen.add(q)
                    final_queries.append(q)
        
        # Limit to needed * 4 queries (allow some failures)
        return final_queries[:max(needed * 4, 20)]
    
    def _generate_generic_queries(self, needed: int) -> List[str]:
        """Fallback generic queries."""
        base_focus = self.config.topic_focus
        queries = [base_focus]
        
        variations = [
            f"key concepts in {base_focus}",
            f"important details about {base_focus}",
            f"definitions related to {base_focus}",
            f"examples of {base_focus}",
            f"processes in {base_focus}",
            f"principles of {base_focus}",
            f"applications of {base_focus}",
            f"comparison in {base_focus}",
            f"advantages disadvantages {base_focus}",
            f"history evolution {base_focus}",
        ]
        
        selected = [base_focus]
        random.shuffle(variations)
        for v in variations:
            if len(selected) >= min(needed * 2, 10):
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
        """Generate fallback questions with broad retrieval using answer-first approach."""
        # Use diverse educational queries for fallback instead of generic query
        fallback_queries = [
            "discriminative model vs generative model",
            "generative AI definition",
            "large language models LLM",
            "transformer architecture",
            "GPT BERT",
            "diffusion models",
            "generative AI evaluation metrics",
            "RAG retrieval augmented generation",
            "fine tuning PEFT LoRA",
            "generative AI applications",
            "generative AI history evolution",
            "GAN VAE generative modeling",
            "content creation generative AI",
            "versatility interactivity generative AI",
            "ChatGPT GPT-3 GPT-4",
        ]
        
        all_filtered_docs = []
        for query in fallback_queries:
            if len(all_filtered_docs) >= needed * 5:
                break
            result = self.retriever.retrieve(
                query=query,
                user_id=user_id,
                document_id=document_id,
                top_k=5
            )
            
            if not result.documents:
                continue
            
            # Filter out non-educational pages
            def is_educational_doc(doc):
                page = doc.metadata.get("page_number", doc.metadata.get("slide_number", 0))
                content = doc.page_content.lower()
                skip_patterns = [
                    'some interesting examples',
                    'identify the context of the image',
                    'response from chatgpt',
                    'response from claude',
                    'response from gemini',
                    'prompt used',
                    'quote attributed',
                    'j.r.r. tolkien',
                    'gandalf',
                    'campus tower',
                    'university landmarks',
                    'chess',
                    'alan turing',
                    'mygreatlearning',
                    'simulation or modeling',
                    'simulation of physical',
                    'image, the intended audience',
                    'research question being addressed',
                    'challenges to be considered',
                    'essence of the quote',
                    'time is a precious',
                    'paralyzed by our circumstances',
                    'well-known line from j.r.r.',
                    'fellowship of the ring',
                    'lord of the rings',
                ]
                for pattern in skip_patterns:
                    if pattern in content:
                        return False
                return True
            
            filtered_docs = [d for d in result.documents if is_educational_doc(d)]
            all_filtered_docs.extend(filtered_docs)
        
        if not all_filtered_docs:
            return []
        
        # Deduplicate by chunk_id
        seen_chunks = set()
        unique_docs = []
        for d in all_filtered_docs:
            chunk_id = d.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_docs.append(d)
        
        # Group chunks by page for diverse coverage
        chunk_groups = self._group_chunks_by_page(unique_docs)
        
        fallback_questions = []
        for page_chunks in chunk_groups:
            if len(fallback_questions) >= needed:
                break
            
            # Use answer-first generation
            gen_result = self.generator.generate_grounded_question_answer_first(
                retrieved_documents=page_chunks[:self.config.max_context_chunks],
                topic_focus=self.config.topic_focus,
                difficulty=self.config.difficulty,
                user_id=user_id,
                document_id=document_id
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