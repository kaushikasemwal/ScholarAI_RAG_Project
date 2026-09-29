"""
generator.py — Grounded Quiz Generation for RAG
================================================
Generates quiz questions grounded in retrieved evidence using FLAN-T5.
Uses a two-step approach with comprehensive validation:
1. FLAN-T5 generates the question stem (what it does well)
2. Code programmatically constructs answer/options/explanation with validation
"""

import logging
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence
except ImportError:
    PromptTemplate = None
    RunnableSequence = None

from ..models import get_t5, get_sbert
from ..config import get_settings
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
    temperature: float = 0.3  # Lower for more focused generation
    max_output_tokens: int = 100  # Reduced: we only need question stem
    max_new_tokens: int = 64  # More precise control over output length
    num_beams: int = 2  # Reduced from 4: good quality/speed balance
    do_sample: bool = False  # Deterministic generation
    top_p: float = 0.9
    no_repeat_ngram_size: int = 3
    context_format: Optional[ContextFormatConfig] = None
    prompt_template: str = "t5_grounded_mcq"
    difficulty: str = "medium"
    topic_focus: str = "the key concepts"
    
    # Phase 3: Retry settings
    max_retries: int = 2  # Reduced from 3
    
    # Phase 3: Validation thresholds (loaded from settings if not overridden)
    answer_min_support_score: float = 0.0
    distractor_min_similarity: float = 0.0
    distractor_max_similarity: float = 0.0
    distractor_max_overlap: float = 0.0
    question_min_length: int = 0
    question_min_length_answer_first: int = 0
    reject_generic: bool = False
    reject_artifacts: bool = False
    
    def __post_init__(self):
        """Load validation thresholds from settings if not explicitly set."""
        settings = get_settings()
        if self.answer_min_support_score == 0.0:
            self.answer_min_support_score = settings.RAG_ANSWER_MIN_SUPPORT_SCORE
        if self.distractor_min_similarity == 0.0:
            self.distractor_min_similarity = settings.RAG_DISTRACTOR_MIN_SIMILARITY
        if self.distractor_max_similarity == 0.0:
            self.distractor_max_similarity = settings.RAG_DISTRACTOR_MAX_SIMILARITY
        if self.distractor_max_overlap == 0.0:
            self.distractor_max_overlap = settings.RAG_DISTRACTOR_MAX_OVERLAP
        if self.question_min_length == 0:
            self.question_min_length = settings.RAG_QUESTION_MIN_LENGTH
        if self.question_min_length_answer_first == 0:
            self.question_min_length_answer_first = settings.RAG_QUESTION_MIN_LENGTH_ANSWER_FIRST
        if not self.reject_generic:
            self.reject_generic = settings.RAG_REJECT_GENERIC
        if not self.reject_artifacts:
            self.reject_artifacts = settings.RAG_REJECT_ARTIFACTS


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
    validation_details: Dict[str, Any] = field(default_factory=dict)


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
            max_new_tokens=self.config.max_new_tokens,
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
    
    def generate_question_only(
        self,
        context: str,
        topic_focus: str,
        difficulty: str = "medium"
    ) -> str:
        """
        Generate only a question stem using FLAN-T5.
        This is what FLAN-T5 does well - generating a single question.
        """
        prompt = (
            "Based on the text below, write ONE question that tests understanding of a key concept. "
            "Output ONLY the question, nothing else. End with a question mark.\n\n"
            f"Text: {context}\n\nQuestion: "
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            early_stopping=True,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample,
            top_p=self.config.top_p,
        )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Clean up any unwanted additions (Options:, A), B), etc.)
        question = re.split(r'\s*(?:Options?|[A-D][):])', question, flags=re.IGNORECASE)[0]
        question = re.split(r'\s*(?:Answer|Explanation):', question, flags=re.IGNORECASE)[0]
        
        question = question.strip()
        
        if not question or len(question.split()) < 4:
            return None
        if not question.endswith("?"):
            question += "?"
        return question


# =====================================================================
# PHASE 3: VALIDATION HELPER FUNCTIONS
# =====================================================================

def _extract_keywords(text: str, n: int = 10) -> list[str]:
    """Extract top TF-IDF keywords from text for distractor generation."""
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


def _is_generic_question(question: str) -> bool:
    """Check if a question is too generic/template-like."""
    generic_patterns = [
        r"which of the following",
        r"what is the main",
        r"which statement",
        r"according to the passage",
        r"based on the text",
        r"the passage states",
        r"the document mentions",
        r"as mentioned in",
        r"as stated in",
    ]
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in generic_patterns)


def _has_artifacts(question: str) -> bool:
    """Check if question contains PDF extraction artifacts."""
    artifact_patterns = [
        r"\.\.\.",  # Truncation
        r"---",     # Separators
        r"\[\d+\]", # Reference numbers
        r"page \d+",
        r"slide \d+",
        r"figure \d+",
        r"table \d+",
        r"source \d+",
        r"chunk \d+",
        r"https?://",
        r"www\.",
        r"\.pdf",
        r"\.pptx?",
        r"copyright",
        r"confidential",
        r"draft",
    ]
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in artifact_patterns)


def _is_meaningful_question(question: str, config: GeneratorConfig, answer_first: bool = False) -> Tuple[bool, str]:
    """Validate question quality."""
    if not question or not question.strip():
        return False, "Empty question"
    
    words = question.split()
    min_length = config.question_min_length_answer_first if answer_first else config.question_min_length
    if len(words) < min_length:
        return False, f"Question too short ({len(words)} words, min {min_length})"
    
    if not question.endswith("?"):
        return False, "Question does not end with question mark"
    
    # Check for malformed endings
    if question.rstrip().endswith(('---?', '...?', '..?')):
        return False, "Question has malformed ending"
    
    if config.reject_generic and _is_generic_question(question):
        return False, "Question is too generic/template-like"
    
    if config.reject_artifacts and _has_artifacts(question):
        return False, "Question contains PDF extraction artifacts"
    
    # Check for PDF artifact patterns in question
    if _is_pdf_artifact_line(question):
        return False, "Question contains PDF artifact patterns"
    
    return True, ""


def _clean_context_for_extraction(context: str) -> str:
    """Remove metadata markers from context, keeping only actual content with sentence boundaries."""
    lines = context.split('\n')
    content_lines = []
    for line in lines:
        line = line.strip()
        # Skip metadata lines
        if (line.startswith('[Source') or line.startswith('Page/Slide:') or 
            line.startswith('Chunk ID:') or line.startswith('Relevance:') or 
            line.startswith('Source:') or line.startswith('Content:')):
            continue
        if line == '---':
            continue
        # Skip PDF artifacts
        if _is_pdf_artifact_line(line):
            continue
        if line:
            content_lines.append(line)
    # Join with newlines to preserve sentence boundaries for tokenization
    return '\n'.join(content_lines)


def _is_pdf_artifact_line(line: str) -> bool:
    """Check if a line is a PDF extraction artifact."""
    line_lower = line.lower()
    artifact_patterns = [
        r'^\s*[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af]',  # Bullet points
        r'^\s*\d+\s*$',  # Page numbers
        r'^\s*page\s+\d+',  # Page labels
        r'^\s*slide\s+\d+',  # Slide labels
        r'^\s*figure\s+\d+',  # Figure labels
        r'^\s*table\s+\d+',  # Table labels
        r'copyright',
        r'confidential',
        r'draft',
        r'^\s*www\.',
        r'^\s*https?://',
        r'\.pdf\s*$',
        r'\.pptx?\s*$',
        r'^[\-\=\_]{3,}\s*$',  # Separator lines
        r'^[\*\#]{3,}\s*$',  # Markdown headers
    ]
    import re
    return any(re.search(pattern, line_lower) for pattern in artifact_patterns)


def _clean_answer_candidate_text(text: str) -> str:
    """
    Clean answer candidate text by removing PDF artifacts and normalizing.
    
    Removes:
    - Leading bullet characters (•, ?, .?, etc.)
    - Trailing incomplete constructions (by ., to ., etc.)
    - Excessive whitespace
    """
    if not text:
        return text
    
    # Strip leading bullet points and symbols (including ? and .? artifacts)
    text = re.sub(r'^[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af\-\*\•\u2022\u25aa\u25cf\u25cb\?\.]+\s*', '', text)
    
    # Strip trailing incomplete prepositional phrases
    # Pattern: word + space + punctuation at end (e.g., "by .", "to .", "for .", "in .", "on .", "with .", "from .", "as .", "is .", "was .", "are .", "were .", "has .", "have .", "had .")
    incomplete_endings = [
        r'\s+by\s*\.\s*$',
        r'\s+to\s*\.\s*$',
        r'\s+for\s*\.\s*$',
        r'\s+in\s*\.\s*$',
        r'\s+on\s*\.\s*$',
        r'\s+with\s*\.\s*$',
        r'\s+from\s*\.\s*$',
        r'\s+as\s*\.\s*$',
        r'\s+is\s*\.\s*$',
        r'\s+was\s*\.\s*$',
        r'\s+are\s*\.\s*$',
        r'\s+were\s*\.\s*$',
        r'\s+has\s*\.\s*$',
        r'\s+have\s*\.\s*$',
        r'\s+had\s*\.\s*$',
        r'\s+by\s*:\s*$',
        r'\s+to\s*:\s*$',
        r'\s+for\s*:\s*$',
        r'\s+in\s*:\s*$',
        r'\s+on\s*:\s*$',
        r'\s+with\s*:\s*$',
        r'\s+from\s*:\s*$',
        r'\s+as\s*:\s*$',
        r'\s+is\s*:\s*$',
        r'\s+was\s*:\s*$',
        r'\s+are\s*:\s*$',
        r'\s+were\s*:\s*$',
        r'\s+has\s*:\s*$',
        r'\s+have\s*:\s*$',
        r'\s+had\s*:\s*$',
        r'\s+by\s*;\s*$',
        r'\s+to\s*;\s*$',
        r'\s+for\s*;\s*$',
        r'\s+in\s*;\s*$',
        r'\s+on\s*;\s*$',
        r'\s+with\s*;\s*$',
        r'\s+from\s*;\s*$',
        r'\s+as\s*;\s*$',
        r'\s+is\s*;\s*$',
        r'\s+was\s*;\s*$',
        r'\s+are\s*;\s*$',
        r'\s+were\s*;\s*$',
        r'\s+has\s*;\s*$',
        r'\s+have\s*;\s*$',
        r'\s+had\s*;\s*$',
        r'\s+by\s*,\s*$',
        r'\s+to\s*,\s*$',
        r'\s+for\s*,\s*$',
        r'\s+in\s*,\s*$',
        r'\s+on\s*,\s*$',
        r'\s+with\s*,\s*$',
        r'\s+from\s*,\s*$',
        r'\s+as\s*,\s*$',
        r'\s+is\s*,\s*$',
        r'\s+was\s*,\s*$',
        r'\s+are\s*,\s*$',
        r'\s+were\s*,\s*$',
        r'\s+has\s*,\s*$',
        r'\s+have\s*,\s*$',
        r'\s+had\s*,\s*$',
    ]
    for pattern in incomplete_endings:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def _is_incomplete_statement(text: str) -> bool:
    """Check if a statement ends with an incomplete construction."""
    text_lower = text.lower().strip()
    
    # Check for trailing prepositions/conjunctions that indicate incompleteness
    incomplete_endings = [
        ' by', ' to', ' for', ' in', ' on', ' with', ' from', ' as',
        ' is', ' was', ' are', ' were', ' has', ' have', ' had',
    ]
    
    # Check if ends with these words followed by punctuation or end
    for ending in incomplete_endings:
        if text_lower.endswith(ending) or text_lower.endswith(ending + '.') or text_lower.endswith(ending + ',') or text_lower.endswith(ending + ';') or text_lower.endswith(ending + ':'):
            return True
    
    return False


def _extract_answer_candidates(context: str, min_words: int = 5, max_words: int = 30) -> List[Dict[str, Any]]:
    """Extract and filter sentences from cleaned context."""
    import nltk
    sentences = nltk.sent_tokenize(context)
    
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Skip very short sentences
        if len(sent.split()) < 5:
            continue
        # Skip metadata-like sentences
        if sent.startswith('[') or 'Source' in sent or 'Page/Slide' in sent:
            continue
        # Skip lines that are mostly punctuation/numbers
        alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
        if alpha_ratio < 0.3:
            continue
        # Skip sentences with PDF artifacts
        if _is_pdf_artifact_line(sent):
            continue
        # Skip sentences that are mostly bullets/symbols
        if sent.startswith('\uf071') or sent.startswith('\uf0b7'):
            continue
        valid_sentences.append(sent)
    
    return valid_sentences


def _is_quality_fact_sentence(sent: str) -> bool:
    """Check if a sentence is a quality factual statement suitable for QA generation."""
    sent_lower = sent.lower().strip()
    
    # Reject questions (they end with ?)
    if sent_lower.endswith('?'):
        return False
    
    # Reject image-related content
    image_patterns = [
        'some interesting examples',
        'identify the context of the image',
        'response from chatgpt',
        'response from claude',
        'response from gemini',
        'prompt used',
        'the image has been created',
        'the image appears to',
        'the image contains',
        'image shows',
        'visual shows',
        'generated image',
        'futuristic classroom',
        'false non-match rate',
        'biometric systems',
        'false non-match',
        'intra-user variability',
        'sensor accuracy',
        'repeated attempts',
    ]
    for pattern in image_patterns:
        if pattern in sent_lower:
            return False
    
    # Reject fragmentary sentences (starting with lowercase, fragments)
    if sent_lower.startswith(('and ', 'or ', 'but ', 'the ', 'a ', 'an ', '• ', '- ', '* ')):
        # Allow if it's a complete sentence with subject and verb
        pass
    
    # Reject sentences that are mostly punctuation/numbers
    alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
    if alpha_ratio < 0.5:
        return False
    
    # Reject very short fragments
    if len(sent.split()) < 6:
        return False
    
    # Reject URLs
    if 'http' in sent_lower or 'www.' in sent_lower:
        return False
    
    # Reject reference citations
    if 'et al.' in sent_lower or 'comprehensive review' in sent_lower:
        return False
    
    # Reject table of contents style entries
    toc_patterns = [
        r'^unit \d+',
        r'^chapter \d+',
        r'^section \d+',
        r'^lesson \d+',
        r'^module \d+',
        r'^\d+\.\s',
    ]
    import re
    for pattern in toc_patterns:
        if re.search(pattern, sent_lower):
            return False
    
    # Reject headings that are just titles
    if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', sent) and len(sent.split()) < 10:
        # Likely a heading like "Future Trends and Developments"
        return False
    
    # Reject headings with "Future Trends", "Challenges to be", etc.
    heading_patterns = [
        'future trends',
        'challenges to be',
        'advancements in',
        'evolution of',
        'introduction to',
    ]
    for pattern in heading_patterns:
        if pattern in sent_lower:
            return False
    
    # Must have at least one verb-like word (indicating a complete statement)
    verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'will', 'would', 'should', 'could', 'learn', 'create', 'generate', 'produce', 'use', 'model', 'train', 'predict', 'classify', 'recognize', 'understand', 'process', 'analyze', 'attempt', 'enable', 'provide', 'support', 'include', 'contain', 'consist', 'represent', 'describe', 'define', 'explain', 'illustrate', 'demonstrate', 'show', 'indicate', 'suggest', 'imply', 'mean', 'refer', 'aim', 'aims', 'aimed', 'aiming', 'improve', 'improves', 'improved', 'improving', 'enhance', 'enhances', 'enhanced', 'enhancing', 'shape', 'shapes', 'shaped', 'shaping', 'enable', 'enables', 'enabled', 'enabling', 'include', 'includes', 'included', 'including', 'integrate', 'integrates', 'integrated', 'integrating', 'drive', 'drives', 'drove', 'driving', 'transform', 'transforms', 'transformed', 'transforming', 'revolutionize', 'revolutionizes', 'revolutionized', 'revolutionizing', 'disrupt', 'disrupts', 'disrupted', 'disrupting', 'automate', 'automates', 'automated', 'automating', 'optimize', 'optimizes', 'optimized', 'optimizing', 'personalize', 'personalizes', 'personalized', 'personalizing', 'enhance', 'enhances', 'enhanced', 'enhancing']
    has_verb = any(f' {v} ' in f' {sent_lower} ' for v in verb_indicators)
    if not has_verb:
        return False
    
    return True


def _extract_answer_candidates(context: str, min_words: int = 5, max_words: int = 30) -> List[Dict[str, Any]]:
    """
    Extract candidate factual statements from context that can serve as answers.
    
    This is the ANSWER-FIRST approach: identify facts in the context first,
    then generate questions about them.
    
    Returns:
        List of dicts with keys: 'text', 'sentence', 'page_number', 'chunk_id', 'source_index'
    """
    import nltk
    from langchain_core.documents import Document
    
    candidates = []
    
    # Helper to check if a sentence is from educational content
    def is_educational_sentence(sent: str) -> bool:
        sent_lower = sent.lower().strip()
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
            'adjust the lighting',
            'immersive atmosphere',
            'star-like patterns',
            'dynamic effects',
            'color transitions',
        ]
        for pattern in skip_patterns:
            if pattern in sent_lower:
                return False
        return True
    
    # Handle both raw context string and list of documents
    if isinstance(context, str):
        # Parse the formatted context to get individual source chunks
        # Format: [Source N]\nPage/Slide: X\nContent:\n<content>\n\n---\n\n
        sources = re.split(r'\n\n---\n\n', context)
        for source_idx, source in enumerate(sources):
            if not source.strip():
                continue
            # Extract page/slide and content
            page_match = re.search(r'Page/Slide:\s*(\d+)', source)
            page_num = int(page_match.group(1)) if page_match else source_idx + 1
            
            content_match = re.search(r'Content:\n(.+)', source, re.DOTALL)
            content = content_match.group(1).strip() if content_match else source
            
            # Extract sentences from this source with improved tokenization
            # Preprocess content to add proper sentence boundaries for bullet points and headings
            processed_content = content
            # First, join line breaks within sentences (lowercase to lowercase)
            processed_content = re.sub(r'(?<=[a-z])\s*\n(?=[a-z])', ' ', processed_content)
            # Join line breaks after punctuation followed by bullet/heading
            processed_content = re.sub(r'(?<=[.!?])\s*\n(?=[A-Z•])', ' ', processed_content)
# Add explicit sentence boundary before bullet points (for NLTK)
            # Only add period if not already followed by period (avoid double periods)
            processed_content = re.sub(r'(?<=[.!?])(?<![.])\s+(?=•)', '. • ', processed_content)
            processed_content = re.sub(r'(?<=\n)\s*(?=•)', '. • ', processed_content)
            # Now add period after bullet points that DON'T end with punctuation
            # (bullet points that already end with punctuation keep their punctuation)
            processed_content = re.sub(r'(•\s*[^\n]*[.!?])(\s*•|\s*[A-Z]\.|\s*\n\n|\s*$)', r'\1\2', processed_content)
            # Add period after headings like "A. Title" followed by content
            processed_content = re.sub(r'([A-Z]\.\s*[^\n]*\n)(?![•\s])', r'\1. ', processed_content)
            # Add period after "•" bullet points that don't end with punctuation
            processed_content = re.sub(r'(•\s*[^\n]*[^.!?])(\s*•|\s*\n|\s*$)', r'\1.\2', processed_content)
            # Split on double newlines
            processed_content = re.sub(r'\n\n+', '.\n\n', processed_content)
            
            # Now tokenize with NLTK
            sentences = nltk.sent_tokenize(processed_content)
            for sent in sentences:
                sent = sent.strip()
                # Strip leading bullet points and symbols
                sent = re.sub(r'^[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af\-\*\•\u2022\u25aa\u25cf\u25cb]+\\s*', '', sent)
                if len(sent.split()) < min_words or len(sent.split()) > max_words:
                    continue
                if _is_pdf_artifact_line(sent):
                    continue
                if sent.startswith('[') or 'Source' in sent or 'Page/Slide' in sent:
                    continue
                if not is_educational_sentence(sent):
                    continue
                if not _is_quality_fact_sentence(sent):
                    continue
                alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
                if alpha_ratio < 0.4:
                    continue
                
                # Clean the candidate text
                cleaned_sent = _clean_answer_candidate_text(sent)
                
                # Skip if cleaning resulted in too short text
                if len(cleaned_sent.split()) < min_words:
                    continue
                
                # Skip incomplete statements
                if _is_incomplete_statement(cleaned_sent):
                    continue
                
                candidates.append({
                    'text': cleaned_sent,
                    'sentence': cleaned_sent,
                    'page_number': page_num,
                    'chunk_id': f'source_{source_idx}',
                    'source_index': source_idx,
                })
    else:
        # Assume it's a list of LangChain Documents
        for doc_idx, doc in enumerate(context):
            if not hasattr(doc, 'page_content') or not doc.page_content.strip():
                continue
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            page_num = metadata.get('page_number', metadata.get('slide_number', doc_idx + 1))
            chunk_id = metadata.get('chunk_id', f'doc_{doc_idx}')
            
# Preprocess content for better sentence tokenization
            content = doc.page_content
            processed_content = content
            # First, join line breaks within sentences (lowercase to lowercase)
            processed_content = re.sub(r'(?<=[a-z])\s*\n(?=[a-z])', ' ', processed_content)
            # Join line breaks after punctuation followed by bullet/heading
            processed_content = re.sub(r'(?<=[.!?])\s*\n(?=[A-Z•])', ' ', processed_content)
# Add explicit sentence boundary before bullet points (for NLTK)
            def _add_bullet_boundary(match):
                # match.group(1) is the punctuation (. ! or ?)
                # match.group(0) is punctuation + spaces
                punct = match.group(1)
                if punct == '.':
                    # Already a period, just ensure space and bullet
                    return '. • '
                elif punct in '!?':
                    # Replace ! or ? with period + space + bullet
                    return '. • '
                return '. • '
            
            processed_content = re.sub(r'([.!?])\s+(?=•)', _add_bullet_boundary, processed_content)
            processed_content = re.sub(r'(?<=\n)\s*(?=•)', '. • ', processed_content)
            # Now add period after bullet points that DON'T end with punctuation
            # (bullet points that already end with punctuation keep their punctuation)
            processed_content = re.sub(r'(•\s*[^\n]*[.!?])(\s*•|\s*[A-Z]\.|\s*\n\n|\s*$)', r'\1\2', processed_content)
            # Add period after headings like "A. Title" followed by content
            processed_content = re.sub(r'([A-Z]\.\s*[^\n]*\n)(?![•\s])', r'\1. ', processed_content)
            # Add period after "•" bullet points that don't end with punctuation
            processed_content = re.sub(r'(•\s*[^\n]*[^.!?])(\s*•|\s*\n|\s*$)', r'\1.\2', processed_content)
            # Split on double newlines
            processed_content = re.sub(r'\n\n+', '.\n\n', processed_content)
            
            sentences = nltk.sent_tokenize(processed_content)
            for sent in sentences:
                sent = sent.strip()
                # Strip leading bullet points and symbols
                sent = re.sub(r'^[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af\-\*\•\u2022\u25aa\u25cf\u25cb]+\s*', '', sent)
                if len(sent.split()) < min_words or len(sent.split()) > max_words:
                    continue
                if _is_pdf_artifact_line(sent):
                    continue
                if sent.startswith('[') or 'Source' in sent or 'Page/Slide' in sent:
                    continue
                if not is_educational_sentence(sent):
                    continue
                if not _is_quality_fact_sentence(sent):
                    continue
                alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
                if alpha_ratio < 0.4:
                    continue
                
                # Clean the candidate text
                cleaned_sent = _clean_answer_candidate_text(sent)
                
                # Skip if cleaning resulted in too short text
                if len(cleaned_sent.split()) < min_words:
                    continue
                
                # Skip incomplete statements
                if _is_incomplete_statement(cleaned_sent):
                    continue
                
                candidates.append({
                    'text': cleaned_sent,
                    'sentence': cleaned_sent,
                    'page_number': page_num,
                    'chunk_id': chunk_id,
                    'source_index': doc_idx,
                })
    
    return candidates


def _generate_question_for_answer(answer_candidate: Dict[str, Any], t5_generator: 'FLANT5Generator', topic_focus: str, difficulty: str) -> Optional[str]:
    """
    Generate a question specifically about the given answer candidate.
    
    This is more reliable than asking FLAN-T5 to invent a question from context.
    """
    answer_text = answer_candidate['text']
    
    # Create a focused prompt asking FLAN-T5 to generate a question 
    # for which the answer_text is the correct answer
    prompt = (
        f"Based on the statement below, write ONE clear, detailed question that tests understanding of this fact. "
        f"The question should be specific and complete. The answer to your question should be directly supported by the statement.\n\n"
        f"Statement: {answer_text}\n\n"
        f"Question: "
    )
    
    inputs = t5_generator.tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    outputs = t5_generator.model.generate(
        inputs["input_ids"],
        max_new_tokens=t5_generator.config.max_new_tokens,
        num_beams=t5_generator.config.num_beams,
        early_stopping=True,
        no_repeat_ngram_size=t5_generator.config.no_repeat_ngram_size,
        temperature=t5_generator.config.temperature,
        do_sample=t5_generator.config.do_sample,
        top_p=t5_generator.config.top_p,
    )
    
    question = t5_generator.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Clean up any unwanted additions
    question = re.split(r'\s*(?:Options?|[A-D][):])', question, flags=re.IGNORECASE)[0]
    question = re.split(r'\s*(?:Answer|Explanation):', question, flags=re.IGNORECASE)[0]
    
    question = question.strip()
    
    if not question or len(question.split()) < 6:
        return None
    if not question.endswith("?"):
        question += "?"
    return question


def _extract_answer_from_context(question: str, context: str, config: GeneratorConfig) -> Optional[str]:
    """
    Extract a specific answer phrase from context that answers the question.
    Uses sentence-level analysis to find the most relevant sentence.
    """
    import nltk
    
    valid_sentences = _filter_clean_sentences(context)
    
    if not valid_sentences:
        return None
    
    # Extract key terms from question (excluding stop words)
    question_words = set(re.findall(r'\b\w{4,}\b', question.lower()))
    stop_words = {'what', 'how', 'why', 'when', 'where', 'which', 'does', 'is', 'are', 'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that', 'about', 'into', 'your', 'based', 'using', 'create', 'learn', 'model', 'generative', 'discriminative', 'according', 'passage', 'document', 'text', 'context'}
    question_terms = question_words - stop_words
    
    # Score sentences by overlap with question terms
    best_sentence = None
    best_score = 0
    
    for sent in valid_sentences:
        sent_words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
        overlap = len(question_terms & sent_words)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sent
    
    if best_sentence and best_score > 0:
        # Extract a concise answer phrase from the best sentence
        if question.lower().startswith('how'):
            for pattern in [r'by\s+([^.]+)', r'through\s+([^.]+)', r'via\s+([^.]+)', r'learning\s+([^.]+)', r'attempts?\s+to\s+([^.]+)']:
                match = re.search(pattern, best_sentence, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if _is_valid_answer_candidate(candidate):
                        return candidate.capitalize()
        
        if question.lower().startswith('what'):
            for pattern in [r'is\s+([^.]+)', r'are\s+([^.]+)', r'attempts?\s+to\s+([^.]+)', r'learns?\s+([^.]+)']:
                match = re.search(pattern, best_sentence, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if _is_valid_answer_candidate(candidate):
                        return candidate.capitalize()
        
        if question.lower().startswith('why'):
            for pattern in [r'because\s+([^.]+)', r'since\s+([^.]+)', r'due to\s+([^.]+)']:
                match = re.search(pattern, best_sentence, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if _is_valid_answer_candidate(candidate):
                        return candidate.capitalize()
        
        # Default: use first clause of the sentence (up to comma or period)
        clause = re.split(r'[,;.]', best_sentence)[0].strip()
        if len(clause.split()) > 3 and _is_valid_answer_candidate(clause):
            return clause
    
    # Fallback: try to find any valid sentence that contains question terms
    for sent in valid_sentences:
        sent_words = set(re.findall(r'\b\w{3,}\b', sent.lower()))
        overlap = len(question_terms & sent_words)
        if overlap >= 2:
            clause = re.split(r'[,;.]', sent)[0].strip()
            if len(clause.split()) > 3 and _is_valid_answer_candidate(clause):
                return clause.capitalize()
    
    return None


def _is_valid_answer_candidate(candidate: str) -> bool:
    """Check if a candidate answer is valid (not an artifact, header, or fragment)."""
    import re
    if not candidate or len(candidate.strip()) < 3:
        return False
    
    candidate_lower = candidate.lower().strip()
    
    # Strip leading bullet points and symbols for validation
    # The PDF extraction may include bullet characters that we should ignore for validation
    candidate_for_validation = re.sub(r'^[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af\-\*\•\u2022\u25aa\u25cf\u25cb]+\s*', '', candidate_lower)
    
    # Reject if it's a URL
    if 'http' in candidate_for_validation or 'www.' in candidate_for_validation:
        return False
    
    # Reject if it's mostly numbers/punctuation
    alpha_ratio = sum(c.isalpha() for c in candidate_for_validation) / max(len(candidate_for_validation), 1)
    if alpha_ratio < 0.4:
        return False
    
    # Reject common header patterns
    header_patterns = [
        r'^(introduction|conclusion|summary|overview|background)\s*:?\s*$',
        r'^(chapter|section|module|unit|lesson)\s+\d+',
        r'^(page|slide|figure|table)\s+\d+',
    ]
    if any(re.search(pattern, candidate_for_validation) for pattern in header_patterns):
        return False
    
    # Reject if it's too short (allow single-word proper nouns like "Transformer", "GPT-2", "BERT")
    if len(candidate_for_validation.split()) < 1:
        return False
    
    # Reject if it's a question fragment
    if candidate_for_validation.startswith(('what ', 'how ', 'why ', 'when ', 'where ', 'which ')):
        return False
    
    return True


def _validate_answer_support(answer: str, context: str, config: GeneratorConfig) -> Tuple[bool, float, str]:
    """
    Validate that the answer is supported by the context.
    Returns (is_valid, support_score, evidence_sentence).
    """
    import nltk
    from ..models import get_sbert
    
    if not answer or answer == "Refer to the document":
        return False, 0.0, ""
    
    # First check if answer itself is a valid candidate
    if not _is_valid_answer_candidate(answer):
        return False, 0.0, ""
    
    valid_sentences = _filter_clean_sentences(context)
    if not valid_sentences:
        return False, 0.0, ""
    
    # First check: exact textual match
    for sent in valid_sentences:
        if answer.lower() in sent.lower():
            return True, 1.0, sent.strip()
    
    # Second check: semantic similarity using SBERT
    try:
        sbert = get_sbert()
        answer_emb = sbert.encode([answer], convert_to_numpy=True, normalize_embeddings=True)
        
        best_score = 0.0
        best_sentence = ""
        
        for sent in valid_sentences:
            sent_emb = sbert.encode([sent], convert_to_numpy=True, normalize_embeddings=True)
            cos_sim = float(answer_emb[0] @ sent_emb[0])
            if cos_sim > best_score:
                best_score = cos_sim
                best_sentence = sent.strip()
        
        is_valid = best_score >= config.answer_min_support_score
        return is_valid, best_score, best_sentence
    except Exception as e:
        log.warning(f"SBERT validation failed: {e}")
        # Fallback: word overlap
        answer_words = set(answer.lower().split())
        for sent in valid_sentences:
            sent_words = set(sent.lower().split())
            overlap = len(answer_words & sent_words) / max(len(answer_words), 1)
            if overlap > 0.5:
                return True, overlap, sent.strip()
        return False, 0.0, ""


def _generate_distractors(correct: str, context: str, config: GeneratorConfig, n: int = 3) -> list[str]:
    """
    Generate plausible wrong answers from the context with validation.
    """
    import nltk
    from ..models import get_sbert
    
    valid_sentences = _filter_clean_sentences(context)
    candidate_phrases = []
    
    for sent in valid_sentences:
        # Extract meaningful sub-phrases (5-15 words)
        clauses = re.split(r'[,;]| (?:and|or|but|while|whereas) ', sent)
        for clause in clauses:
            clause = clause.strip()
            words = clause.split()
            if 3 <= len(words) <= 12:
                clause = re.sub(r'^[^\w]+|[^\w]+$', '', clause)
                if clause and clause.lower() != correct.lower():
                    if _is_valid_distractor_candidate(clause, correct):
                        candidate_phrases.append(clause)
        
        # Also add the full sentence if it's not too long
        if 5 <= len(sent.split()) <= 15:
            if _is_valid_distractor_candidate(sent.strip(), correct):
                candidate_phrases.append(sent.strip())
    
    # Filter out phrases too similar to correct answer, URLs, headers, fragments
    filtered_phrases = []
    for phrase in candidate_phrases:
        if phrase.lower() == correct.lower() or correct.lower() in phrase.lower():
            continue
        # Check word overlap with correct answer
        correct_words = set(correct.lower().split())
        phrase_words = set(phrase.lower().split())
        overlap = len(correct_words & phrase_words) / max(len(correct_words), 1)
        if overlap >= config.distractor_max_overlap:
            continue
        filtered_phrases.append(phrase)
    
    # Use SBERT to score distractors
    try:
        sbert = get_sbert()
        correct_emb = sbert.encode([correct], convert_to_numpy=True, normalize_embeddings=True)
        
        scored_distractors = []
        for phrase in filtered_phrases:
            phrase_emb = sbert.encode([phrase], convert_to_numpy=True, normalize_embeddings=True)
            cos_sim = float(correct_emb[0] @ phrase_emb[0])
            # Good distractor: related but not too similar
            if config.distractor_min_similarity <= cos_sim <= config.distractor_max_similarity:
                scored_distractors.append((cos_sim, phrase))
        
        scored_distractors.sort(key=lambda x: abs(x[0] - 0.45))  # Target ~0.45 similarity
        distractors = [p for _, p in scored_distractors[:n]]
    except Exception as e:
        log.warning(f"SBERT distractor scoring failed: {e}")
        random.shuffle(filtered_phrases)
        distractors = filtered_phrases[:n]
    
    # Pad with generic placeholders if not enough distractors
    placeholders = [
        "None of the above",
        "All of the above",
        "Cannot be determined from the text",
        "Not mentioned in the context"
    ]
    while len(distractors) < n:
        p = placeholders.pop(0) if placeholders else f"Option {len(distractors)+1}"
        distractors.append(p)
    
    return distractors[:n]


def _is_valid_distractor_candidate(candidate: str, correct_answer: str) -> bool:
    """Check if a candidate distractor is valid."""
    import re
    if not candidate or len(candidate.strip()) < 3:
        return False
    
    candidate_lower = candidate.lower().strip()
    
    # Reject if it's a PDF artifact
    if _is_pdf_artifact_line(candidate):
        return False
    
    # Strip leading bullet points and symbols for validation
    candidate_for_validation = re.sub(r'^[\uf071\uf0b7\uf0a7\uf0a8\uf0a9\uf0aa\uf0ab\uf0ac\uf0ad\uf0ae\uf0af\-\*\•\u2022\u25aa\u25cf\u25cb]+\s*', '', candidate_lower)
    
    # Reject if it's a PDF artifact
    if _is_pdf_artifact_line(candidate):
        return False
    
    # Reject if it's a URL
    if 'http' in candidate_for_validation or 'www.' in candidate_for_validation:
        return False
    
    # Reject if it's mostly numbers/punctuation
    alpha_ratio = sum(c.isalpha() for c in candidate_for_validation) / max(len(candidate_for_validation), 1)
    if alpha_ratio < 0.4:
        return False
    
    # Reject if it starts with bullet/symbol (after stripping)
    if candidate_for_validation.startswith(('\uf071', '\uf0b7', '\uf0a7', '-', '*', '•')):
        return False
    
    # Reject common header patterns
    header_patterns = [
        r'^(introduction|conclusion|summary|overview|background)\s*:?\s*$',
        r'^(chapter|section|module|unit|lesson)\s+\d+',
        r'^(page|slide|figure|table)\s+\d+',
    ]
    if any(re.search(pattern, candidate_for_validation) for pattern in header_patterns):
        return False
    
    # Reject if it's too short
    if len(candidate_for_validation.split()) < 3:
        return False
    
    # Reject if it's a question fragment
    if candidate_for_validation.startswith(('what ', 'how ', 'why ', 'when ', 'where ', 'which ')):
        return False
    
    # Reject if too similar to correct answer (checked separately but double-check)
    correct_words = set(correct_answer.lower().split())
    candidate_words = set(candidate_for_validation.split())
    overlap = len(correct_words & candidate_words) / max(len(correct_words), 1)
    if overlap > 0.5:
        return False
    
    return True


def _generate_reasoning(question: str, correct: str, context: str) -> str:
    """Generate a brief explanation for why the correct answer is right, grounded in context."""
    import nltk
    valid_sentences = _filter_clean_sentences(context)
    for sent in valid_sentences:
        if correct.lower() in sent.lower():
            return f'"{sent.strip()}" — this directly supports the answer.'
    return f'According to the document, "{correct}" is the correct answer based on the provided context.'


def build_mcq_from_question(question: str, context: str, config: GeneratorConfig) -> GroundedQuizQuestion:
    """
    Build a complete MCQ from a question stem and context with full validation.
    Raises ValueError if validation fails.
    """
    clean_context = _clean_context_for_extraction(context)
    
    # Extract answer
    correct_answer = _extract_answer_from_context(question, clean_context, config)
    
    if not correct_answer:
        # Fallback: use keyword extraction
        keywords = _extract_keywords(clean_context, n=15)
        for kw in sorted(keywords, key=len, reverse=True):
            if len(kw) > 3 and kw.lower() in clean_context.lower():
                import nltk
                sentences = nltk.sent_tokenize(clean_context)
                for sent in sentences:
                    if kw.lower() in sent.lower() and len(sent.split()) > 5:
                        correct_answer = kw.title()
                        break
                if correct_answer:
                    break
    
    if not correct_answer or not _is_valid_answer_candidate(correct_answer):
        raise ValueError(f"Could not extract valid answer for question: {question}")
    
    # Validate answer support
    is_valid, support_score, evidence = _validate_answer_support(correct_answer, clean_context, config)
    if not is_valid:
        log.warning(f"Answer validation failed: '{correct_answer}' (score: {support_score:.3f})")
        # Try to find a better answer from evidence
        if evidence and support_score > 0.3:
            correct_answer = evidence.split('.')[0].strip()[:100]
            is_valid, support_score, evidence = _validate_answer_support(correct_answer, clean_context, config)
        
        if not is_valid:
            raise ValueError(f"Answer not supported by context: '{correct_answer}' (score: {support_score:.3f})")
    
    # Generate distractors
    distractors = _generate_distractors(correct_answer, clean_context, config, n=3)
    options = [correct_answer] + distractors
    random.shuffle(options)
    correct_idx = options.index(correct_answer)
    
    # Validate options
    if len(options) != 4:
        raise ValueError("Invalid number of options")
    if len(set(opt.lower() for opt in options)) != 4:
        raise ValueError("Duplicate options detected")
    
    # Generate explanation
    explanation = _generate_reasoning(question, correct_answer, clean_context)
    
    return GroundedQuizQuestion(
        question=question,
        options=options,
        correct_answer=correct_answer,
        explanation=explanation
    )


# =====================================================================
# GROUNDED QUIZ GENERATOR WITH RETRY LOGIC
# =====================================================================

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
        Generate a grounded quiz question from retrieved evidence with retry logic.
        
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
        
        # Retry loop for validation
        max_retries = self.config.max_retries
        last_error = None
        
        for attempt in range(max_retries):
            # Step 1: Generate question stem using FLAN-T5
            try:
                question = self.t5_generator.generate_question_only(
                    context=context,
                    topic_focus=topic_focus or self.config.topic_focus,
                    difficulty=difficulty or self.config.difficulty
                )
            except Exception as e:
                log.error(f"FLAN-T5 question generation failed: {e}")
                return GroundedQuestionResult(
                    success=False,
                    error=f"Model generation failed: {e}",
                    raw_model_output=None,
                    prompt_used="question_only_generation",
                    provenance=extract_provenance(retrieved_documents),
                    retrieval_metadata={
                        "retrieved_count": len(retrieved_documents),
                        "generation_error": str(e)
                    }
                )
            
            if not question:
                last_error = "Failed to generate question stem"
                continue
            
            # Validate question quality (two-step approach uses standard min length)
            is_valid, reason = _is_meaningful_question(question, self.config, answer_first=False)
            if not is_valid:
                last_error = f"Question validation failed: {reason}"
                log.debug(f"Attempt {attempt+1}: {last_error}")
                continue
            
            # Step 2: Build complete MCQ from question and context
            try:
                mcq = build_mcq_from_question(question, context, self.config)
            except Exception as e:
                log.error(f"MCQ building failed: {e}")
                last_error = f"MCQ construction failed: {e}"
                continue
            
            # Validate answer support
            clean_context = _clean_context_for_extraction(context)
            is_valid, support_score, evidence = _validate_answer_support(mcq.correct_answer, clean_context, self.config)
            if not is_valid:
                last_error = f"Answer validation failed (support score: {support_score:.3f})"
                log.debug(f"Attempt {attempt+1}: {last_error}")
                continue
            
            # Validate distractors
            if len(mcq.options) != 4:
                last_error = "Invalid number of options"
                continue
            
            if mcq.correct_answer not in mcq.options:
                last_error = "Correct answer not in options"
                continue
            
            # Check for duplicate options
            if len(set(opt.lower() for opt in mcq.options)) != 4:
                last_error = "Duplicate options detected"
                continue
            
            # All validations passed
            provenance = extract_provenance(retrieved_documents)
            
            retrieval_metadata = {
                "retrieved_count": len(retrieved_documents),
                "context_chunks_used": min(len(retrieved_documents), self.config.max_context_chunks),
                "topic_focus": topic_focus or self.config.topic_focus,
                "difficulty": difficulty or self.config.difficulty,
                "user_id": user_id,
                "document_id": document_id,
                "generation_method": "two_step_validated",
                "attempts": attempt + 1,
                "answer_support_score": support_score,
            }
            
            validation_details = {
                "question_valid": True,
                "answer_valid": True,
                "distractors_valid": True,
                "answer_support_score": support_score,
                "evidence_sentence": evidence[:200] if evidence else "",
            }
            
            return GroundedQuestionResult(
                success=True,
                question=mcq,
                raw_model_output=question,
                prompt_used="question_only_generation",
                provenance=provenance,
                retrieval_metadata=retrieval_metadata,
                validation_details=validation_details
            )
        
        # All retries exhausted
        return GroundedQuestionResult(
            success=False,
            error=f"Failed after {max_retries} attempts. Last error: {last_error}",
            raw_model_output=None,
            prompt_used="question_only_generation",
            provenance=extract_provenance(retrieved_documents),
            retrieval_metadata={
                "retrieved_count": len(retrieved_documents),
                "generation_error": last_error,
                "attempts": max_retries
            }
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

    def generate_grounded_question_answer_first(
        self,
        retrieved_documents: List[Any],
        topic_focus: Optional[str] = None,
        difficulty: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> GroundedQuestionResult:
        """
        Generate a grounded quiz question using the ANSWER-FIRST approach.
        
        Strategy:
        1. Extract candidate factual statements (answer candidates) from retrieved chunks
        2. Select one candidate as the correct answer
        3. Generate a question specifically about that fact using FLAN-T5
        4. Use other candidates as distractors
        5. Validate the complete MCQ
        
        This is more reliable than generating question stems first because:
        - The answer is guaranteed to be in the context
        - Distractors come from other facts in the same document
        - Questions are naturally grounded and specific
        """
        # Handle empty retrieval
        if not retrieved_documents:
            return GroundedQuestionResult(
                success=False,
                error="No retrieved documents provided",
                provenance=[],
                retrieval_metadata={"retrieved_count": 0}
            )
        
        # Format context for prompt (used for provenance and validation)
        context = format_context_for_prompt(
            retrieved_documents,
            max_chunks=self.config.max_context_chunks,
            config=self.config.context_format
        )
        
        # Check if context is effectively empty
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
        
        # Step 1: Extract answer candidates from retrieved documents
        answer_candidates = _extract_answer_candidates(retrieved_documents)
        
        if not answer_candidates:
            return GroundedQuestionResult(
                success=False,
                error="No valid answer candidates found in retrieved context",
                provenance=extract_provenance(retrieved_documents),
                retrieval_metadata={
                    "retrieved_count": len(retrieved_documents),
                    "no_answer_candidates": True
                }
            )
        
        log.debug(f"Extracted {len(answer_candidates)} answer candidates from retrieved documents")
        
        # Retry loop: try different answer candidates
        max_retries = min(self.config.max_retries, len(answer_candidates))
        last_error = None
        
        for attempt in range(max_retries):
            # Select an answer candidate (rotate through candidates for diversity)
            candidate_idx = attempt % len(answer_candidates)
            answer_candidate = answer_candidates[candidate_idx]
            correct_answer_text = answer_candidate['text']
            
            # Step 2: Generate question for this specific answer
            try:
                question = _generate_question_for_answer(
                    answer_candidate,
                    self.t5_generator,
                    topic_focus or self.config.topic_focus,
                    difficulty or self.config.difficulty
                )
            except Exception as e:
                log.error(f"FLAN-T5 question generation failed: {e}")
                last_error = f"Model generation failed: {e}"
                continue
            
            if not question:
                last_error = "Failed to generate question stem for answer candidate"
                continue
            
            # Validate question quality (answer-first uses lower min length)
            is_valid, reason = _is_meaningful_question(question, self.config, answer_first=True)
            if not is_valid:
                last_error = f"Question validation failed: {reason}"
                log.debug(f"Attempt {attempt+1}: {last_error}")
                continue
            
            # Step 3: Build MCQ using the answer-first approach
            try:
                mcq = self._build_mcq_from_answer_candidate(
                    question=question,
                    answer_candidate=answer_candidate,
                    all_candidates=answer_candidates,
                    context=context,
                    config=self.config
                )
            except Exception as e:
                log.error(f"MCQ building failed: {e}")
                last_error = f"MCQ construction failed: {e}"
                continue
            
            # Validate answer support (should pass since we extracted from context)
            clean_context = _clean_context_for_extraction(context)
            is_valid, support_score, evidence = _validate_answer_support(
                mcq.correct_answer, clean_context, self.config
            )
            if not is_valid:
                last_error = f"Answer validation failed (support score: {support_score:.3f})"
                log.debug(f"Attempt {attempt+1}: {last_error}")
                continue
            
            # Validate distractors
            if len(mcq.options) != 4:
                last_error = "Invalid number of options"
                continue
            
            if mcq.correct_answer not in mcq.options:
                last_error = "Correct answer not in options"
                continue
            
            # Check for duplicate options
            if len(set(opt.lower() for opt in mcq.options)) != 4:
                last_error = "Duplicate options detected"
                continue
            
            # All validations passed
            provenance = extract_provenance(retrieved_documents)
            
            retrieval_metadata = {
                "retrieved_count": len(retrieved_documents),
                "context_chunks_used": min(len(retrieved_documents), self.config.max_context_chunks),
                "topic_focus": topic_focus or self.config.topic_focus,
                "difficulty": difficulty or self.config.difficulty,
                "user_id": user_id,
                "document_id": document_id,
                "generation_method": "answer_first",
                "attempts": attempt + 1,
                "answer_support_score": support_score,
                "answer_candidate_source": answer_candidate.get('chunk_id'),
                "answer_candidate_page": answer_candidate.get('page_number'),
            }
            
            validation_details = {
                "question_valid": True,
                "answer_valid": True,
                "distractors_valid": True,
                "answer_support_score": support_score,
                "evidence_sentence": evidence[:200] if evidence else "",
            }
            
            return GroundedQuestionResult(
                success=True,
                question=mcq,
                raw_model_output=question,
                prompt_used="answer_first_generation",
                provenance=provenance,
                retrieval_metadata=retrieval_metadata,
                validation_details=validation_details
            )
        
        # All retries exhausted
        return GroundedQuestionResult(
            success=False,
            error=f"Failed after {max_retries} attempts. Last error: {last_error}",
            raw_model_output=None,
            prompt_used="answer_first_generation",
            provenance=extract_provenance(retrieved_documents),
            retrieval_metadata={
                "retrieved_count": len(retrieved_documents),
                "generation_error": last_error,
                "attempts": max_retries,
                "total_answer_candidates": len(answer_candidates),
            }
)

    def _extract_concise_answer(self, question: str, full_answer: str, context: str) -> str:
        """
        Extract a concise answer from the full answer sentence based on the question type.
        
        This function attempts to extract the key entity/phrase that directly answers the question,
        rather than using the entire source sentence as the answer.
        """
        import re
        
        question_lower = question.lower().strip()
        full_answer_lower = full_answer.lower().strip()
        
        # For "What is the name of..." questions, try to extract the entity name
        if question_lower.startswith('what is the name'):
            # First, try to find "called X" or "named X" patterns (highest priority for name questions)
            name_patterns = [
                r'\bcalled\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)',
                r'\bnamed\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)',
                r'\bknown\s+as\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)',
            ]
            for pattern in name_patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
            
            # For "What is the name of..." questions, try to extract the entity BEFORE "by X"
            # Pattern: "the X model by Y" or "the X by Y" -> extract "X model" or "X"
            intro_patterns = [
                r'introduction of the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\s+by\s+[A-Z]',
                r'introduction of\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\s+by\s+[A-Z]',
                r'the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\s+by\s+[A-Z]',
                r'the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+by\s+[A-Z]',
            ]
            for pattern in intro_patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    # Trim common suffixes like "model", "architecture", "system"
                    entity = re.sub(r'\s+(model|architecture|system)$', '', entity, flags=re.IGNORECASE)
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
            
            # Patterns for incomplete text (missing "by X"): "introduction of the X model", "the X model"
            intro_patterns_no_by = [
                r'introduction of the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\b',
                r'introduction of\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\b',
                r'the\s+([A-Z][a-zA-Z0-9]+(?:\s+[A-Za-z0-9]+)*)\s+model\b',
            ]
            for pattern in intro_patterns_no_by:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    # Trim common suffixes like "model", "architecture", "system"
                    entity = re.sub(r'\s+(model|architecture|system)$', '', entity, flags=re.IGNORECASE)
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
            
            # Look for capitalized proper nouns in the answer (developer/company patterns)
            # Pattern: "by Google", "by OpenAI", "from Microsoft", "developed by Google", etc.
            dev_patterns = [
                r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bfrom\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bdeveloped\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bcreated\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bintroduced\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\breleased\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:introduced|developed|created|released)',
            ]
            for pattern in dev_patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    # Validate it's a reasonable entity (not a common word)
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
            
            # For "releases X" or "X releases" patterns (e.g., "OpenAI releases GPT-2")
            release_patterns = [
                r'\breleases?\s+([A-Z][a-zA-Z0-9\-\.]+(?:\s+[A-Z][a-zA-Z0-9\-\.]+)*)',
                r'\b([A-Z][a-zA-Z0-9\-\.]+(?:\s+[A-Z][a-zA-Z0-9\-\.]+)*)\s+releases?',
            ]
            for pattern in release_patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
        
        # For "Who..." questions, try to extract the person/organization
        if question_lower.startswith('who '):
            # Pattern: "was developed by X", "was created by X", "was introduced by X"
            who_patterns = [
                r'\bwas\s+(?:developed|created|introduced|released|designed)\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bdeveloped\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bcreated\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            ]
            for pattern in who_patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip()
                    if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                        return entity
            
            # Also check for "X developed Y" pattern
            match = re.search(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:developed|created|introduced|released)\s+', full_answer, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()
                if len(entity) > 1 and entity.lower() not in ('the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'):
                    return entity
        if question_lower.startswith(('what are some', 'what are the', 'which ', 'list ')):
            # Look for comma-separated lists or "like X, Y, and Z" patterns
            patterns = [
                r'like\s+([^.]+)',
                r'including\s+([^.]+)',
                r'such\s+as\s+([^.]+)',
                r'examples?\s+(?:include|are)\s+([^.]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    examples = match.group(1).strip()
                    # Clean up the examples text
                    examples = re.sub(r'\s+', ' ', examples)
                    if len(examples.split()) <= 10:  # Reasonable length
                        return examples
        
        # For "What is..." definitional questions, extract the definition core
        if question_lower.startswith(('what is ', 'what does ', 'define ')):
            # Try to extract the main clause after "is/are/was/were"
            patterns = [
                r'\bis\s+(?:a|an|the)?\s*([^.]+)',
                r'\bare\s+(?:the)?\s*([^.]+)',
                r'\bwas\s+(?:a|an|the)?\s*([^.]+)',
                r'\bwere\s+(?:the)?\s*([^.]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    definition = match.group(1).strip()
                    # Take first clause (up to comma or semicolon)
                    definition = re.split(r'[,;]', definition)[0].strip()
                    if 2 <= len(definition.split()) <= 15:
                        return definition
        
        # For "How..." questions, extract the method/process
        if question_lower.startswith('how '):
            patterns = [
                r'\bby\s+([^.]+)',
                r'\bthrough\s+([^.]+)',
                r'\bvia\s+([^.]+)',
                r'\busing\s+([^.]+)',
                r'\blearning\s+([^.]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, full_answer, re.IGNORECASE)
                if match:
                    method = match.group(1).strip()
                    method = re.split(r'[,;]', method)[0].strip()
                    if 2 <= len(method.split()) <= 15:
                        return method
        
        # Default: return the first clause of the sentence (up to first comma or semicolon)
        # but only if it's reasonably short AND the question is not an explanatory question
        first_clause = re.split(r'[,;]', full_answer)[0].strip()
        if 2 <= len(first_clause.split()) <= 15:
            # For explanatory questions ("What is...", "How...", "Why..."), 
            # only return first clause if it's a complete thought (not a fragment)
            if question_lower.startswith(('what is ', 'what does ', 'define ', 'how ', 'why ')):
                # Check if first clause looks like a complete answer (has verb)
                clause_words = first_clause.split()
                has_verb = any(v in ' '.join(clause_words) for v in [' is ', ' are ', ' was ', ' were ', ' can ', ' will ', ' would ', ' should ', ' could ', ' learns? ', ' creates? ', ' generates? ', ' produces? ', ' uses? ', ' models? ', ' trains? ', ' predicts? ', ' classifies? ', ' recognizes? ', ' understands? ', ' processes? ', ' analyzes? '])
                if has_verb:
                    return first_clause
                # Otherwise fall through to return full answer
            else:
                return first_clause
        
        # Fallback: return the full answer if no concise extraction works
        return full_answer

    def _build_mcq_from_answer_candidate(
        self,
        question: str,
        answer_candidate: Dict[str, Any],
        all_candidates: List[Dict[str, Any]],
        context: str,
        config: GeneratorConfig
    ) -> GroundedQuizQuestion:
        """
        Build a complete MCQ from a question and pre-selected answer candidate.
        
        Uses other answer candidates as distractors for better quality.
        """
        clean_context = _clean_context_for_extraction(context)
        # Use the cleaned answer candidate text (already cleaned in _extract_answer_candidates)
        correct_answer = answer_candidate['text']
        
        # Extract a concise answer span from the full sentence based on the question
        concise_answer = self._extract_concise_answer(question, correct_answer, clean_context)
        if concise_answer and _is_valid_answer_candidate(concise_answer):
            # For "name of" questions, trim common suffixes like "model", "architecture", "system"
            if question.lower().startswith('what is the name'):
                concise_answer = re.sub(r'\s+(model|architecture|system)$', '', concise_answer, flags=re.IGNORECASE)
            correct_answer = concise_answer
        
        # Validate the answer candidate itself
        if not _is_valid_answer_candidate(correct_answer):
            raise ValueError(f"Answer candidate failed validation: {correct_answer}")
        
        # Validate answer support
        is_valid, support_score, evidence = _validate_answer_support(correct_answer, clean_context, config)
        if not is_valid:
            raise ValueError(f"Answer not supported by context: '{correct_answer}' (score: {support_score:.3f})")
        
        # Generate distractors from OTHER answer candidates (not random fragments)
        distractors = []
        other_candidates = [c for c in all_candidates if c['text'] != correct_answer and c['text'] != answer_candidate['text']]
        
        # Also include the original full sentence from answer_candidate if different
        original_text = answer_candidate.get('sentence', answer_candidate['text'])
        if original_text != correct_answer:
            other_candidates = [c for c in other_candidates if c['text'] != original_text]
        
        # Use SBERT to find good distractors from other candidates
        try:
            from ..models import get_sbert
            sbert = get_sbert()
            correct_emb = sbert.encode([correct_answer], convert_to_numpy=True, normalize_embeddings=True)
            
            scored_distractors = []
            for cand in other_candidates:
                cand_text = cand['text']
                if cand_text.lower() == correct_answer.lower() or correct_answer.lower() in cand_text.lower():
                    continue
                # Check word overlap
                correct_words = set(correct_answer.lower().split())
                cand_words = set(cand_text.lower().split())
                overlap = len(correct_words & cand_words) / max(len(correct_words), 1)
                if overlap >= config.distractor_max_overlap:
                    continue
                
                cand_emb = sbert.encode([cand_text], convert_to_numpy=True, normalize_embeddings=True)
                cos_sim = float(correct_emb[0] @ cand_emb[0])
                if config.distractor_min_similarity <= cos_sim <= config.distractor_max_similarity:
                    scored_distractors.append((cos_sim, cand_text))
            
            scored_distractors.sort(key=lambda x: abs(x[0] - 0.45))
            # Ensure uniqueness among distractors
            seen_distractor_texts = set()
            for _, d in scored_distractors:
                d_lower = d.lower().strip()
                if d_lower not in seen_distractor_texts and d_lower != correct_answer.lower().strip():
                    seen_distractor_texts.add(d_lower)
                    distractors.append(d)
                    if len(distractors) >= 3:
                        break
        except Exception as e:
            log.warning(f"SBERT distractor scoring failed: {e}")
            # Fallback: use first few other candidates, ensuring uniqueness
            seen_distractor_texts = set()
            for c in other_candidates:
                cand_text = c['text']
                cand_lower = cand_text.lower().strip()
                if cand_lower != correct_answer.lower().strip() and cand_lower not in seen_distractor_texts:
                    seen_distractor_texts.add(cand_lower)
                    distractors.append(cand_text)
                    if len(distractors) >= 3:
                        break
        
        # Pad with generic placeholders if needed, ensuring uniqueness
        placeholders = [
            "None of the above",
            "All of the above", 
            "Cannot be determined from the text",
            "Not mentioned in the context"
        ]
        seen_distractor_texts = {d.lower().strip() for d in distractors}
        seen_distractor_texts.add(correct_answer.lower().strip())
        for p in placeholders:
            if len(distractors) >= 3:
                break
            p_lower = p.lower().strip()
            if p_lower not in seen_distractor_texts:
                seen_distractor_texts.add(p_lower)
                distractors.append(p)
        
        distractors = distractors[:3]
        
        # Build options
        options = [correct_answer] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        # Validate options
        if len(options) != 4:
            raise ValueError("Invalid number of options")
        if len(set(opt.lower() for opt in options)) != 4:
            raise ValueError("Duplicate options detected")
        
        # Generate explanation
        explanation = _generate_reasoning(question, correct_answer, clean_context)
        
        return GroundedQuizQuestion(
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation
        )


def _filter_clean_sentences(context: str) -> List[str]:
    """Extract and filter sentences from cleaned context."""
    import nltk
    sentences = nltk.sent_tokenize(context)
    
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Skip very short sentences
        if len(sent.split()) < 5:
            continue
        # Skip metadata-like sentences
        if sent.startswith('[') or 'Source' in sent or 'Page/Slide' in sent:
            continue
        # Skip lines that are mostly punctuation/numbers
        alpha_ratio = sum(c.isalpha() for c in sent) / max(len(sent), 1)
        if alpha_ratio < 0.3:
            continue
        # Skip sentences with PDF artifacts
        if _is_pdf_artifact_line(sent):
            continue
        # Skip sentences that are mostly bullets/symbols
        if sent.startswith('\uf071') or sent.startswith('\uf0b7'):
            continue
        valid_sentences.append(sent)
    
    return valid_sentences


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
    "_extract_keywords",
    "_generate_distractors",
    "_generate_reasoning",
    "build_mcq_from_question",
    "_clean_context_for_extraction",
    "_extract_answer_from_context",
    "_validate_answer_support",
    "_is_meaningful_question",
    "_filter_clean_sentences",
    "_extract_answer_candidates",
    "_generate_question_for_answer",
    "_is_quality_fact_sentence",
]