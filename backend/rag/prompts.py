"""
prompts.py — Prompt Templates for RAG Grounded Generation
==========================================================
Stores prompt templates separately from generation logic for versioning and configurability.
"""

from typing import Optional


# Version for tracking prompt evolution
PROMPT_VERSION = "1.0.0"

# System prompt defining the grounded generation task
GROUNDED_QG_SYSTEM_PROMPT = """You are an expert educational question generator. Your task is to create high-quality multiple-choice questions STRICTLY based on the provided context.

CORE RULES - VIOLATION MEANS FAILURE:
1. Use ONLY the supplied context — do NOT use outside knowledge, training data, or common sense
2. Do NOT invent facts, names, dates, numbers, or concepts not explicitly in the context
3. Generate a question that CAN be fully answered from the context alone
4. If the context is insufficient to generate a valid question, you MUST respond with INSUFFICIENT_CONTEXT
5. The correct answer MUST be directly supported by text in the context
6. Distractors (wrong options) should be plausible but clearly incorrect based on the context
7. Provide a brief explanation citing the specific part of the context that supports the answer

FORBIDDEN:
- Using knowledge not in the provided context
- Making assumptions beyond what is written
- Inventing answer options not grounded in the text
- Generating questions about topics not covered in the context

REQUIRED: Every question, option, and explanation must be traceable to the provided context."""

# Main prompt template for grounded MCQ generation
GROUNDED_QG_PROMPT_TEMPLATE = """{system_prompt}

CONTEXT:
{context}

TASK:
Generate a {difficulty} multiple-choice question about {topic_focus} based ONLY on the above context.

CRITICAL: Your response will be validated for groundedness. Any fact not in the context will cause rejection.

REQUIRED OUTPUT FORMAT (JSON):
{{
  "question": "Your question here?",
  "options": [
    "Option A text",
    "Option B text", 
    "Option C text",
    "Option D text"
  ],
  "correct_answer": "The exact text of the correct option",
  "explanation": "Brief explanation citing context evidence (e.g., 'The context states...')"
}}

If the context does not contain enough information to generate a valid question on this topic, output exactly:
{{
  "error": "INSUFFICIENT_CONTEXT",
  "reason": "Brief reason why context is insufficient"
}}"""

# Alternative simpler prompt for cases where structured output is difficult
SIMPLE_GROUNDED_QG_PROMPT = """{system_prompt}

CONTEXT:
{context}

Generate a {difficulty} multiple-choice question about {topic_focus} based ONLY on the context above.

Question: [your question]
A) [option 1]
B) [option 2]
C) [option 3]
D) [option 4]
Correct: [A/B/C/D]
Explanation: [explanation citing context]"""

# Prompt for generating distractor-aware questions
DISTRACTOR_AWARE_PROMPT = """{system_prompt}

CONTEXT:
{context}

TASK:
Generate a {difficulty} multiple-choice question about {topic_focus} using ONLY the context.

CRITICAL: Your response will be validated for groundedness. Any fact not in the context will cause rejection.

IMPORTANT: The distractors (wrong answers) should be:
- Plausible to someone who hasn't read the context carefully
- Clearly incorrect based on the context
- Related to the topic but not the correct answer
- Distinct from each other
- Derived from or related to content in the context (not invented)

OUTPUT FORMAT (JSON):
{{
  "question": "Your question here?",
  "options": [
    "Option A text",
    "Option B text", 
    "Option C text",
    "Option D text"
  ],
  "correct_answer": "The exact text of the correct option",
  "explanation": "Brief explanation citing context evidence"
}}"""


def get_prompt_template(template_name: str = "grounded_qg") -> str:
    """Get a prompt template by name."""
    templates = {
        "grounded_qg": GROUNDED_QG_PROMPT_TEMPLATE,
        "simple": SIMPLE_GROUNDED_QG_PROMPT,
        "distractor_aware": DISTRACTOR_AWARE_PROMPT,
    }
    return templates.get(template_name, GROUNDED_QG_PROMPT_TEMPLATE)


def format_prompt(
    context: str,
    topic_focus: str = "the key concepts",
    difficulty: str = "medium",
    template_name: str = "grounded_qg",
    system_prompt: Optional[str] = None
) -> str:
    """
    Format a prompt with the given context and parameters.
    
    Args:
        context: Retrieved evidence chunks formatted as text
        topic_focus: What the question should focus on
        difficulty: "easy", "medium", or "hard"
        template_name: Which prompt template to use
        system_prompt: Optional custom system prompt
    
    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(template_name)
    sys_prompt = system_prompt or GROUNDED_QG_SYSTEM_PROMPT
    return template.format(
        system_prompt=sys_prompt,
        context=context,
        topic_focus=topic_focus,
        difficulty=difficulty
    )


# Export
__all__ = [
    "PROMPT_VERSION",
    "GROUNDED_QG_SYSTEM_PROMPT",
    "GROUNDED_QG_PROMPT_TEMPLATE",
    "SIMPLE_GROUNDED_QG_PROMPT",
    "DISTRACTOR_AWARE_PROMPT",
    "get_prompt_template",
    "format_prompt",
]