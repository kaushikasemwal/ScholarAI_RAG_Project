"""
output_parser.py — Structured Output Parsing for RAG Generation
===============================================================
Parses FLAN-T5 output into structured quiz questions with validation.
"""

import json
import re
from typing import List, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, ValidationError

try:
    from langchain_core.output_parsers import BaseOutputParser
except ImportError:
    BaseOutputParser = object


class GroundedQuizQuestion(BaseModel):
    """Structured output model for a grounded quiz question."""
    question: str = Field(..., min_length=5, description="The question text")
    options: List[str] = Field(
        ..., 
        min_length=4, 
        max_length=4, 
        description="Exactly 4 answer options"
    )
    correct_answer: str = Field(..., description="The exact text of the correct option")
    explanation: str = Field(..., description="Explanation citing context evidence")

    @field_validator("correct_answer")
    @classmethod
    def validate_correct_answer_in_options(cls, v, info):
        """Ensure correct_answer matches one of the options."""
        options = info.data.get("options", [])
        if options and v not in options:
            # Allow partial match (e.g., case-insensitive)
            if not any(v.lower() == opt.lower() for opt in options):
                raise ValueError(f"correct_answer '{v}' not found in options: {options}")
        return v


class InsufficientContextResult(BaseModel):
    """Result when context is insufficient for generation."""
    error: str = "INSUFFICIENT_CONTEXT"
    reason: str


class GenerationResult(BaseModel):
    """Result of grounded generation attempt."""
    success: bool
    question: Optional[GroundedQuizQuestion] = None
    error: Optional[str] = None
    raw_output: Optional[str] = None
    provenance: List[dict] = []


@dataclass
class ParseResult:
    """Result of parsing attempt."""
    success: bool
    question: Optional[GroundedQuizQuestion] = None
    error: Optional[str] = None
    raw_output: str = ""


class GroundedOutputParser:
    """
    Parser for grounded quiz generation output.
    Handles both JSON and simple text formats from FLAN-T5.
    """
    
    def __init__(self, strict: bool = True):
        self.strict = strict
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse model output into a structured quiz question.
        
        Args:
            text: Raw model output
            
        Returns:
            ParseResult with success status and parsed question or error
        """
        if not text or not text.strip():
            return ParseResult(
                success=False,
                error="Empty model output",
                raw_output=text
            )
        
        # Try JSON format first (primary format)
        result = self._try_parse_json(text)
        if result.success:
            return result
        # If JSON parsing failed with a specific known error (like INSUFFICIENT_CONTEXT),
        # return it directly instead of falling through
        if result.error and "insufficient context" in result.error.lower():
            return result
        
        # Try simple text format as fallback
        result = self._try_parse_simple(text)
        if result.success:
            return result

        # Try labeled format (T5-friendly)
        result = self._try_parse_labeled(text)
        if result.success:
            return result
        
        return ParseResult(
            success=False,
            error="Failed to parse output in any supported format",
            raw_output=text
        )
    
    def _try_parse_json(self, text: str) -> ParseResult:
        """Try to parse as JSON format."""
        # Extract JSON from potential surrounding text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return ParseResult(success=False, error="No JSON found", raw_output=text)
        
        json_str = json_match.group(0)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return ParseResult(success=False, error=f"JSON decode error: {e}", raw_output=text)
        
        # Check for explicit insufficient context marker
        if data.get("error") == "INSUFFICIENT_CONTEXT":
            return ParseResult(
                success=False,
                error=f"Insufficient context: {data.get('reason', 'No reason given')}",
                raw_output=text
            )
        
        # Validate with Pydantic
        try:
            question = GroundedQuizQuestion(**data)
            return ParseResult(success=True, question=question, raw_output=text)
        except ValidationError as e:
            if self.strict:
                return ParseResult(success=False, error=f"Validation error: {e}", raw_output=text)
            # In non-strict mode, try to recover
            return self._attempt_recovery(data, text)
    
    def _try_parse_simple(self, text: str) -> ParseResult:
        """Try to parse simple text format."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        question = None
        options = []
        correct_idx = None
        explanation = None
        
        for line in lines:
            if line.lower().startswith("question:"):
                question = line[len("question:"):].strip()
            elif line.lower().startswith(("a)", "a.")):
                options.append(line[2:].strip())
            elif line.lower().startswith(("b)", "b.")):
                options.append(line[2:].strip())
            elif line.lower().startswith(("c)", "c.")):
                options.append(line[2:].strip())
            elif line.lower().startswith(("d)", "d.")):
                options.append(line[2:].strip())
            elif line.lower().startswith("correct:"):
                correct_text = line[len("correct:"):].strip().upper()
                if correct_text in ("A", "B", "C", "D"):
                    correct_idx = ord(correct_text) - ord("A")
            elif line.lower().startswith("explanation:"):
                explanation = line[len("explanation:"):].strip()
        
        if not question or len(options) != 4 or correct_idx is None or not explanation:
            return ParseResult(success=False, error="Incomplete simple format", raw_output=text)
        
        correct_answer = options[correct_idx]
        
        try:
            q = GroundedQuizQuestion(
                question=question,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation
            )
            return ParseResult(success=True, question=q, raw_output=text)
        except ValidationError as e:
            return ParseResult(success=False, error=f"Validation error: {e}", raw_output=text)

    def _try_parse_labeled(self, text: str) -> ParseResult:
        """Try to parse T5-friendly labeled format:
        
        QUESTION: <question>
        A: <option A>
        B: <option B>
        C: <option C>
        D: <option D>
        ANSWER: <exact correct option text>
        EXPLANATION: <explanation>
        
        Also accepts compact format without line breaks:
        QUESTION: <q> A: <a> B: <b> C: <c> D: <d> ANSWER: <ans> EXPLANATION: <exp>
        
        Also accepts the compact format FLAN-T5 actually produces:
        Question: <q> Options:A <a> B <b> C <c> D <d> Answer:<letter>
        
        Also accepts:
        - "Options:A <text> B <text> C <text> D <text> Answer:<letter>" format
        - "Answer:<letter>" where letter maps to option
        - Missing EXPLANATION (will be generated from context)
        
        Also accepts ANSWER: A (letter) which maps to the first option.
        """
        if not text or not text.strip():
            return ParseResult(success=False, error="Empty model output", raw_output=text)
        
        import re
        text_single = text.replace('\n', ' ').strip()
        
        question = None
        options = {}
        correct_answer = None
        explanation = None
        
        # Try to extract QUESTION (case-insensitive) - handles "Question:" or "QUESTION:"
        q_match = re.search(r'QUESTION:\s*(.+?)(?:\s*(?:OPTIONS?:|A:|B:|C:|D:|ANSWER:|Answer:|$))', text_single, re.IGNORECASE)
        if q_match:
            question = q_match.group(1).strip()
        
        # Try to extract options from "Options:A ... B ... C ... D ..." format (case-insensitive)
        # This handles the compact format FLAN-T5 produces: "Options:A textB textC textD text"
        opt_match = re.search(r'OPTIONS?:\s*(.+?)(?:\s*ANSWER:|\s*Answer:|\s*$)', text_single, re.IGNORECASE)
        if opt_match:
            opt_text = opt_match.group(1).strip()
            # Parse A, B, C, D from the options text using a more robust approach
            # The format is: A <text>B <text>C <text>D <text> where letters are delimiters
            # We'll split by the option markers (A, B, C, D) that appear at word boundaries
            
            # First try to find each option by looking for the letter followed by content
            # Use a regex that captures from each letter to the next letter or end
            for opt_letter in ['A', 'B', 'C', 'D']:
                # Pattern: letter followed by content until next letter (A/B/C/D) or ANSWER
                if opt_letter == 'A':
                    pattern = r'A\s*(.+?)(?=\s*[BCD]\s*|$|ANSWER|Answer)'
                elif opt_letter == 'B':
                    pattern = r'B\s*(.+?)(?=\s*[CD]\s*|$|ANSWER|Answer)'
                elif opt_letter == 'C':
                    pattern = r'C\s*(.+?)(?=\s*D\s*|$|ANSWER|Answer)'
                else:  # D
                    pattern = r'D\s*(.+?)(?=\s*$|ANSWER|Answer)'
                
                m = re.search(pattern, opt_text, re.IGNORECASE)
                if m:
                    options[opt_letter] = m.group(1).strip()
        
        # If Options: format didn't work, try "A: ... B: ... C: ... D: ..." format directly in text
        if len(options) < 4:
            for opt_letter in ['A', 'B', 'C', 'D']:
                if opt_letter not in options:
                    # Try "A: text" format
                    pattern = rf'{opt_letter}:\s*(.+?)(?:\s+[BCD]:|\s*ANSWER:|\s*Answer:|\s*EXPLANATION:|\s*$)'
                    m = re.search(pattern, text_single, re.IGNORECASE)
                    if m:
                        options[opt_letter] = m.group(1).strip()
                    else:
                        # Try "A text" format (space after letter) - but only if it looks like an option start
                        # Look for letter at word boundary followed by content
                        pattern = rf'(?:^|\s){opt_letter}\s+(.+?)(?:\s+[BCD]\s+|\s*ANSWER:|\s*Answer:|\s*EXPLANATION:|\s*$)'
                        m = re.search(pattern, text_single, re.IGNORECASE)
                        if m:
                            options[opt_letter] = m.group(1).strip()
        
        # Extract ANSWER (case-insensitive, could be letter or full text)
        ans_match = re.search(r'ANSWER:\s*(.+?)(?:\s*EXPLANATION:|\s*$)', text_single, re.IGNORECASE)
        if not ans_match:
            ans_match = re.search(r'Answer:\s*(.+?)(?:\s*EXPLANATION:|\s*$)', text_single, re.IGNORECASE)
        if ans_match:
            correct_answer = ans_match.group(1).strip()
        
        # Extract EXPLANATION (case-insensitive)
        exp_match = re.search(r'EXPLANATION:\s*(.+)$', text_single, re.IGNORECASE)
        if exp_match:
            explanation = exp_match.group(1).strip()
        
        # Check for INSUFFICIENT_CONTEXT marker
        if question and question.upper().strip() == "INSUFFICIENT_CONTEXT":
            return ParseResult(
                success=False,
                error="Insufficient context: model indicated insufficient information",
                raw_output=text
            )
        
        # Validate required fields
        if not question:
            return ParseResult(success=False, error="Missing QUESTION field", raw_output=text)
        
        if len(options) != 4 or not all(k in options for k in ['A', 'B', 'C', 'D']):
            return ParseResult(success=False, error=f"Missing or incomplete options (need A, B, C, D), got: {list(options.keys())}", raw_output=text)
        
        if not correct_answer:
            return ParseResult(success=False, error="Missing ANSWER field", raw_output=text)
        
        # EXPLANATION is optional - generate a default if missing
        if not explanation:
            explanation = f"Based on the context provided, the correct answer is {correct_answer}."
        
        # Build options list in order
        options_list = [options['A'], options['B'], options['C'], options['D']]
        
        # Resolve correct_answer: if it's a letter (A/B/C/D), map to option text
        if correct_answer.upper() in ['A', 'B', 'C', 'D']:
            correct_text = options[correct_answer.upper()]
        else:
            # Exact text match
            correct_text = correct_answer
            # Verify it matches one of the options (case-insensitive)
            if not any(correct_text.lower() == opt.lower() for opt in options_list):
                return ParseResult(
                    success=False, 
                    error=f"ANSWER '{correct_text}' does not match any option", 
                    raw_output=text
                )
        
        try:
            q = GroundedQuizQuestion(
                question=question,
                options=options_list,
                correct_answer=correct_text,
                explanation=explanation
            )
            return ParseResult(success=True, question=q, raw_output=text)
        except ValidationError as e:
            return ParseResult(success=False, error=f"Validation error: {e}", raw_output=text)
        
        # Check for INSUFFICIENT_CONTEXT marker
        if question and question.upper().strip() == "INSUFFICIENT_CONTEXT":
            return ParseResult(
                success=False,
                error="Insufficient context: model indicated insufficient information",
                raw_output=text
            )
        
        # Validate required fields
        if not question:
            return ParseResult(success=False, error="Missing QUESTION field", raw_output=text)
        
        if len(options) != 4 or not all(k in options for k in ['A', 'B', 'C', 'D']):
            return ParseResult(success=False, error=f"Missing or incomplete options (need A, B, C, D), got: {list(options.keys())}", raw_output=text)
        
        if not correct_answer:
            return ParseResult(success=False, error="Missing ANSWER field", raw_output=text)
        
        if not explanation:
            return ParseResult(success=False, error="Missing EXPLANATION field", raw_output=text)
        
        # Build options list in order
        options_list = [options['A'], options['B'], options['C'], options['D']]
        
        # Resolve correct_answer: if it's a letter (A/B/C/D), map to option text
        if correct_answer.upper() in ['A', 'B', 'C', 'D']:
            correct_text = options[correct_answer.upper()]
        else:
            # Exact text match
            correct_text = correct_answer
            # Verify it matches one of the options (case-insensitive)
            if not any(correct_text.lower() == opt.lower() for opt in options_list):
                return ParseResult(
                    success=False, 
                    error=f"ANSWER '{correct_text}' does not match any option", 
                    raw_output=text
                )
        
        try:
            q = GroundedQuizQuestion(
                question=question,
                options=options_list,
                correct_answer=correct_text,
                explanation=explanation
            )
            return ParseResult(success=True, question=q, raw_output=text)
        except ValidationError as e:
            return ParseResult(success=False, error=f"Validation error: {e}", raw_output=text)

    def _attempt_recovery(self, data: dict, raw_text: str) -> ParseResult:
        """Attempt to recover a valid question from partially invalid data."""
        # Try to fix common issues
        fixed = data.copy()
        
        # Ensure exactly 4 options
        if "options" in fixed and isinstance(fixed["options"], list):
            opts = fixed["options"]
            if len(opts) > 4:
                fixed["options"] = opts[:4]
            elif len(opts) < 4:
                # Pad with generic options
                while len(fixed["options"]) < 4:
                    fixed["options"].append(f"Option {len(fixed['options'])+1}")
        
        # Ensure correct_answer is in options
        if "correct_answer" in fixed and "options" in fixed:
            if fixed["correct_answer"] not in fixed["options"]:
                # Try case-insensitive match
                for opt in fixed["options"]:
                    if opt.lower() == fixed["correct_answer"].lower():
                        fixed["correct_answer"] = opt
                        break
                else:
                    # Default to first option
                    fixed["correct_answer"] = fixed["options"][0]
        
        try:
            question = GroundedQuizQuestion(**fixed)
            return ParseResult(success=True, question=question, raw_output=raw_text)
        except ValidationError:
            return ParseResult(success=False, error="Recovery failed", raw_output=raw_text)


def create_grounded_output_parser(strict: bool = True) -> GroundedOutputParser:
    """Factory function to create a grounded output parser."""
    return GroundedOutputParser(strict=strict)


# LangChain-compatible output parser wrapper
class LangChainGroundedParser(BaseOutputParser):
    """LangChain-compatible output parser for grounded questions."""
    
    def __init__(self, strict: bool = True):
        super().__init__()
        self._parser = GroundedOutputParser(strict=strict)
    
    @property
    def parser(self):
        return self._parser
    
    def parse(self, text: str) -> GroundedQuizQuestion:
        result = self._parser.parse(text)
        if not result.success:
            raise ValueError(f"Failed to parse grounded output: {result.error}")
        return result.question
    
    def get_format_instructions(self) -> str:
        return """Output a JSON object with:
- question: string
- options: array of exactly 4 strings
- correct_answer: string (must match one option exactly)
- explanation: string"""


# Export
__all__ = [
    "GroundedQuizQuestion",
    "InsufficientContextResult",
    "GenerationResult",
    "ParseResult",
    "GroundedOutputParser",
    "LangChainGroundedParser",
    "create_grounded_output_parser",
]