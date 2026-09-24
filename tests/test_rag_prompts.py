"""
test_rag_prompts.py — Prompt Template Tests
============================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.rag.prompts import (
    GROUNDED_QG_SYSTEM_PROMPT,
    GROUNDED_QG_PROMPT_TEMPLATE,
    SIMPLE_GROUNDED_QG_PROMPT,
    DISTRACTOR_AWARE_PROMPT,
    get_prompt_template,
    format_prompt,
    PROMPT_VERSION,
)


class TestPromptVersion:
    def test_version_exists(self):
        assert PROMPT_VERSION == "1.0.0"


class TestSystemPrompt:
    def test_contains_core_rules(self):
        assert "Use ONLY the supplied context" in GROUNDED_QG_SYSTEM_PROMPT
        assert "Do NOT invent facts" in GROUNDED_QG_SYSTEM_PROMPT  # Updated to match new prompt
        assert "INSUFFICIENT_CONTEXT" in GROUNDED_QG_SYSTEM_PROMPT
        assert "correct answer MUST be directly supported" in GROUNDED_QG_SYSTEM_PROMPT


class TestPromptTemplates:
    def test_get_prompt_template_default(self):
        template = get_prompt_template()
        assert template == GROUNDED_QG_PROMPT_TEMPLATE

    def test_get_prompt_template_by_name(self):
        assert get_prompt_template("grounded_qg") == GROUNDED_QG_PROMPT_TEMPLATE
        assert get_prompt_template("simple") == SIMPLE_GROUNDED_QG_PROMPT
        assert get_prompt_template("distractor_aware") == DISTRACTOR_AWARE_PROMPT

    def test_get_prompt_template_unknown(self):
        # Unknown should return default
        template = get_prompt_template("unknown")
        assert template == GROUNDED_QG_PROMPT_TEMPLATE


class TestFormatPrompt:
    def test_format_basic(self):
        context = "Machine learning is a subset of AI."
        prompt = format_prompt(
            context=context,
            topic_focus="machine learning",
            difficulty="medium"
        )
        
        assert "Machine learning is a subset of AI." in prompt
        assert "machine learning" in prompt
        assert "medium" in prompt
        assert "Use ONLY the supplied context" in prompt
        assert "INSUFFICIENT_CONTEXT" in prompt

    def test_format_with_custom_system_prompt(self):
        context = "Test context"
        custom_system = "Custom system prompt"
        prompt = format_prompt(
            context=context,
            system_prompt=custom_system
        )
        assert custom_system in prompt

    def test_format_different_templates(self):
        context = "Test context"
        
        grounded = format_prompt(context, template_name="grounded_qg")
        simple = format_prompt(context, template_name="simple")
        distractor = format_prompt(context, template_name="distractor_aware")
        
        assert "JSON" in grounded
        assert "Question:" in simple
        assert "Distractors" in distractor

    def test_format_all_placeholders_filled(self):
        """Ensure all template placeholders are filled."""
        prompt = format_prompt(
            context="Context here",
            topic_focus="specific topic",
            difficulty="hard"
        )
        # Should not contain unfilled placeholders
        assert "{context}" not in prompt
        assert "{topic_focus}" not in prompt
        assert "{difficulty}" not in prompt
        assert "{system_prompt}" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])