from __future__ import annotations

import re


class BoxedAnswerExtractor:
    def extract(self, text: str) -> str | None:
        matches = re.findall(r"\\boxed\s*\{\s*([A-Za-z])\s*\}", text)
        if not matches:
            return None
        return matches[-1].upper()


class ThinkTagReasoningExtractor:
    def __init__(self, fallback_reasoning: str) -> None:
        self.fallback_reasoning = fallback_reasoning
        self.meta_patterns = [
            r"(?i)\b(given|provided|supplied|told)\s+(the\s+)?(correct\s+)?answer\b[^.?!]*[.?!]?",
            r"(?i)\b(the answer was provided|as provided|since the answer is given)\b[^.?!]*[.?!]?",
            r"(?i)\bi\s+(know|was told)\s+the\s+answer\b[^.?!]*[.?!]?",
        ]

    def extract(self, text: str) -> str:
        think_matches = re.findall(
            r"<think>(.*?)</think>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        reasoning = think_matches[-1] if think_matches else text

        reasoning = re.sub(r"<\\?think>", "", reasoning)
        reasoning = re.sub(r"</think>", "", reasoning)
        reasoning = re.sub(r"<\\think>", "", reasoning)
        reasoning = re.sub(r"\\boxed\{[^}]*\}", "", reasoning)

        for pattern in self.meta_patterns:
            reasoning = re.sub(pattern, " ", reasoning)

        reasoning = re.sub(r"\s+", " ", reasoning).strip()
        if not reasoning:
            reasoning = self.fallback_reasoning

        return reasoning


class TargetFormatter:
    def __init__(self, fallback_reasoning: str, use_think_tags: bool = True) -> None:
        self.fallback_reasoning = fallback_reasoning
        self.use_think_tags = use_think_tags

    def format(self, reasoning: str, final_answer: str) -> str:
        cleaned = re.sub(r"\s+", " ", reasoning).strip()
        if not cleaned:
            cleaned = self.fallback_reasoning

        if self.use_think_tags:
            return f"<think>\n{cleaned}\n</think>\n\\boxed{{{final_answer}}}."

        return f"\n{cleaned}\n\n\\boxed{{{final_answer}}}."