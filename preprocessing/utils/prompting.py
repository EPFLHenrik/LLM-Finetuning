from __future__ import annotations

from typing import Callable


class HiddenCotPromptBuilder:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build(self, tokenizer, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


class FallbackPromptBuilder:
    def __init__(self, fallback_system_prompt_factory: Callable[[str], str]) -> None:
        self.fallback_system_prompt_factory = fallback_system_prompt_factory

    def build(self, tokenizer, user_text: str, expected_answer: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.fallback_system_prompt_factory(expected_answer),
            },
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )