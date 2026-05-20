from __future__ import annotations

from typing import Callable


class HiddenCotPromptBuilder:
    def __init__(self, prompt_builder: Callable[[str], str]) -> None:
        self.prompt_builder = prompt_builder

    def build(self, tokenizer, user_text: str, think: bool) -> str:
        messages = [
            {"role": "user", "content": self.prompt_builder(user_text)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )


class FallbackPromptBuilder:
    def __init__(self, fallback_prompt_factory: Callable[[str, str], str]) -> None:
        self.fallback_prompt_factory = fallback_prompt_factory

    def build(self, tokenizer, user_text: str, expected_answer: str, think: bool) -> str:
        messages = [
            {
                "role": "user",
                "content": self.fallback_prompt_factory(user_text, expected_answer),
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )