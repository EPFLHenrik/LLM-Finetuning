from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationTask:
    row_id: int
    input_text: str
    expected_answer: str
    primary_prompt: str
    fallback_prompt: str


@dataclass
class AttemptState:
    distinct_try_count: int = 0
    wrong_distinct_try_count: int = 0
    total_generation_calls: int = 0
    seen_generations: set[str] = field(default_factory=set)
    chosen_reasoning: str = ""
    chosen_answer: str | None = None
    used_fallback: bool = False


@dataclass
class GenerationResult:
    task: GenerationTask
    state: AttemptState

    @property
    def succeeded_without_fallback(self) -> bool:
        return (
            self.state.chosen_answer == self.task.expected_answer
            and not self.state.used_fallback
        )

    @property
    def required_fallback(self) -> bool:
        return self.state.used_fallback