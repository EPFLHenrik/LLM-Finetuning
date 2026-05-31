from __future__ import annotations

from typing import Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from preprocessing.distillation.utils.generation_types import (
    AttemptState,
    GenerationResult,
    GenerationTask,
)
from preprocessing.distillation.utils.output_parsing import (
    BoxedAnswerExtractor,
    ThinkTagReasoningExtractor,
)


class RetryGenerationRunner:
    def __init__(
        self,
        *,
        model_name: str,
        max_new_tokens: int,
        max_distinct_tries: int,
        max_total_generations: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_distinct_tries = max_distinct_tries
        self.max_total_generations = max_total_generations
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = LLM(
            model=self.model_name,
            dtype="auto",
            gpu_memory_utilization=0.9,
        )

    def run_batch(
        self,
        tasks: list[GenerationTask],
        *,
        answer_extractor: BoxedAnswerExtractor,
        reasoning_extractor: ThinkTagReasoningExtractor,
    ) -> list[GenerationResult]:
        states = {task.row_id: AttemptState() for task in tasks}
        tasks_by_id = {task.row_id: task for task in tasks}

        pending_ids = [task.row_id for task in tasks]
        round_id = 0

        while pending_ids:
            round_id += 1
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                seed=self.seed + round_id,
            )

            prompts = [tasks_by_id[row_id].primary_prompt for row_id in pending_ids]
            outputs = self.model.generate(prompts, sampling_params)

            next_pending_ids: list[int] = []

            for row_id, output in zip(pending_ids, outputs):
                task = tasks_by_id[row_id]
                state = states[row_id]
                rationale = output.outputs[0].text.strip()
                state.total_generation_calls += 1

                if rationale not in state.seen_generations:
                    state.seen_generations.add(rationale)
                    state.distinct_try_count += 1
                    predicted_answer = answer_extractor.extract(rationale)

                    if predicted_answer == task.expected_answer:
                        state.chosen_reasoning = reasoning_extractor.extract(rationale)
                        state.chosen_answer = predicted_answer
                    else:
                        state.wrong_distinct_try_count += 1

                if state.chosen_answer != task.expected_answer and (
                    state.distinct_try_count < self.max_distinct_tries
                    and state.total_generation_calls < self.max_total_generations
                ):
                    next_pending_ids.append(row_id)

            pending_ids = next_pending_ids

        fallback_ids = [
            task.row_id
            for task in tasks
            if states[task.row_id].chosen_answer != task.expected_answer
        ]

        if fallback_ids:
            fallback_prompts = [tasks_by_id[row_id].fallback_prompt for row_id in fallback_ids]
            fallback_sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                seed=self.seed + 777,
            )
            fallback_outputs = self.model.generate(fallback_prompts, fallback_sampling_params)

            for row_id, output in zip(fallback_ids, fallback_outputs):
                state = states[row_id]
                state.chosen_reasoning = reasoning_extractor.extract(
                    output.outputs[0].text.strip()
                )
                state.chosen_answer = tasks_by_id[row_id].expected_answer
                state.used_fallback = True

        return [
            GenerationResult(task=task, state=states[task.row_id])
            for task in tasks
        ]

    def build_primary_prompt(self, user_text: str, prompt_builder, think: bool) -> str:
        return prompt_builder.build(self.tokenizer, user_text, think)

    def build_fallback_prompt(
        self,
        user_text: str,
        expected_answer: str,
        fallback_prompt_builder,
        think: bool,
    ) -> str:
        return fallback_prompt_builder.build(
            self.tokenizer,
            user_text,
            expected_answer,
            think,
        )