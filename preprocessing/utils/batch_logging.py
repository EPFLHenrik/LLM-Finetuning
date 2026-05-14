from __future__ import annotations

from pathlib import Path

from preprocessing.utils.generation_types import GenerationResult


class BatchFailureLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.lines: list[str] = []

    def initialize(
        self,
        *,
        input_path: Path,
        output_path: Path,
        fallback_output_path: Path,
        effective_from: int,
        effective_to: int,
    ) -> None:
        self.lines = [
            f"# Failed generations during processing of {input_path}",
            f"# output={output_path}",
            f"# fallback_output={fallback_output_path}",
            f"# processing_range=[{effective_from}, {effective_to})",
            "",
        ]
        self.flush()

    def append_batch(
        self,
        *,
        batch_number: int,
        results: list[GenerationResult],
        max_distinct_tries: int,
    ) -> None:
        if not results:
            return

        batch_ids = [result.task.row_id for result in results]
        batch_start_id = batch_ids[0]
        batch_end_id = batch_ids[-1] + 1

        self.lines.append(f"Batch N.{batch_number} ({batch_start_id} - {batch_end_id}):")

        fallback_results = [result for result in results if result.state.used_fallback]
        for result in fallback_results:
            self.lines.append(
                "FAIL (Fallback): "
                f"[id={result.task.row_id}] "
                f"distinct_failed_tries={result.state.wrong_distinct_try_count}/{max_distinct_tries}, "
                f"total_generations={result.state.total_generation_calls}."
            )

        for tries in range(max_distinct_tries, 0, -1):
            grouped = [
                result
                for result in results
                if (not result.state.used_fallback)
                and result.state.wrong_distinct_try_count == tries
            ]
            for result in grouped:
                self.lines.append(
                    f"FAIL ({tries}/{max_distinct_tries} tries): "
                    f"[id={result.task.row_id}] "
                    f"matched on distinct_try={result.state.distinct_try_count}, "
                    f"total_generations={result.state.total_generation_calls}."
                )

        self.lines.append("")
        self.flush()

    def flush(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.lines).rstrip() + "\n"
        self.log_path.write_text(content, encoding="utf-8")