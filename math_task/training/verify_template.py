"""Local sanity check for the chat template.

Runs on your laptop (no GPU). Downloads only the Qwen3-1.7B tokenizer (~10MB).

From the repo root:
    python -m math_task.training.verify_template

Pass criteria:
  - Template renders without errors
  - Output ends with: <|im_start|>assistant\\n<think>\\n
  - No `<think>` block appears inside the user turn
"""
from pathlib import Path

from transformers import AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-1.7B"
TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"


def main() -> None:
    template = TEMPLATE_PATH.read_text()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.chat_template = template

    test_msg = [
        {"role": "user",
         "content": "What is the smallest prime greater than 100? Put the final answer in \\boxed{}."}
    ]
    rendered = tok.apply_chat_template(
        test_msg, tokenize=False, add_generation_prompt=True
    )

    print("=== rendered template ===")
    print(rendered)
    print("=== END ===\n")

    expected_suffix = "<|im_start|>assistant\n<think>\n"
    if not rendered.endswith(expected_suffix):
        raise AssertionError(
            f"Template does not end with thinking-ON generation prompt.\n"
            f"  Expected suffix: {expected_suffix!r}\n"
            f"  Got last 80 chars: ...{rendered[-80:]!r}"
        )
    if "<think>" in rendered[: -len(expected_suffix)]:
        raise AssertionError(
            "Unexpected `<think>` tag inside non-generation portion of the template."
        )
    print("OK: template ends with thinking-ON generation prompt and has no stray <think> tags.")


if __name__ == "__main__":
    main()
