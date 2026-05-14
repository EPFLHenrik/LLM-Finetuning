"""multilingual/config.py"""

from pathlib import Path

REPO = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
TARGET_LANGUAGES = {
    "russian": {
        "include_config": "Russian",
        "mgsm_configs": ["ru"],
    },
    "chinese": {
        "include_config": "Chinese",
        "mgsm_configs": ["zh"],
    },
    "italian": {
        "include_config": "Italian",
        "mgsm_configs": [],
    },
    "spanish": {
        "include_config": "Spanish",
        "mgsm_configs": ["es"],
    },
    "hindi": {
        "include_config": "Hindi",
        "mgsm_configs": [],
    },
    "english": {
        "include_config": None,
        "mgsm_configs": ["en"],
    },
}

LETTERS = [chr(ord("A") + i) for i in range(26)]
MCQ_LETTERS = ["A", "B", "C", "D"]

# ------------------------------------------------------------------
# Official evaluation format constraints
# ------------------------------------------------------------------
# During the official CI eval, vLLM is called with:
#   - this system prompt (the only thing we control)
#   - the raw question as the user turn (we cannot modify it)
# max_new_tokens is fixed at 16384 by the CI; we use a lower cap
# locally to keep evaluation fast, but think mode needs much more
# headroom than the original 64-token MCQ budget.
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a multilingual reasoning assistant. "
    "Think carefully before answering. "
    "Always enclose your final answer in \\boxed{}, "
    "for example \\boxed{A} for multiple-choice questions "
    "or \\boxed{42} for numeric answers. "
    "Respond in the same language as the question."
)

# local token budgets for fast eval runs.
# think mode produces a <think>...</think> block before the answer,
# which can easily be 500-2000 tokens even for simple MCQ questions.
# 2048 is a reasonable local cap; use 16384 to match CI exactly.
DEFAULT_MAX_NEW_TOKENS_BY_TASK = {
    "mcq": 2048,
    "math": 4096,
}

# fraction of the available data to hold out as a local validation set.
# applied when no dedicated val/dev split exists (e.g. INCLUDE only has test).
VAL_FRACTION = 0.15
VAL_SEED = 42