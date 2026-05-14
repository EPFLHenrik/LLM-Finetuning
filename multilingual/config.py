"""multilingual/config.py"""
from pathlib import Path

REPO = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B"
SFT_DATA_DIR = REPO / "multilingual" / "data" / "sft"
TARGET_LANGUAGES = {
    "russian": {"include_config": "Russian",  "mgsm_configs": ["ru"]},
    "chinese": {"include_config": "Chinese",  "mgsm_configs": ["zh"]},
    "italian": {"include_config": "Italian",  "mgsm_configs": []},
    "spanish": {"include_config": "Spanish",  "mgsm_configs": ["es"]},
    "hindi":   {"include_config": "Hindi",    "mgsm_configs": []},
    "english": {"include_config": None,       "mgsm_configs": ["en"]},
}

LETTERS = [chr(ord("A") + i) for i in range(26)]
MCQ_LETTERS = ["A", "B", "C", "D"]

# Only the system prompt is under our control during official CI eval.
# The user turn is the raw question, unmodified.
SYSTEM_PROMPT = (
    "You are a multilingual reasoning assistant. "
    "Think briefly before answering — avoid repetition and unnecessary elaboration. "
    "For multiple-choice questions, choose exactly one option. "
    "Always end your response with exactly one boxed answer on the last line: "
    "use \\boxed{X} for multiple-choice questions, where X is the chosen letter, "
    "or \\boxed{42} for numeric answers. "
    "Do not output anything after the boxed answer. "
    "Respond in the same language as the question."
)

# Local token budgets. 3072 gives MCQ enough headroom for a short think
# block without letting the model loop. CI uses 16384.
DEFAULT_MAX_NEW_TOKENS_BY_TASK = {
    "mcq":  3072,
    "math": 4096,
}

# Used when no dedicated val split exists (e.g. INCLUDE, MGSM only have test).
VAL_FRACTION = 0.15
VAL_SEED = 42