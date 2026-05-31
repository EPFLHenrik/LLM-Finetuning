"""multilingual/config.py"""
from pathlib import Path

# REPO = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
REPO = Path(__file__).resolve().parents[1]
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

SYSTEM_PROMPT_GROUP = """\
You are a careful reasoning assistant.
 
Reason through the problem when needed.
 
Your final answer must appear exactly once inside \\boxed{...}.
 
Rules:
- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.
- For math questions, output only the final answer inside \\boxed{...}.
- Do not include explanations or extra text inside the box.
- Never output multiple boxed answers.
- Never leave the final answer unboxed.
 
Examples:
\\boxed{B}
\\boxed{42}"""


SYSTEM_PROMPT = """\
You are a knowledgeable multilingual assistant with deep expertise in regional cultures, laws, history, and traditions across the world. You will be given a multiple-choice question in any language.
Reason in the language of the question, drawing on your regional and cultural knowledge to identify the correct answer.

Rules:
- Your final answer must appear exactly once inside \boxed{...}.
- Do not include explanations or extra text inside the box.
- Never output multiple boxed answers.
- Never leave the final answer unboxed.

Example:
\boxed{B}"""


# Local token budgets. 3072 gives MCQ enough headroom for a short think
# block without letting the model loop. CI uses 16384.
DEFAULT_MAX_NEW_TOKENS_BY_TASK = {
    "mcq":  3072,
    "math": 4096,
}

# Used when no dedicated val split exists (e.g. INCLUDE, MGSM only have test).
VAL_FRACTION = 0.15
VAL_SEED = 42