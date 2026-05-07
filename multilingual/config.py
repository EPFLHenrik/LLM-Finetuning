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
DEFAULT_MAX_NEW_TOKENS_BY_TASK = {
    "mcq": 64,
    "math": 1024,
}