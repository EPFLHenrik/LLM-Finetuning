class DatasetFieldNames:
    DS_INPUT = "input"
    DS_ANSWER = "answer"

    BENCHMARK_INPUT = "prompt"
    BENCHMARK_COMPLETIONS = "completions"
    BENCHMARK_ANSWER = "answer"

    SFT_INPUT = "prompt"
    SFT_TARGET = "target"
    SFT_ANSWER = "answer"

    RLVR_INPUT = "prompt"
    RLVR_ANSWER = "answer"

    DPO_INPUT = "prompt"
    DPO_REJECTED = "rejected"
    DPO_CHOSEN = "chosen"

class DistillationFieldNames:
    TEACHER_MODEL = "teacher_model"

class Random:
    SEED = 42

class TrainingConstants:
    MAX_NEW_TOKENS = 16384