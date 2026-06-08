# Merging

Contains model merge utilities for multilingual checkpoint combination.

Key files:
- `full_slerp.py`, `full_task_arithmetic.py`, `full_ties.py`: full-model merge experiments.
- `LoRA_dare_linear.py`, `LoRA_dare_ties.py`, `LoRA_task_arithmetic.py`, `LoRA_ties.py`: LoRA-specific merge experiments.
- `utils/parse_arguments.py`: shared argument parsing helper.
- `utils/peft_merge.py`: reusable PEFT merge logic.

These scripts are for combining adapters or full weights after training. They are not required for the basic dataset preparation and SFT build pipeline.
