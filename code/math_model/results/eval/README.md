# Evaluation results — broad profile

These are the committed evaluation artifacts backing the results reported for the
math model. All four were produced by `evaluation/evaluate.py` with the **same**
configuration:

- profile: `broad` (MATH-500 stratified L1–5, MATH-500 L5, AIME 2024, AIME 2025, GSM8K)
- budget: `final` (16384 tokens), thinking on, `n = 8` (pass@8)

Each file is the raw evaluator output (`summary` + `per_subset_detailed`). The
`summary.model` field has been normalized to the logical checkpoint path (matching
the README's Outputs section); the original eval-time identifier is listed under
"Provenance" below.

| File | Checkpoint (lineage stage) | `summary.model` | weighted pass@1 / pass@8 | MATH-500 (L1–5) pass@8 |
|---|---|---|---|---|
| `base_broad.json` | Base Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 0.490 / 0.655 | 0.810 |
| `sft_checkpoint-600_broad.json` | + SFT (checkpoint-600, merged) | `checkpoints/math_model/sft_merged` | 0.424 / 0.625 | 0.890 |
| `rl_gspo_phase1_broad.json` | + GSPO phase 1 (merged) | `checkpoints/math_model/rl_phase1_merged` | 0.417 / 0.645 | 0.940 |
| `rl_gspo_phase2_final_broad.json` | + GSPO phase 2 (checkpoint-800, **final shipped**) | `checkpoints/math_model/final` | 0.419 / 0.684 | 0.910 |

**Provenance** — original eval-time `summary.model` strings (before normalization):

| File | Original identifier |
|---|---|
| `base_broad.json` | `Qwen/Qwen3-1.7B` |
| `sft_checkpoint-600_broad.json` | `/scratch/user/ckpt/qwen3_1.7b_sft600_merged` |
| `rl_gspo_phase1_broad.json` | `cs-552-2026-mystery-machine/math_model` (HF slug held the phase-1 merge at eval time) |
| `rl_gspo_phase2_final_broad.json` | `/tmp/merged_r2ckpt800` |

To regenerate any row:

```
python -m code.math_model.evaluation.run_benchmark \
    --name <label> --checkpoint <ckpt_dir_or_HF_id> --benchmark broad
```
