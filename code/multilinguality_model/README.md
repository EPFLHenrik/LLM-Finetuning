# Multilingual component

Code for the multilingual individual model (M3 submission): data preparation, SFT training, model merging, and evaluation across Russian, Chinese, Hindi, Italian, and Spanish.

## How to reproduce

From the repository root:

```bash
source setup_env.sh
bash code/multilinguality_model/repro/reproduce_best_model.sh
```

This downloads data from Hugging Face, trains three source models (`v1`, `v20-russian`, `v20-hindi`), merges them with TIES, and evaluates the result.

For a detached run:

```bash
nohup bash code/multilinguality_model/repro/reproduce_best_model.sh > logs/reproduce_best_model.log 2>&1 &
```

To run step by step:

```bash
bash code/multilinguality_model/repro/00_install_repro_requirements.sh
bash code/multilinguality_model/repro/01_sync_repro_data.sh
bash code/multilinguality_model/repro/03_train_export_all_sources.sh  # or 02_ for individual models
bash code/multilinguality_model/repro/04_merge_best_model.sh
bash code/multilinguality_model/repro/05_eval_best_model.sh
```

Output:
- Final model: `checkpoints/multilinguality_model/best_model`
- Evaluation results: `runs/benchmark/results/multilinguality_model/best_model`

See `repro/README.md` for full details.

## Data

No dataset files are committed. The `data/` folder is git-ignored; all artifacts live on HuggingFace at [`cs-552-2026-mystery-machine/multilingual`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual). The repro scripts handle syncing automatically.

| HF folder | Contents |
|---|---|
| [`processed_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/processed_data) | Preprocessed JSONL per source dataset |
| [`benchmarks/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/benchmarks) | Evaluation benchmark splits |
| [`sft_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/sft_data) | Final SFT mixtures used for training |
| [`raw_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/raw_data) | Raw source files before preprocessing |
| [`distilled_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/distilled_data) | Distilled reasoning traces (generated but not used in final mixture) |

To fetch individual subsets on demand without running the full repro pipeline, use `data_pipeline/download_processed_data.py` or `data_pipeline/download_distilled_data.py`. The main local data root is `data/multilinguality/`.

## Repository tree

```
multilinguality_model/
├── README.md
├── __init__.py
├── config.py
├── paths.py
├── utils.py
├── generation.py
├── setup_env.sh                       # environment setup
├── assets/                            # prompt templates and chat formatting assets
│   ├── README.md
│   ├── chat_template.jinja
│   ├── chat_template_multilingual-augmented.jinja
│   ├── chat_template_multilingual-nonthinking.jinja
│   ├── chat_template_multilingual-thinking.jinja
│   └── generation_config.json
├── configs/
│   └── merging/
│       └── best_model_ties.yaml       # TIES merge config used by repro workflow
├── data_pipeline/                     # download, parse, preprocess, distill datasets
│   ├── README.md
│   ├── distillation/
│   ├── generation/
│   ├── parsing/
│   └── preprocessing/
├── loaders/                           # dataset loaders for unified example schema
│   └── README.md
├── train/                             # SFT mixture build, training, and export
│   ├── build_sft_data.py
│   ├── build_distilled_sft_data.py
│   ├── build_sft_augmented_mcq.py
│   ├── sft.py
│   ├── export.py
│   └── export_full.py
├── eval/                              # evaluation and scoring logic, including vLLM runner
│   ├── README.md
│   ├── aggregate_results.py
│   ├── benchmarks.py
│   ├── eval_utils.py
│   ├── evaluation.py
│   ├── extract_answer.py
│   ├── pass_at_k.py
│   ├── run_eval_vllm.py
│   ├── score.py
│   └── results/                       # pre-computed evaluation results
│       ├── Qwen3-1.7B/                # base model, untuned
│       ├── v1/                        # best SFT multilingual checkpoint
│       │   ├── full/
│       │   ├── mmlu_prox_hard/
│       │   └── validation_set_mnlp/
│       └── final-merged/              # final individual model (TIES merge)
├── merging/                           # merge experiments and PEFT merge helpers
│   └── README.md
├── repro/                             # reproduction workflow and HF sync scripts
│   └── README.md
└── tests/
    └── test_sft_data.py
```

## Evaluation

Evaluation logic lives in `eval/`. It uses a vLLM-based runner that builds chat prompts from raw MCQ inputs, runs batched inference, extracts the final `\boxed{...}` answer, and computes accuracy per dataset and language.

To evaluate a checkpoint:

```bash
python -m multilinguality_model.eval.run_eval_vllm \
  --model_name <checkpoint_path> \
  --val_file <benchmark_jsonl> \
  --output_dir <output_dir>
```

Pre-computed results for the base model, SFT checkpoint, and final merged model are stored in `eval/results/`. See `eval/README.md` for the full interface.

## Training

Training code lives in `train/`. The `repro/` scripts wrap these steps end-to-end; use `train/` directly for custom runs or ablations.

| File | What it does |
|---|---|
| `build_sft_data.py` | Builds the main SFT mixture from processed source datasets with language-balanced sampling. |
| `build_distilled_sft_data.py` | Builds an SFT mixture from distilled reasoning traces (generated but not used in the final model). |
| `build_sft_augmented_mcq.py` | Builds an augmented MCQ mixture with distractor shuffling for data augmentation. |
| `sft.py` | Main LoRA SFT training entry point. |
| `export.py` | Merges a LoRA adapter into the base model and saves the result as a standalone HF checkpoint. |
| `export_full.py` | Full-model export variant (no adapter merging; used for full fine-tune checkpoints). |

## Notes

- Checkpoints are saved under `checkpoints/multilinguality_model/`; the merged best model lands at `checkpoints/multilinguality_model/best_model`.
- Fetched data, SFT mixtures, and benchmarks are saved under `data/multilinguality/`.
- Merge config: `code/multilinguality_model/configs/merging/best_model_ties.yaml`.
- Evaluation requires a GPU pod (vLLM).