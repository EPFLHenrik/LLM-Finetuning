Local scripts for preparing a baseline Qwen3 checkpoint and running quick validation.

Setup baseline (downloads files from HF and writes minimal config):

```bash
python scripts/setup_baseline.py --dest models/qwen3_baseline
```

This copies files from `Qwen/Qwen3-1.7B` into `models/qwen3_baseline` and writes a
`generation_config.json` and `tokenizer/chat_template.jinja` if they don't exist.

Quick validation run (uses local checkpoint or fallback to `gpt2`):

```bash
# Use your local prepared checkpoint
python scripts/run_validation.py --model-dir models/qwen3_baseline

# Quick smoke test with a tiny model (fast)
python scripts/run_validation.py --use-tiny
```

Load a local checkpoint, adapt selected files, and save a full new checkpoint.
Edit constants inside `scripts/model_config.py` first (for example `SYSTEM_MESSAGE`,
`GENERATION_CONFIG`, `CONFIG_OVERRIDE`, `TOKENIZER_CONFIG_OVERRIDE`), then run:

```bash
# Apply all built-in overrides from scripts/model_config.py
python scripts/model_config.py \
  --model-path models/qwen3_baseline \
  --output-dir models/qwen3_adapted \
  --cmd all

# Override exactly one artifact type (still saves full model+tokenizer+configs)
python scripts/model_config.py \
  --model-path models/qwen3_baseline \
  --output-dir models/qwen3_adapted_gen \
  --cmd generation_config

# Output dir is always overwritten if it already exists
python scripts/model_config.py --model-path models/qwen3_baseline --output-dir models/qwen3_adapted
```

Notes:
- For real evaluation you must have vLLM-loadable safetensors and the Qwen tokenizer
  with `chat_template.jinja` as required by the course README.
- The `setup_baseline.py` script only copies files from the HF repo — converting
  weights to safetensors or adapting the format for vLLM is outside its scope.
