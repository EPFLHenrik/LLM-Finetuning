"""Training utilities for the math individual model.

Modules:
  - chat_template.jinja  : Qwen3 chat template with thinking ON, no system prompt
  - verify_template.py   : local sanity check (no GPU needed)
  - push_base.py         : push base Qwen3-1.7B + our template + gen config to HF
  - smoke_test_vllm.py   : on-cluster vLLM loadability check
"""
