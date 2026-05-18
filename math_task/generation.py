"""vLLM-based batch generation with EXACTLY the CI's prompt construction:

    tokenizer.apply_chat_template(messages, add_generation_prompt=True)

No system prompt, no extra instructions. Anything custom must live inside
the chat_template.jinja itself — that's the only signal CI honors.

Local ablations may prepend a string to the user message (PROMPT_VARIANTS in
config.py) and override Qwen3's enable_thinking default. Both are CI-invisible
side channels for probing only.
"""
from typing import List, Dict

from .config import EvalConfig, PROMPT_VARIANTS


def build_prompts(items: List[Dict], tokenizer, config: EvalConfig) -> List[str]:
    prefix = PROMPT_VARIANTS[config.prompt_variant]
    prompts = []
    for item in items:
        user_content = prefix + item["prompt"] if prefix else item["prompt"]
        messages = [{"role": "user", "content": user_content}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=config.thinking_enabled,
            )
        )
    return prompts


def run_generation(items: List[Dict], config: EvalConfig) -> List[Dict]:
    """Generate n completions per problem with vLLM.

    Returns one dict per problem:
        {
          "prompt": ...,
          "answer": ...,
          "completions": [{"text", "finish_reason", "n_tokens"}, ...]
        }

    `finish_reason` and `n_tokens` are kept so we can compute truncation_rate
    and average reasoning length downstream.
    """
    # Lazy imports — keep the rest of the package importable without GPU/vllm.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    llm = LLM(
        model=config.model_id,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        seed=config.seed,
    )
    sp = SamplingParams(
        n=config.n_samples,
        max_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=config.seed,
    )
    prompts = build_prompts(items, tokenizer, config)
    outputs = llm.generate(prompts, sp)

    results: List[Dict] = []
    for item, output in zip(items, outputs):
        completions = [
            {
                "text": c.text,
                "finish_reason": c.finish_reason,
                "n_tokens": len(c.token_ids),
            }
            for c in output.outputs
        ]
        results.append({
            "prompt": item["prompt"],
            "answer": item["answer"],
            "completions": completions,
        })
    return results