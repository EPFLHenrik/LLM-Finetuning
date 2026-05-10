"""vLLM-based batch generation with EXACTLY the CI's prompt construction:

    tokenizer.apply_chat_template(messages, add_generation_prompt=True)

No system prompt, no extra instructions. Anything custom must live inside
the chat_template.jinja itself — that's the only signal CI honors.
"""
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_task.config import EvalConfig


def build_prompts(items: List[Dict], tokenizer) -> List[str]:
    prompts = []
    for item in items:
        messages = [{"role": "user", "content": item["prompt"]}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        )
    return prompts


def run_generation(items: List[Dict], config: EvalConfig) -> List[Dict]:
    """Generate n=8 completions per problem with vLLM. Returns one dict per
    problem with the gold answer and all 8 completions (text + finish_reason
    + token count) preserved for downstream analysis."""
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
    prompts = build_prompts(items, tokenizer)
    outputs = llm.generate(prompts, sp)

    results = []
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
            "gold": item["answer"],
            "completions": completions,
        })
    return results