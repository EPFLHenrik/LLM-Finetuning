"""multilingual/generation.py

Generation follows the official CI evaluation contract:
  - System prompt: SYSTEM_PROMPT from config (the only thing we control)
  - User turn:     the raw question string, unmodified
  - Think mode:    always enabled (no /no_think suffix)
  - Sampling:      greedy by default for reproducibility; set do_sample=True
                   and pass temperature/top_p to match your generation_config
"""

from typing import Any, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual.config import SYSTEM_PROMPT


def build_messages(raw_question: str) -> List[Dict[str, str]]:
    """
    Build the chat messages for a single question.

    The user turn is the raw question exactly as it would arrive during the
    official eval — no extra instructions injected here.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": raw_question},
    ]


def load_model_and_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    prompt: str,           # raw question string — no extra instructions
    task_type: str,        # kept for API compatibility; no longer drives prompt format
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Generate an answer for a single question.

    Args:
        prompt:         Raw question text (user turn only).
        task_type:      'mcq' or 'math' — used by the caller for token budget,
                        but no longer changes the prompt construction.
        max_new_tokens: Token budget for this call.
        do_sample:      Set True to use temperature sampling (matches the
                        generation_config you'll submit to HF).
    """
    messages = build_messages(prompt)

    # apply_chat_template adds the <think> trigger automatically for Qwen3
    # when enable_thinking=True (default). We do NOT pass /no_think.
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3 tokenizer respects this kwarg to keep thinking enabled
        enable_thinking=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        generate_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **generate_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)

    return {
        "text": text,
        "n_generated_tokens": int(generated.shape[-1]),
        "hit_max_tokens": int(generated.shape[-1]) >= max_new_tokens,
    }