from typing import Any, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_messages(prompt: str, task_type: str) -> List[Dict[str, str]]:
    if task_type == "mcq":
        system = (
            "You are solving a multiple-choice question. "
            "You must output exactly one final answer in the format \\boxed{X}, "
            "where X is the letter of the correct option. "
            "Do not output the option text. Do not output explanations. "
            "Do not output anything after the boxed answer. /no_think"
        )
        user = (
            f"{prompt}\n\n"
            "Return only one of: \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}.\n"
            "/no_think"
        )

    elif task_type == "math":
        system = (
            "You are a multilingual mathematical reasoning assistant. "
            "Solve the problem step by step. "
            "Put the final numeric answer inside \\boxed{...}. "
            "Do not put units or explanations inside the box."
        )

    else:
        system = (
            "You are a multilingual evaluation assistant. "
            "Answer the question, putting the final answer inside \\boxed{...}. "
            "Do not put explanations inside the box."
        )
        user = prompt

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
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
    prompt: str,
    task_type: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = build_messages(prompt, task_type)

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)

    return {
        "text": text,
        "n_generated_tokens": generated.shape[-1],
        "hit_max_tokens": generated.shape[-1] >= max_new_tokens,
    }