"""Test chat templates for both inference and training compatibility.

Usage:
    # Test inference template from export output dir
    python test_chat_template.py --inference_dir path/to/export

    # Test training template from adapter/checkpoint dir
    python test_chat_template.py --training_dir path/to/best_adapter

    # Test both at once
    python test_chat_template.py \
        --inference_dir path/to/export \
        --training_dir  path/to/best_adapter
"""

import argparse
import os
from transformers import AutoTokenizer

SAMPLE = [
    {"role": "user", "content": "What is the capital of France?\n\nA) London\nB) Paris\nC) Berlin\nD) Rome"},
]
WITH_SYSTEM = [
    {"role": "system", "content": "You are a careful reasoning assistant."},
] + SAMPLE
TRAINING_MSGS = SAMPLE + [{"role": "assistant", "content": "\\boxed{B}"}]


def test_inference(tok, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"INFERENCE TEST — {label}")
    print('='*60)

    print("\n--- 1. No system msg (default system prompt should appear) ---")
    rendered = tok.apply_chat_template(
        SAMPLE, tokenize=False, add_generation_prompt=True,
    )
    print(rendered)
    if "<|im_start|>system" in rendered:
        print("✅ Default system prompt injected")
    else:
        print("❌ No system prompt — CI evaluation may fail!")

    print("\n--- 2. Explicit system msg ---")
    rendered_sys = tok.apply_chat_template(
        WITH_SYSTEM, tokenize=False, add_generation_prompt=True,
    )
    print(rendered_sys)

    print("\n--- 3. Thinking mode check ---")
    if "enable_thinking" in (tok.chat_template or ""):
        print("✅ enable_thinking present in template")
    else:
        print("⚠️  enable_thinking not found in template")


def test_training(tok, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"TRAINING TEST — {label}")
    print('='*60)

    print("\n--- 1. Training render ---")
    rendered = tok.apply_chat_template(
        TRAINING_MSGS, tokenize=False, add_generation_prompt=False,
    )
    print(rendered)

    print("\n--- 2. Assistant token mask ---")
    try:
        out = tok.apply_chat_template(
            TRAINING_MSGS,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        tokens = tok.convert_ids_to_tokens(out["input_ids"])
        mask   = out["assistant_masks"]
        train_tokens = [t for t, m in zip(tokens, mask) if m]
        if all(m == 0 for m in mask):
            print("❌ All masks zero — {% generation %} markers missing!")
        else:
            print(f"✅ {len(train_tokens)} tokens marked for training:")
            for t in train_tokens:
                print(f"   TRAIN → {t}")
    except Exception as e:
        print(f"❌ assistant_masks failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_dir", type=str, default=None,
                        help="Export output dir to test for inference.")
    parser.add_argument("--training_dir",  type=str, default=None,
                        help="Adapter/checkpoint dir to test for training.")
    return parser.parse_args()


def main():
    args     = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")

    if not args.inference_dir and not args.training_dir:
        print("Pass --inference_dir and/or --training_dir.")
        return

    if args.inference_dir:
        tok = AutoTokenizer.from_pretrained(args.inference_dir, token=hf_token)
        test_inference(tok, args.inference_dir)

    if args.training_dir:
        from trl.chat_template_utils import get_training_chat_template
        tok = AutoTokenizer.from_pretrained(args.training_dir, token=hf_token)
        training_template = get_training_chat_template(tok)
        if training_template:
            tok.chat_template = training_template
            print(f"\n[info] TRL training template loaded for {args.training_dir}")
        else:
            print(f"\n[warn] No TRL training template found — using checkpoint template")
        test_training(tok, args.training_dir)


if __name__ == "__main__":
    main()