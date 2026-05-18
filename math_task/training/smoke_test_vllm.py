"""On-cluster vLLM loadability + single-prompt smoke test.

Run AFTER push_base.py has saved a local checkpoint, BEFORE the HF push (or
after, pointed at the HF repo) to confirm vLLM can load it without surprises.

Usage:
    # Test the local dir before pushing:
    python -m math_task.training.smoke_test_vllm \\
        /scratch/<gaspar>/checkpoints/math_model_base

    # Test the pushed HF repo (after pushing):
    python -m math_task.training.smoke_test_vllm \\
        cs-552-2026-mystery-machine/math_model

Pass criteria:
  - vLLM loads the model without errors
  - Generation completes (not all tokens consumed by errors)
  - Output contains \\boxed{...} (warning if not, since base Qwen3 at 4k
    budget with thinking ON often truncates — this is expected for the base
    model and not fatal for milestone 2)
"""
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main(ckpt: str) -> None:
    print(f"[tokenizer] loading from {ckpt}")
    tok = AutoTokenizer.from_pretrained(ckpt)

    rendered = tok.apply_chat_template(
        [{
            "role": "user",
            "content": "What is the smallest prime greater than 100? "
                       "Put the final answer in \\boxed{}.",
        }],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"[tokenizer] rendered prompt (last 80 chars): ...{rendered[-80:]!r}")

    print(f"[vllm] loading {ckpt} at max_model_len=4096")
    llm = LLM(ckpt, max_model_len=4096, dtype="float16", seed=42)

    sp = SamplingParams(
        n=1, max_tokens=3500, temperature=0.7, top_p=0.95, top_k=20, seed=42,
    )
    print("[vllm] generating...")
    out = llm.generate([rendered], sp)
    text = out[0].outputs[0].text
    finish = out[0].outputs[0].finish_reason
    n_tokens = len(out[0].outputs[0].token_ids)

    print(f"\n[output] finish_reason={finish} n_tokens={n_tokens}")
    print("=" * 60)
    print(text)
    print("=" * 60)

    if "\\boxed{" in text:
        print("\n[verify] PASS: output contains \\boxed{...}")
    else:
        print("\n[verify] WARN: no \\boxed{...} in output.")
        print("        Expected for base Qwen3 at 4k with thinking ON.")
        print("        Milestone 2 only requires that the model loads and")
        print("        produces extractable answers on the 5 CI examples,")
        print("        which are likely easier than this prompt.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Local dir or HF repo path")
    args = ap.parse_args()
    main(args.checkpoint)
