"""Merge a LoRA checkpoint into Qwen3-1.7B and push it to HuggingFace.

Replaces the base-model push (`push_base.py`) on `cs-552-2026-mystery-machine/math_model`
with the SFT- (or RL-) trained checkpoint of your choice.

Usage:
    # Push a specific checkpoint:
    python -m math_task.training.push_checkpoint \\
        /scratch/<user>/ckpt/math_sft_v1/checkpoint-500

    # Push the final (top-level) adapter:
    python -m math_task.training.push_checkpoint /scratch/<user>/ckpt/math_sft_v1

    # Push to a different repo (e.g. an experiment slot):
    python -m math_task.training.push_checkpoint /scratch/.../checkpoint-500 \\
        --repo cs-552-2026-mystery-machine/math_model_experimental

The script:
  1. Loads the base Qwen3-1.7B
  2. Attaches the LoRA adapter
  3. Merges and unloads (full-weight model)
  4. Saves locally with the team chat template + generation config
  5. Smoke-tests with vLLM (optional via --skip-vllm-check)
  6. Pushes to HF Hub
"""
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
CHAT_TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"
DEFAULT_REPO = "cs-552-2026-mystery-machine/math_model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path,
                    help="LoRA adapter directory (final or checkpoint-N).")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF repo to push to (default: {DEFAULT_REPO}).")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"HF repo or local path to merge the LoRA into. "
                         f"Default {DEFAULT_BASE_MODEL}. For RL-on-top-of-SFT, "
                         f"pass cs-552-2026-mystery-machine/math_model so the "
                         f"RL LoRA merges into the already-merged SFT weights "
                         f"(not bare base, which would lose SFT learnings).")
    ap.add_argument("--local-out", type=Path, default=None,
                    help="Where to save the merged model locally. "
                         "Default: /scratch/<USER>/checkpoints/<ckpt-stem>_merged")
    ap.add_argument("--skip-vllm-check", action="store_true",
                    help="Skip the vLLM smoke test (saves ~1 min).")
    args = ap.parse_args()

    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    template = CHAT_TEMPLATE_PATH.read_text()

    if args.local_out is None:
        user = os.environ.get("USER", "user")
        ckpt_stem = args.checkpoint.name if args.checkpoint.name.startswith("checkpoint-") \
            else args.checkpoint.parent.name + "_final"
        args.local_out = Path(f"/scratch/{user}/checkpoints/{ckpt_stem}_merged")
    args.local_out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- merge
    print(f"[load] base {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    )
    print(f"[load] adapter {args.checkpoint}")
    peft_model = PeftModel.from_pretrained(base, str(args.checkpoint))
    print("[merge] merging LoRA into base...")
    merged = peft_model.merge_and_unload()

    # --------------------------------------------------------------- tokenizer
    print("[tokenizer] loading and applying team chat template")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.chat_template = template

    # Sanity-render. The new template embeds a system prompt and lets the
    # model emit <think> itself, so the prompt now ends with just
    # "<|im_start|>assistant\n". We assert the looser invariant: must end with
    # an open assistant turn, and must include our system-prompt boxed rules.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    assert rendered.rstrip().endswith("<|im_start|>assistant"), (
        f"Chat template doesn't open an assistant turn at the end. "
        f"Got last 80 chars: {rendered[-80:]!r}"
    )
    assert "\\boxed{" in rendered, (
        f"System prompt with \\boxed{{}} rules is missing from the rendered "
        f"chat. Did the template lose its system block? "
        f"First 200 chars: {rendered[:200]!r}"
    )
    print(f"[tokenizer] OK (renders to ...{rendered[-60:]!r})")

    # ------------------------------------------------------- generation config
    gen_cfg = GenerationConfig(
        bos_token_id=151643,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
    )

    # ----------------------------------------------------------------- save
    print(f"[save] {args.local_out}")
    merged.save_pretrained(args.local_out)
    tok.save_pretrained(args.local_out)
    gen_cfg.save_pretrained(args.local_out)
    (args.local_out / "chat_template.jinja").write_text(template)

    # -------------------------------------------------------- vLLM smoke test
    if not args.skip_vllm_check:
        print("[vllm] smoke-test loading the saved checkpoint...")
        from vllm import LLM, SamplingParams
        llm = LLM(str(args.local_out), max_model_len=4096, dtype="float16", seed=42)
        sp = SamplingParams(n=1, max_tokens=2000, temperature=0.7, top_p=0.95, top_k=20)
        out = llm.generate([rendered + "What is 17 * 19? Put the answer in \\boxed{}.\n"], sp)
        text = out[0].outputs[0].text
        print(f"[vllm] finish_reason={out[0].outputs[0].finish_reason}  n_tokens={len(out[0].outputs[0].token_ids)}")
        if "\\boxed{" in text:
            print("[vllm] PASS: model produced \\boxed{...}")
        else:
            print("[vllm] WARN: no \\boxed{} in output — model may have regressed.")
            print(f"[vllm] last 200 chars: ...{text[-200:]}")
        # Free vLLM's GPU memory before pushing.
        del llm

    # ----------------------------------------------------------------- push
    print(f"[push] {args.repo}")
    merged.push_to_hub(args.repo)
    tok.push_to_hub(args.repo)
    gen_cfg.push_to_hub(args.repo)
    # Also push the jinja file explicitly (belt-and-suspenders).
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj=str(args.local_out / "chat_template.jinja"),
        path_in_repo="chat_template.jinja",
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"[done] pushed to https://huggingface.co/{args.repo}")
    print("        Wait for nightly CI (~23:59 UTC) for EVAL_REPORT.md.")


if __name__ == "__main__":
    main()
