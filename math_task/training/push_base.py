"""Push the base Qwen3-1.7B with our team's chat template + generation config.

This is the milestone-2 sanity push: it validates the entire HF upload -> CI
loop with NO training risk. Once nightly CI accepts this checkpoint, replace
it with an SFT-trained one via push_checkpoint.py (to be added later).

Pipeline:
  1. Download Qwen3-1.7B base weights + tokenizer
  2. Override tokenizer.chat_template with chat_template.jinja (thinking ON)
  3. Build a GenerationConfig (sampling params we'll use at eval time)
  4. Save everything to /scratch (so vLLM smoke test can use the same dir)
  5. Push the directory to HuggingFace Hub

Run on the cluster (needs HF_TOKEN in env):
    python -m math_task.training.push_base
"""
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ====== EDIT IF NEEDED ======
HF_REPO = "cs-552-2026-mystery-machine/math_model"
GASPAR = "busato"
# ============================

BASE_MODEL = "Qwen/Qwen3-1.7B"
TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"
LOCAL_DIR = Path(f"/scratch/{GASPAR}/checkpoints/math_model_base")


def main() -> None:
    template = TEMPLATE_PATH.read_text()

    print(f"[load] {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="bfloat16"
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.chat_template = template

    # Sanity: template renders correctly before we push.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    print("[verify] rendered template (last 100 chars):")
    print(repr(rendered[-100:]))
    expected_suffix = "<|im_start|>assistant\n<think>\n"
    assert rendered.endswith(expected_suffix), (
        f"Template doesn't end as expected. Got: ...{rendered[-80:]!r}"
    )
    print("[verify] OK\n")

    # Generation config the CI will use (no max_new_tokens here — vLLM controls
    # that at inference time; CI uses 4096 nightly / 16384 final).
    gen_cfg = GenerationConfig(
        bos_token_id=151643,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=20,
    )

    # Save locally first — vLLM smoke test will load from this directory.
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[save] {LOCAL_DIR}")
    model.save_pretrained(LOCAL_DIR)
    tok.save_pretrained(LOCAL_DIR)
    gen_cfg.save_pretrained(LOCAL_DIR)
    # Belt-and-suspenders: also write chat_template.jinja as a sibling file.
    # Some vLLM versions prefer the file; tokenizer_config.json has it embedded.
    (LOCAL_DIR / "chat_template.jinja").write_text(template)

    # Push.
    print(f"[push] {HF_REPO}")
    model.push_to_hub(HF_REPO)
    tok.push_to_hub(HF_REPO)
    gen_cfg.push_to_hub(HF_REPO)
    print("\n[done] Pushed. Wait for nightly CI at 23:59.")
    print(f"        Check: https://huggingface.co/{HF_REPO}/discussions")


if __name__ == "__main__":
    main()
