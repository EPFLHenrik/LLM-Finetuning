"""Post-process filter output: prepend an explicit system message to each row.

ms-swift's `swift rlhf` renders prompts through its own registered chat template
(e.g. "qwen3"), NOT through the chat_template.jinja that lives on the model's
HF tokenizer. So the 4580-byte system-prompt template we pushed to
cs-552-2026-mystery-machine/math_model would be silently ignored during RL
rollouts — the model would see Qwen3's default formatting instead of our
boxed-rules system prompt.

Workaround: add an explicit {"role":"system","content":SYSTEM_PROMPT} entry to
each row's `messages`. ms-swift's qwen3 template (and every reasonable
template) honors an explicit system message. The rollouts then see the same
system prompt that CI inference will see.

Usage:
    python -m math_task.data.add_system_prompt \\
        --input  /scratch/user/rl_data/deepscaler_filtered.jsonl \\
        --output /scratch/user/rl_data/deepscaler_filtered_sft.jsonl

Idempotent: rows that already have a system message at index 0 are passed
through unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from .system_prompt import SYSTEM_PROMPT  # single source of truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="Filter output JSONL (rows with `messages` and `solution`).")
    ap.add_argument("--output", type=Path, required=True,
                    help="JSONL with system message prepended to each row.")
    args = ap.parse_args()

    n_in = 0
    n_already = 0
    n_added = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f_in, open(args.output, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            messages = rec.get("messages") or []
            if messages and messages[0].get("role") == "system":
                n_already += 1
            else:
                rec["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages,
                ]
                n_added += 1
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[add_system_prompt] in={n_in}  added={n_added}  already_had={n_already}")
    print(f"[add_system_prompt] wrote {args.output}")


if __name__ == "__main__":
    main()
