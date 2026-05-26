"""Build the SFT v2 training corpus from a curated multi-source mixture.

Two-phase pipeline (each phase is idempotent on disk):

  phase=prep  — pull every dataset, filter the ones that already ship CoT
                traces, and emit a `needs_distill.jsonl` for the ones that
                don't. Stops there. Run `distill_traces.py` on the
                needs-distill list before phase=finalize.

  phase=finalize — read the filtered CoT pools + the distilled-traces JSONL,
                   stratify-sample to the per-source targets in PLAN.md
                   §5.2.B.1, dedupe by normalized problem fingerprint, format
                   every row with `messages=[system, user, assistant]`, and
                   write the final SFT v2 JSONL + a stats sidecar.

Usage:
    python -m math_task.data.build_sft_v2 --phase prep \\
        --workdir /scratch/user/sft_data/v2_workdir
    # ... run distill_traces.py on needs_distill.jsonl ...
    python -m math_task.data.build_sft_v2 --phase finalize \\
        --workdir /scratch/user/sft_data/v2_workdir \\
        --output /scratch/user/sft_data/sft_v2_30k.jsonl

The workdir holds intermediate per-source `*_cot.jsonl` files and the
`needs_distill.jsonl` so re-running phase=prep doesn't re-pull from HF on
every invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .system_prompt import SYSTEM_PROMPT


# =============================================================================
# Per-source configuration
# =============================================================================
# Target counts (PLAN.md §5.2.B.1). Sum ~30K after dedupe.
# Rebalanced 2026-05-26 after first prep run:
#   - NuminaMath-CoT only yielded 3,418 (filter too strict for short formal
#     solutions) — set target to cap available, not the original 10K
#   - LIMO 557, s1K-1.1 388 — set targets to cap available
#   - MATH train (`hendrycks/competition_math` / `lighteval/MATH`) failed to
#     load; College-Math repos don't exist or schema mismatch — drop both
#   - To still reach ~30K, bump OpenR1-Math 8K→9K and OpenMathReasoning 5K→16K
#     (both have plenty of headroom: 45K and 20K available respectively)
#   - distilled:aime_amc bumped to 400 since 100 problems × ~4 keep-top-k traces
TARGETS = {
    "open-r1/OpenR1-Math-220k":     9000,
    "AI-MO/NuminaMath-CoT":         3418,    # cap = available
    "GAIR/LIMO":                     557,    # cap = available
    "simplescaling/s1K-1.1":         388,    # cap = available
    "nvidia/OpenMathReasoning":    16000,
    "distilled:aime_amc":            400,    # 100 problems × ~4 traces each
}

MIN_TOKENS = 1500
MAX_TOKENS = 8000


# =============================================================================
# Tokenizer helpers (Qwen3 length-filtering)
# =============================================================================
def _qwen3_tokenizer():
    from transformers import AutoTokenizer
    if not hasattr(_qwen3_tokenizer, "_tok"):
        _qwen3_tokenizer._tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return _qwen3_tokenizer._tok


def _token_len(text: str) -> int:
    return len(_qwen3_tokenizer().encode(text, add_special_tokens=False))


# =============================================================================
# Answer extraction + structural filters
# =============================================================================
def _extract_boxed(text: str) -> Optional[str]:
    from evaluate.extract_answer import extract_boxed_answer
    return extract_boxed_answer(text, strip_double_curly_brace=True)


def _has_think_block(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def _length_ok(text: str) -> bool:
    n = _token_len(text)
    return MIN_TOKENS <= n <= MAX_TOKENS


def _is_clean_trace(text: str) -> bool:
    """A trace is keepable iff it has <think>…</think>, an extractable boxed,
    and falls within the length window."""
    if not text or not _has_think_block(text):
        return False
    if _extract_boxed(text) is None:
        return False
    return _length_ok(text)


def _wrap_as_think_and_box(solution: str, gold: str) -> str:
    """Synthesize a `<think>{solution}</think>\\n\\n\\boxed{gold}` trace if
    the solution doesn't already have <think> markers. Used for datasets
    whose 'solutions' are reasoning text without R1-style think tags
    (LIMO, s1K-1.1, NuminaMath-CoT).

    If the wrapped solution already ends with a `\\boxed{}`, we still append
    `\\boxed{gold}` at the very end so that `extract_boxed_answer` (which
    picks the LAST boxed) matches the gold reliably.
    """
    s = (solution or "").strip()
    if not s or not gold:
        return s
    if "<think>" not in s:
        s = f"<think>\n{s}\n</think>\n\n\\boxed{{{gold}}}"
    return s


# =============================================================================
# Problem fingerprinting (for dedupe across sources)
# =============================================================================
_WS = re.compile(r"\s+")
_LATEX_CMDS = re.compile(r"\\[a-zA-Z]+\*?")  # \frac, \boxed, \begin, ...


def fingerprint(problem: str) -> str:
    """Normalize a problem into a stable hash so the same prompt under two
    sources doesn't enter the corpus twice."""
    s = problem.lower()
    s = _LATEX_CMDS.sub("", s)
    s = _WS.sub(" ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Per-source loaders
# =============================================================================
# Each loader returns a list of {"problem": str, "assistant": str, "source": str}
# Loaders are lazy so phase=finalize doesn't re-pull from HF.

def load_openr1_math(progress_every: int = 5000) -> List[Dict]:
    from datasets import load_dataset
    print("[load] open-r1/OpenR1-Math-220k  split=default")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    out: List[Dict] = []
    dropped = Counter()
    for i, row in enumerate(ds):
        if i and i % progress_every == 0:
            print(f"  [openr1] {i}/{len(ds)} kept={len(out)}")
        gens = row.get("generations") or []
        flags = row.get("correctness_math_verify") or []
        gen = next((g for g, ok in zip(gens, flags) if ok), None)
        if gen is None:
            dropped["no_correct_gen"] += 1
            continue
        if not _is_clean_trace(gen):
            dropped["unclean"] += 1
            continue
        problem = (row.get("problem") or "").strip()
        if not problem:
            dropped["empty_problem"] += 1
            continue
        out.append({"problem": problem, "assistant": gen, "source": "open-r1/OpenR1-Math-220k"})
    print(f"  [openr1] kept={len(out)} dropped={dict(dropped)}")
    return out


def load_numina_cot(progress_every: int = 10000) -> List[Dict]:
    """`AI-MO/NuminaMath-CoT` ships step-by-step solutions without R1 `<think>`
    markers. We extract the gold answer from the solution (or `answer` field
    when present), then synthetically wrap the solution as
    `<think>{solution}</think>\\n\\n\\boxed{gold}` so the model learns to put
    its reasoning inside a think block during SFT v2."""
    from datasets import load_dataset
    print("[load] AI-MO/NuminaMath-CoT  split=train")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    out: List[Dict] = []
    dropped = Counter()
    for i, row in enumerate(ds):
        if i and i % progress_every == 0:
            print(f"  [numina-cot] {i}/{len(ds)} kept={len(out)}")
        problem = (row.get("problem") or "").strip()
        raw_solution = row.get("solution") or row.get("reasoning") or ""
        if not problem or not raw_solution:
            dropped["empty"] += 1
            continue
        # Try to find a gold answer: either embedded as \boxed{} in the
        # solution, or in an `answer` field if present.
        gold = _extract_boxed(raw_solution) or str(row.get("answer") or "").strip()
        if not gold:
            dropped["no_gold"] += 1
            continue
        wrapped = _wrap_as_think_and_box(raw_solution, gold)
        if not _is_clean_trace(wrapped):
            dropped["unclean"] += 1
            continue
        out.append({"problem": problem, "assistant": wrapped, "source": "AI-MO/NuminaMath-CoT"})
    print(f"  [numina-cot] kept={len(out)} dropped={dict(dropped)}")
    return out


def load_limo() -> List[Dict]:
    """LIMO ships R1-distilled solutions as plain reasoning text (no `<think>`
    markers). Same fix as NuminaMath-CoT: extract gold + wrap synthetically."""
    from datasets import load_dataset
    print("[load] GAIR/LIMO  split=train")
    ds = load_dataset("GAIR/LIMO", split="train")
    out: List[Dict] = []
    dropped = Counter()
    for row in ds:
        problem = (row.get("question") or row.get("problem") or "").strip()
        raw_solution = row.get("solution") or row.get("response") or ""
        if not problem or not raw_solution:
            dropped["empty"] += 1
            continue
        gold = _extract_boxed(raw_solution) or str(row.get("answer") or "").strip()
        if not gold:
            dropped["no_gold"] += 1
            continue
        wrapped = _wrap_as_think_and_box(raw_solution, gold)
        if not _is_clean_trace(wrapped):
            dropped["unclean"] += 1
            continue
        out.append({"problem": problem, "assistant": wrapped, "source": "GAIR/LIMO"})
    print(f"  [LIMO] kept={len(out)} dropped={dict(dropped)}")
    return out


def load_s1k() -> List[Dict]:
    from datasets import load_dataset
    print("[load] simplescaling/s1K-1.1  split=train")
    ds = load_dataset("simplescaling/s1K-1.1", split="train")
    out: List[Dict] = []
    dropped = Counter()
    for row in ds:
        problem = (row.get("question") or row.get("problem") or "").strip()
        # s1K-1.1 stores the reasoning under various keys depending on revision.
        solution = (
            row.get("deepseek_thinking_trajectory")
            or row.get("attempt")
            or row.get("solution")
            or ""
        )
        if not problem or not solution:
            dropped["empty"] += 1
            continue
        # s1K traces are raw thinking; wrap if not already wrapped.
        if "<think>" not in solution:
            # Append the gold answer as a boxed conclusion if we have one.
            gold = row.get("answer") or row.get("solution") or ""
            solution = f"<think>\n{solution}\n</think>\n\n\\boxed{{{gold}}}"
        if not _is_clean_trace(solution):
            dropped["unclean"] += 1
            continue
        out.append({"problem": problem, "assistant": solution, "source": "simplescaling/s1K-1.1"})
    print(f"  [s1K] kept={len(out)} dropped={dict(dropped)}")
    return out


# OpenMathReasoning is huge (3.2M rows). We only need ~5K (TARGETS) so even
# this 4× safety cap finishes in ~10-15 min instead of ~6h.
OPENMATH_MAX_KEPT = int(os.environ.get("OPENMATH_MAX_KEPT", "20000"))


def load_openmath_reasoning(progress_every: int = 10000) -> List[Dict]:
    """`nvidia/OpenMathReasoning` has multiple SPLITS (`cot`, `tir`,
    `genselect`, `additional_problems`), not configs. We use `cot` — the
    R1-style chain-of-thought split (~3.2M rows). Solutions may or may not
    ship with `<think>` markers; wrap with `_wrap_as_think_and_box` when
    needed (using `expected_answer` for the gold).

    Early-stops at OPENMATH_MAX_KEPT (env-overridable, default 20K) so we
    don't burn 5h iterating the full 3.2M when our target is 5K rows.
    """
    from datasets import load_dataset
    print(f"[load] nvidia/OpenMathReasoning  split=cot  (early-stop at {OPENMATH_MAX_KEPT} kept)")
    ds = load_dataset("nvidia/OpenMathReasoning", split="cot")
    out: List[Dict] = []
    dropped = Counter()
    for i, row in enumerate(ds):
        if i and i % progress_every == 0:
            print(f"  [openmath] {i}/{len(ds)} kept={len(out)}")
        if len(out) >= OPENMATH_MAX_KEPT:
            print(f"  [openmath] hit OPENMATH_MAX_KEPT={OPENMATH_MAX_KEPT}, stopping early")
            break
        problem = (row.get("problem") or row.get("question") or "").strip()
        raw_solution = (
            row.get("generated_solution")
            or row.get("output")
            or row.get("solution")
            or ""
        )
        if not problem or not raw_solution:
            dropped["empty"] += 1
            continue
        # Try the answer field NVIDIA ships, then fall back to extracting from solution.
        gold = (
            str(row.get("expected_answer") or "").strip()
            or _extract_boxed(raw_solution)
            or ""
        )
        # If the solution already has <think>...</think> + boxed, use as-is;
        # otherwise wrap with the gold (when we have one).
        if "<think>" in raw_solution and _extract_boxed(raw_solution) is not None:
            candidate = raw_solution
        elif gold:
            candidate = _wrap_as_think_and_box(raw_solution, gold)
        else:
            dropped["no_gold"] += 1
            continue
        if not _is_clean_trace(candidate):
            dropped["unclean"] += 1
            continue
        out.append({"problem": problem, "assistant": candidate, "source": "nvidia/OpenMathReasoning"})
    print(f"  [openmath] kept={len(out)} dropped={dict(dropped)}")
    return out


# Per-source LOADER table
LOADERS: Dict[str, Callable[[], List[Dict]]] = {
    "open-r1/OpenR1-Math-220k": load_openr1_math,
    "AI-MO/NuminaMath-CoT":     load_numina_cot,
    "GAIR/LIMO":                load_limo,
    "simplescaling/s1K-1.1":    load_s1k,
    "nvidia/OpenMathReasoning": load_openmath_reasoning,
}


# =============================================================================
# Loaders for the no-CoT problem sources (write `needs_distill.jsonl`)
# =============================================================================
def collect_needs_distill_problems() -> List[Dict]:
    """Pull problems from datasets that don't ship CoT traces. Each row is
    {problem, gold, source}. distill_traces.py turns these into traces."""
    from datasets import load_dataset
    out: List[Dict] = []

    # MATH train (Hendrycks competition_math). The trickiest part is which
    # repo on HF is mirrored: hendrycks/competition_math or lighteval/MATH.
    for repo in ("hendrycks/competition_math", "lighteval/MATH"):
        try:
            print(f"[load no-CoT] {repo} split=train")
            ds = load_dataset(repo, split="train", trust_remote_code=False)
            for row in ds:
                problem = (row.get("problem") or "").strip()
                solution = row.get("solution") or ""
                gold = _extract_boxed(solution) or ""
                if problem and gold:
                    out.append({"problem": problem, "gold": gold,
                                "source": f"distilled:competition_math"})
            break
        except Exception as e:
            print(f"  [warn] failed to load {repo}: {e}")

    # AIME 2024
    try:
        print("[load no-CoT] Maxwell-Jia/AIME_2024 split=train")
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for row in ds:
            problem = (row.get("Problem") or row.get("problem") or "").strip()
            gold = str(row.get("Answer") or row.get("answer") or "").strip()
            if problem and gold:
                out.append({"problem": problem, "gold": gold, "source": "distilled:aime_amc"})
    except Exception as e:
        print(f"  [warn] AIME_2024 unavailable: {e}")

    # AIME 2025 (community repo, name varies)
    for repo in ("MathArena/aime_2025", "yentinglin/aime_2025"):
        try:
            print(f"[load no-CoT] {repo} split=train")
            ds = load_dataset(repo, split="train")
            for row in ds:
                problem = (row.get("problem") or row.get("Problem") or "").strip()
                gold = str(row.get("answer") or row.get("Answer") or "").strip()
                if problem and gold:
                    out.append({"problem": problem, "gold": gold, "source": "distilled:aime_amc"})
            break
        except Exception as e:
            print(f"  [warn] failed to load {repo}: {e}")

    # AMC23
    try:
        print("[load no-CoT] math-ai/amc23 split=test")
        ds = load_dataset("math-ai/amc23", split="test")
        for row in ds:
            problem = (row.get("question") or row.get("problem") or "").strip()
            gold = str(row.get("answer") or "").strip()
            if problem and gold:
                out.append({"problem": problem, "gold": gold, "source": "distilled:aime_amc"})
    except Exception as e:
        print(f"  [warn] AMC23 unavailable: {e}")

    # College Math — there are several repos; try a popular one.
    for repo in ("Mathwell/College-Math", "TIGER-Lab/MathInstruct"):
        try:
            print(f"[load no-CoT] {repo}")
            ds = load_dataset(repo, split="train")
            n_before = len(out)
            for row in ds:
                problem = (row.get("problem") or row.get("question") or "").strip()
                gold = str(row.get("answer") or row.get("gold") or "").strip()
                if problem and gold:
                    out.append({"problem": problem, "gold": gold, "source": "distilled:college_math"})
            print(f"  [{repo}] +{len(out) - n_before}")
            break
        except Exception as e:
            print(f"  [warn] {repo} unavailable: {e}")

    print(f"[no-CoT] total problems needing distillation: {len(out)}")
    return out


# =============================================================================
# Phase: prep — pull, filter, write intermediates
# =============================================================================
def phase_prep(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)

    for source, loader in LOADERS.items():
        out_path = workdir / (source.replace("/", "_") + "_cot.jsonl")
        if out_path.exists():
            print(f"[prep] {source}: cached → {out_path}")
            continue
        items = loader()
        with open(out_path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[prep] {source}: wrote {len(items)} → {out_path}")

    needs_path = workdir / "needs_distill.jsonl"
    if needs_path.exists():
        print(f"[prep] needs_distill cached → {needs_path}")
    else:
        items = collect_needs_distill_problems()
        with open(needs_path, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[prep] wrote {len(items)} → {needs_path}")
        print(f"\n[next] Run distillation on the needs-distill set, e.g.:")
        print(f"  python -m math_task.data.distill_traces \\")
        print(f"    --input {needs_path} \\")
        print(f"    --output {workdir / 'distilled.jsonl'}")


# =============================================================================
# Phase: finalize — sample, dedupe, format, write final JSONL
# =============================================================================
def phase_finalize(workdir: Path, output: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    by_source: Dict[str, List[Dict]] = {}

    # Load CoT pools written by phase_prep.
    for source in LOADERS:
        path = workdir / (source.replace("/", "_") + "_cot.jsonl")
        if not path.exists():
            print(f"[finalize] MISSING {path} — re-run --phase prep")
            return
        with open(path) as f:
            by_source[source] = [json.loads(l) for l in f if l.strip()]

    # Load distilled traces (one row per problem, possibly multiple traces).
    distilled_path = workdir / "distilled.jsonl"
    if not distilled_path.exists():
        print(f"[finalize] MISSING {distilled_path}")
        print(f"[finalize] run distill_traces.py first, see prep output for the exact command.")
        return

    with open(distilled_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            traces = rec.get("traces") or []
            source = rec.get("source", "distilled:unknown")
            if not traces:
                continue
            for trace in traces:
                if _is_clean_trace(trace):
                    by_source.setdefault(source, []).append({
                        "problem": rec["problem"],
                        "assistant": trace,
                        "source": source,
                    })

    # Stratified sample to TARGETS.
    sampled: List[Dict] = []
    for source, target in TARGETS.items():
        pool = by_source.get(source, [])
        rng.shuffle(pool)
        take = min(target, len(pool))
        if take < target:
            print(f"[finalize] WARN {source}: wanted {target}, only have {len(pool)}")
        sampled.extend(pool[:take])

    # Dedupe by problem fingerprint, preserving first occurrence (which is
    # likely the highest-quality source since we ordered TARGETS by quality).
    seen_fp = set()
    deduped: List[Dict] = []
    n_dupes = 0
    for it in sampled:
        fp = fingerprint(it["problem"])
        if fp in seen_fp:
            n_dupes += 1
            continue
        seen_fp.add(fp)
        deduped.append(it)
    print(f"[finalize] dedupe: {len(sampled)} → {len(deduped)} ({n_dupes} dupes)")

    # Format every row with the system prompt baked in.
    rng.shuffle(deduped)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for it in deduped:
            row = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": it["problem"]},
                    {"role": "assistant", "content": it["assistant"]},
                ],
                "source": it["source"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[finalize] wrote {len(deduped)} rows → {output}")

    # Sidecar stats file.
    stats_path = output.with_name(output.stem + "_stats.json")
    per_source = Counter(it["source"] for it in deduped)
    lens = [_token_len(it["assistant"]) for it in deduped[:500]]
    lens.sort()
    stats = {
        "n_total": len(deduped),
        "per_source": dict(per_source),
        "target_per_source": TARGETS,
        "n_dupes_dropped": n_dupes,
        "assistant_token_len_sample500": {
            "p50": lens[len(lens) // 2] if lens else 0,
            "p90": lens[int(len(lens) * 0.9)] if lens else 0,
            "p99": lens[int(len(lens) * 0.99)] if lens else 0,
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[finalize] stats → {stats_path}")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "finalize"], required=True)
    ap.add_argument("--workdir", type=Path, required=True,
                    help="Directory for per-source intermediate JSONLs.")
    ap.add_argument("--output", type=Path, default=None,
                    help="(phase=finalize) Final SFT v2 JSONL output path.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.phase == "prep":
        phase_prep(args.workdir)
    else:
        if args.output is None:
            ap.error("--output is required for phase=finalize")
        phase_finalize(args.workdir, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
