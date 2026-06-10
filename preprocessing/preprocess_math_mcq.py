#!/usr/bin/env python3
"""Preprocess MATH train.jsonl into 4-choice MCQ format.

For each row:
  - Keep the `problem` field as-is.
  - Extract the gold answer from \\boxed{} in `solution`.
  - Classify the answer type; generate 3 plausible same-type distractors.
  - Randomly place the correct answer at A, B, C, or D.
  - Discard rows where extraction or type classification fails.

Output per-line JSON:
  {"prompt": "<problem>\\nChoices:\\nA) ...\\nB) ...\\nC) ...\\nD) ...", "answer": "<letter>"}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate.extract_answer import extract_boxed_answer

LETTERS = "ABCD"


# ---------------------------------------------------------------------------
# Distractor generators
# ---------------------------------------------------------------------------

def _sample3(pool: list, exclude: set) -> list | None:
    available = [x for x in pool if x not in exclude]
    if len(available) < 3:
        return None
    return random.sample(available, 3)


def _int_distractors(n: int) -> list[str] | None:
    s: set[int] = set()
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]:
        s.add(n + d)
        s.add(n - d)
    for scale in [2, 3, 4]:
        s.add(n * scale)
        if n != 0 and n % scale == 0:
            s.add(n // scale)
    if abs(n) > 100:
        for pct in [0.05, 0.10, 0.15, 0.20, -0.05, -0.10, -0.15, -0.20]:
            s.add(round(n * (1 + pct)))
    s.discard(n)
    chosen = _sample3(sorted(s), {n})
    return [str(x) for x in chosen] if chosen else None


def _frac_distractors(p: int, q: int, sign: str = "", cmd: str = "frac") -> list[str] | None:
    s: set[tuple[int, int]] = set()
    for dp in [-2, -1, 1, 2, 3]:
        s.add((p + dp, q))
    for dq in [-2, -1, 1, 2]:
        nq = q + dq
        if nq > 0:
            s.add((p, nq))
    if p > 0:
        s.add((q, p))
    s.discard((p, q))
    pool = [(pp, qq) for pp, qq in s if qq > 0]
    chosen = _sample3(pool, {(p, q)})
    return [f"{sign}\\{cmd}{{{pp}}}{{{qq}}}" for pp, qq in chosen] if chosen else None


def _decimal_distractors(val: float, orig: str) -> list[str] | None:
    dec_places = len(orig.split(".")[-1]) if "." in orig else 2
    fmt = f"{{:.{dec_places}f}}"
    orig_str = fmt.format(val)
    s: set[str] = set()
    for d in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        s.add(fmt.format(round(val + d, dec_places)))
        s.add(fmt.format(round(val - d, dec_places)))
    s.discard(orig_str)
    return _sample3(sorted(s), {orig_str})


def _sqrt_distractors(coeff: int, val: int) -> list[str] | None:
    s: set[tuple[int, int]] = set()
    for dc in [-2, -1, 1, 2, 3]:
        nc = coeff + dc
        if nc > 0:
            s.add((nc, val))
    for dv in [-3, -2, -1, 1, 2, 3, 5, 7]:
        nv = val + dv
        if nv > 1:
            s.add((coeff, nv))
    s.discard((coeff, val))
    pool = [(c, v) for c, v in s if c > 0 and v > 0]

    def fmt(c: int, v: int) -> str:
        prefix = str(c) if c != 1 else ""
        return f"{prefix}\\sqrt{{{v}}}"

    chosen = _sample3(pool, {(coeff, val)})
    return [fmt(c, v) for c, v in chosen] if chosen else None


_ANGLES = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 210, 240, 270, 300, 315, 330, 360]


def _degree_distractors(deg: int) -> list[str] | None:
    pool = [a for a in _ANGLES if a != deg]
    nearby = sorted(pool, key=lambda x: abs(x - deg))[:8]
    if len(nearby) < 3:
        nearby = [deg + d for d in [-45, -30, -15, 15, 30, 45] if deg + d > 0 and deg + d != deg]
    if len(nearby) < 3:
        return None
    chosen = random.sample(nearby, 3)
    return [f"{d}^\\circ" for d in chosen]


def _interval_distractors(a: int, b: int, lb: str, rb: str, sep: str = ",") -> list[str] | None:
    s: set[tuple[int, int]] = set()
    for da in [-3, -2, -1, 1, 2, 3]:
        for db in [-3, -2, -1, 1, 2, 3]:
            na, nb = a + da, b + db
            if na < nb:
                s.add((na, nb))
    s.discard((a, b))
    chosen = _sample3(sorted(s), {(a, b)})
    return [f"{lb}{x}{sep}{y}{rb}" for x, y in chosen] if chosen else None


def _pair_distractors(a: int, b: int, lb: str = "", rb: str = "", sep: str = ",") -> list[str] | None:
    s: set[tuple[int, int]] = set()
    for da in [-2, -1, 0, 1, 2]:
        for db in [-2, -1, 0, 1, 2]:
            if da != 0 or db != 0:
                s.add((a + da, b + db))
    s.discard((a, b))
    chosen = _sample3(sorted(s), {(a, b)})
    return [f"{lb}{x}{sep}{y}{rb}" for x, y in chosen] if chosen else None


# ---------------------------------------------------------------------------
# Classify and dispatch
# ---------------------------------------------------------------------------

def _classify_and_distract(answer: str) -> list[str] | None:
    ans = answer.strip()

    # Plain integer (possibly negative)
    if re.fullmatch(r"-?\d+", ans):
        return _int_distractors(int(ans))

    # Fractions: \frac or \dfrac, braced or single-digit shorthand, optional leading minus
    # Braced form: -?\(d?)frac{p}{q}
    m = re.fullmatch(r"(-?)\\(d?frac)\{(-?\d+)\}\{(\d+)\}", ans)
    if m and int(m.group(4)) > 0:
        return _frac_distractors(int(m.group(3)), int(m.group(4)), sign=m.group(1), cmd=m.group(2))
    # Shorthand form: -?\(d?)frac pq  (single-digit args, no braces)
    m = re.fullmatch(r"(-?)\\(d?frac)(\d)(\d)", ans)
    if m and int(m.group(4)) > 0:
        return _frac_distractors(int(m.group(3)), int(m.group(4)), sign=m.group(1), cmd=m.group(2))

    # Decimal (no LaTeX)
    if re.fullmatch(r"-?\d+\.\d+", ans):
        return _decimal_distractors(float(ans), ans)

    # k\sqrt{n} or k\sqrt n — optional leading minus, optional coefficient
    m = re.fullmatch(r"(-?)(\d+)?\\sqrt\{?(\d+)\}?", ans)
    if m:
        sign = m.group(1)
        coeff = int(m.group(2)) if m.group(2) else 1
        val = int(m.group(3))
        dists = _sqrt_distractors(coeff, val)
        return [f"{sign}{d}" for d in dists] if dists else None

    # Degree: n^\circ or n^{\circ}
    m = re.fullmatch(r"(\d+)\^?\{?\\circ\}?", ans)
    if m:
        return _degree_distractors(int(m.group(1)))

    # Closed interval [a,b] — always a valid interval (a < b)
    m = re.fullmatch(r"\[(-?\d+)(,\s*)(-?\d+)\]", ans)
    if m:
        a, sep, b = int(m.group(1)), m.group(2), int(m.group(3))
        if a < b:
            return _interval_distractors(a, b, "[", "]", sep)

    # (a,b) — interval if a < b, coordinate pair otherwise
    m = re.fullmatch(r"\((-?\d+)(,\s*)(-?\d+)\)", ans)
    if m:
        a, sep, b = int(m.group(1)), m.group(2), int(m.group(3))
        if a < b:
            return _interval_distractors(a, b, "(", ")", sep)
        else:
            return _pair_distractors(a, b, lb="(", rb=")", sep=sep)

    # Bare comma-separated integer pair: a,b  or  a, b
    m = re.fullmatch(r"(-?\d+)(,\s*)(-?\d+)", ans)
    if m:
        a, sep, b = int(m.group(1)), m.group(2), int(m.group(3))
        return _pair_distractors(a, b, sep=sep)

    return None


# ---------------------------------------------------------------------------
# MCQ building
# ---------------------------------------------------------------------------

def _build_prompt(problem: str, choices: list[tuple[str, str]]) -> str:
    parts = [problem.strip(), "Choices:"]
    for label, text in choices:
        parts.append(f"{label}) {text}")
    return "\n".join(parts)


def process_row(row: dict) -> dict | None:
    problem = str(row.get("problem", "")).strip()
    solution = str(row.get("solution", "")).strip()
    if not problem or not solution:
        return None

    answer = extract_boxed_answer(solution)
    if not answer:
        return None
    answer = answer.strip()

    distractors = _classify_and_distract(answer)
    if distractors is None:
        return None

    correct_pos = random.randint(0, 3)
    choices_texts = list(distractors)
    choices_texts.insert(correct_pos, answer)
    choices = [(LETTERS[i], choices_texts[i]) for i in range(4)]
    return {
        "prompt": _build_prompt(problem, choices),
        "answer": LETTERS[correct_pos],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess MATH train JSONL into MCQ format")
    parser.add_argument("input", type=Path, help="Input JSONL path")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output JSONL path (default: <input stem>.mcq.jsonl next to input)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.mcq.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = written = discarded = 0
    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{input_path}:{lineno}: invalid JSON: {exc}") from exc
            out = process_row(row)
            if out is None:
                discarded += 1
                continue
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Read      {total:>6} rows")
    print(f"Written   {written:>6} rows  →  {output_path}")
    print(f"Discarded {discarded:>6} rows (unrecognized answer type or no \\boxed{{}})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
