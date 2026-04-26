"""
Prompt loader — 20 items × 5 tasks = 100 prompts.

Tries HuggingFace `datasets` first (real benchmark data, deterministic sample).
Falls back to a small bundled set in `prompts_fallback.json` if `datasets`
isn't installed or there's no internet at run time.

Cached to `benchmark/data/prompts.json` after first load.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Optional

HERE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(DATA_DIR, "prompts.json")
FALLBACK   = os.path.join(HERE, "prompts_fallback.json")

TASKS      = ["humaneval", "gsm8k", "triviaqa", "xsum", "rag"]
PER_TASK   = 4
SEED       = 42


@dataclass
class Prompt:
    """One benchmark item."""
    id:        str
    task:      str
    prompt:    str
    reference: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _try_load_from_hf() -> Optional[list[Prompt]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    rng = random.Random(SEED)
    out: list[Prompt] = []

    try:
        ds = load_dataset("openai_humaneval", split="test")
        idxs = list(range(len(ds)))[:PER_TASK]
        for i in idxs:
            row = ds[i]
            out.append(Prompt(
                id=f"humaneval/{row['task_id']}",
                task="humaneval",
                prompt=(
                    "Complete the following Python function. Return ONLY the "
                    "function body and any helper code, no markdown.\n\n" + row["prompt"]
                ),
                reference={
                    "entry_point": row["entry_point"],
                    "test":        row["test"],
                    "prompt":      row["prompt"],
                },
            ))
    except Exception as exc:
        print(f"[prompts] humaneval skipped via HF: {exc}")

    try:
        ds = load_dataset("gsm8k", "main", split="test")
        idxs = rng.sample(range(len(ds)), PER_TASK)
        for n, i in enumerate(idxs):
            row = ds[i]
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            out.append(Prompt(
                id=f"gsm8k/{n}",
                task="gsm8k",
                prompt=row["question"] + "\n\nThink step by step, then end with 'Answer: <number>'.",
                reference={"answer": answer},
            ))
    except Exception as exc:
        print(f"[prompts] gsm8k skipped via HF: {exc}")

    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        idxs = rng.sample(range(len(ds)), PER_TASK)
        for n, i in enumerate(idxs):
            row = ds[i]
            aliases = list({row["answer"]["value"], *row["answer"]["aliases"]})
            out.append(Prompt(
                id=f"triviaqa/{n}",
                task="triviaqa",
                prompt=row["question"] + "\n\nAnswer in one short phrase.",
                reference={"answers": aliases},
            ))
    except Exception as exc:
        print(f"[prompts] triviaqa skipped via HF: {exc}")

    try:
        ds = load_dataset("EdinburghNLP/xsum", split="test")
        idxs = rng.sample(range(len(ds)), PER_TASK)
        for n, i in enumerate(idxs):
            row = ds[i]
            article = row["document"][:4000]
            out.append(Prompt(
                id=f"xsum/{n}",
                task="xsum",
                prompt=f"Summarise the article in ONE sentence.\n\nARTICLE:\n{article}",
                reference={"summary": row["summary"]},
            ))
    except Exception as exc:
        print(f"[prompts] xsum skipped via HF: {exc}")

    try:
        ds = load_dataset("rajpurkar/squad", split="validation")
        idxs = rng.sample(range(len(ds)), PER_TASK)
        for n, i in enumerate(idxs):
            row = ds[i]
            out.append(Prompt(
                id=f"rag/{n}",
                task="rag",
                prompt=(
                    "Answer the question using ONLY the context below. "
                    "If the answer isn't in the context, say 'unknown'.\n\n"
                    f"CONTEXT:\n{row['context']}\n\nQUESTION: {row['question']}"
                ),
                reference={"answers": row["answers"]["text"]},
            ))
    except Exception as exc:
        print(f"[prompts] rag/squad skipped via HF: {exc}")

    counts: dict[str, int] = {}
    for p in out:
        counts[p.task] = counts.get(p.task, 0) + 1
    missing = [t for t in TASKS if counts.get(t, 0) < PER_TASK]
    if missing:
        have = ", ".join(f"{t}={counts.get(t, 0)}" for t in TASKS)
        print(f"[prompts] HF load incomplete (need {PER_TASK} per task, have {have}); "
              f"missing: {missing}. Falling back to bundled set.")
        return None
    return out


def _load_from_fallback() -> list[Prompt]:
    with open(FALLBACK, "r") as f:
        raw = json.load(f)
    return [Prompt(**p) for p in raw]


def load_prompts(force_refresh: bool = False) -> list[Prompt]:
    """Load 100 prompts, cache to disk after first call."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not force_refresh and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            raw = json.load(f)
        return [Prompt(**p) for p in raw]

    prompts = _try_load_from_hf()
    if not prompts:
        print("[prompts] using bundled fallback (HuggingFace `datasets` unavailable)")
        prompts = _load_from_fallback()

    with open(CACHE_PATH, "w") as f:
        json.dump([p.to_dict() for p in prompts], f, indent=2)

    return prompts


if __name__ == "__main__":
    ps = load_prompts()
    print(f"Loaded {len(ps)} prompts")
    by_task: dict[str, int] = {}
    for p in ps:
        by_task[p.task] = by_task.get(p.task, 0) + 1
    for t, n in by_task.items():
        print(f"  {t:12s} {n}")
