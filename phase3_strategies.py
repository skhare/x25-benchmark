"""
Phase 3 — Strategy simulation.

Replays the same 100 prompts through 4 strategies, using cached results from
phase 2 (NO new API calls).

  always_frontier  : always use anthropic/claude-sonnet-4.5
  always_cheap     : always use deepseek/deepseek-chat
  random           : pick one of the 5 models uniformly at random per prompt
  x25              : task-aware static cascade — start cheap, accept if quality
                     ≥ 0.6, escalate up the cost-spread ladder otherwise.
                     Per-prompt cost = sum of every attempt that was made.

CAVEAT: the live X25 gateway in `gateway/agent.py` is more sophisticated than
this — it uses per-org Thompson Sampling, dynamic tier resolution from the
OpenRouter registry, and task-dependent quality thresholds (gateway/evaluator.py
uses 0.60–0.75 depending on task). This phase is therefore a *static cascade
baseline* approximating X25's deterministic component, not a full reproduction
of its bandit behaviour. With 100 prompts the bandit hasn't converged anyway,
so the static approximation is a reasonable lower bound on what the live
agent would achieve.

Output: benchmark/data/strategy_results.json
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from models import MODELS, CASCADE_ORDER

SCORED_PATH = os.path.join(HERE, "data", "scored_results.json")
OUT_PATH    = os.path.join(HERE, "data", "strategy_results.json")

QUALITY_THRESHOLD = 0.6


TASK_CASCADES: dict[str, list[str]] = {
    "humaneval": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4.5",
    ],
    "gsm8k": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4.5",
    ],
    "triviaqa": [
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
    ],
    "xsum": [
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o",
    ],
    "rag": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4.5",
    ],
}

DEFAULT_CASCADE = CASCADE_ORDER


def _index(scored: list[dict]) -> dict[str, dict[str, dict]]:
    """idx[prompt_id][model_id] -> result row."""
    idx: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in scored:
        idx[r["prompt_id"]][r["model_id"]] = r
    return idx


def _replay_single(idx: dict[str, dict[str, dict]], prompt_id: str,
                   model_id: str) -> dict:
    row = idx.get(prompt_id, {}).get(model_id)
    if row is None:
        return {"model_used": model_id, "quality": 0.0, "cost_usd": 0.0,
                "latency_ms": 0.0, "cascade": [model_id], "missing": True}
    return {
        "model_used": model_id,
        "quality":    row.get("quality", 0.0),
        "cost_usd":   row.get("cost_usd", 0.0),
        "latency_ms": row.get("latency_ms", 0.0),
        "cascade":    [model_id],
    }


def _replay_x25(idx: dict[str, dict[str, dict]], prompt_id: str,
                task: str) -> dict:
    cascade = TASK_CASCADES.get(task, DEFAULT_CASCADE)
    total_cost = 0.0
    total_latency = 0.0
    used: list[str] = []
    final_quality = 0.0
    final_model   = cascade[-1]
    for mid in cascade:
        row = idx.get(prompt_id, {}).get(mid)
        if row is None:
            continue
        used.append(mid)
        total_cost    += row.get("cost_usd", 0.0)
        total_latency += row.get("latency_ms", 0.0)
        q = row.get("quality", 0.0)
        if q >= QUALITY_THRESHOLD:
            final_quality, final_model = q, mid
            break
        final_quality, final_model = q, mid
    return {
        "model_used": final_model,
        "quality":    final_quality,
        "cost_usd":   total_cost,
        "latency_ms": total_latency,
        "cascade":    used,
    }


def simulate() -> str:
    if not os.path.exists(SCORED_PATH):
        raise FileNotFoundError(
            f"{SCORED_PATH} not found — run phase2_score.py first."
        )
    with open(SCORED_PATH, "r") as f:
        scored = json.load(f)

    idx = _index(scored)
    prompts = sorted({(r["prompt_id"], r["task"]) for r in scored})
    rng = random.Random(42)

    strategies: dict[str, list[dict]] = {
        "always_frontier": [],
        "always_cheap":    [],
        "random":          [],
        "x25":             [],
    }

    for pid, task in prompts:
        strategies["always_frontier"].append(
            {"prompt_id": pid, "task": task,
             **_replay_single(idx, pid, "anthropic/claude-sonnet-4.5")}
        )
        strategies["always_cheap"].append(
            {"prompt_id": pid, "task": task,
             **_replay_single(idx, pid, "deepseek/deepseek-chat")}
        )
        chosen = rng.choice([m["id"] for m in MODELS])
        strategies["random"].append(
            {"prompt_id": pid, "task": task,
             **_replay_single(idx, pid, chosen)}
        )
        strategies["x25"].append(
            {"prompt_id": pid, "task": task,
             **_replay_x25(idx, pid, task)}
        )

    summary: dict[str, dict] = {}
    for name, rows in strategies.items():
        n         = max(len(rows), 1)
        total_q   = sum(r["quality"]    for r in rows)
        total_c   = sum(r["cost_usd"]   for r in rows)
        total_l   = sum(r["latency_ms"] for r in rows)
        per_task: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_task[r["task"]].append(r["quality"])
        summary[name] = {
            "n_calls":       len(rows),
            "avg_quality":   total_q / n,
            "total_cost":    total_c,
            "avg_latency":   total_l / n,
            "has_audit_log": name == "x25",
            "per_task_quality": {k: sum(v) / max(len(v), 1) for k, v in per_task.items()},
            "per_task_count":   {k: len(v) for k, v in per_task.items()},
        }

    out = {"strategies": strategies, "summary": summary,
           "quality_threshold": QUALITY_THRESHOLD}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[phase3] done — wrote {OUT_PATH}")
    print()
    print(f"{'strategy':18s}  {'avg_q':>6s}  {'total$':>8s}  {'avg_lat_ms':>10s}  audit")
    for name, s in summary.items():
        print(f"  {name:16s}  {s['avg_quality']:6.3f}  ${s['total_cost']:7.4f}  "
              f"{s['avg_latency']:10.0f}  {'yes' if s['has_audit_log'] else 'no'}")
    return OUT_PATH


if __name__ == "__main__":
    simulate()
