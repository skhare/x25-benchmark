"""
Phase 1 — Data collection.

Run every prompt against every model. Results are streamed to disk so an
interruption (rate-limit, cost cap, Ctrl-C) doesn't lose work — the script is
resumable and skips (prompt_id, model_id) pairs that already exist.

Output:  benchmark/data/raw_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from prompts import load_prompts
from models  import OpenRouter, MODELS, CallResult

OUT_PATH = os.path.join(HERE, "data", "raw_results.json")


def _load_existing() -> list[dict]:
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            return json.load(f)
    return []


def _key(prompt_id: str, model_id: str) -> str:
    return f"{prompt_id}::{model_id}"


def _atomic_write(path: str, payload: list[dict]) -> None:
    """Write JSON atomically so an interrupt during write can't corrupt the file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def collect(max_tokens: int = 1024, sleep_between: float = 0.0) -> str:
    prompts = load_prompts()
    client  = OpenRouter()

    results = _load_existing()
    by_key  = {_key(r["prompt_id"], r["model_id"]): i for i, r in enumerate(results)}
    seen_ok = {k for k, i in by_key.items() if not results[i].get("error")}

    total = len(prompts) * len(MODELS)
    done  = len(seen_ok)
    print(f"[phase1] {len(prompts)} prompts × {len(MODELS)} models = {total} calls "
          f"({done} successfully cached, {total - done} to attempt)", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one_call(p, m):
        try:
            resp = client.call(m["id"], p.prompt, max_tokens=max_tokens)
            return CallResult(
                model_id=m["id"], prompt_id=p.id, task=p.task,
                text=resp["text"], cost_usd=resp["cost_usd"],
                latency_ms=resp["latency_ms"],
                prompt_tokens=resp["prompt_tokens"],
                output_tokens=resp["output_tokens"],
            ).to_dict()
        except Exception as exc:
            return CallResult(
                model_id=m["id"], prompt_id=p.id, task=p.task,
                text="", cost_usd=0.0, latency_ms=0.0,
                prompt_tokens=0, output_tokens=0,
                error=str(exc)[:200],
            ).to_dict()

    for p in prompts:
        pending = [m for m in MODELS if _key(p.id, m["id"]) not in seen_ok]
        if not pending:
            continue
        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futs = {pool.submit(one_call, p, m): m for m in pending}
            for fut in as_completed(futs):
                row = fut.result()
                k = _key(row["prompt_id"], row["model_id"])
                m = futs[fut]
                if row.get("error"):
                    print(f"  [   /  ] {p.task:10s} {m['label']:20s} ERROR (will retry): {row['error']}", flush=True)
                else:
                    done += 1
                    print(f"  [{done}/{total}] {p.task:10s} {m['label']:20s} "
                          f"q={row['output_tokens']:4d}t  ${row['cost_usd']:.5f}  "
                          f"{row['latency_ms']:6.0f}ms", flush=True)
                if k in by_key:
                    results[by_key[k]] = row
                else:
                    by_key[k] = len(results)
                    results.append(row)
                if not row.get("error"):
                    seen_ok.add(k)
        _atomic_write(OUT_PATH, results)
        if sleep_between > 0:
            time.sleep(sleep_between)

    print(f"[phase1] done — wrote {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sleep",      type=float, default=0.0,
                    help="seconds to sleep between calls (rate limiting)")
    args = ap.parse_args()
    collect(max_tokens=args.max_tokens, sleep_between=args.sleep)
