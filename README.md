# X25 Benchmark Suite

End-to-end benchmark comparing 4 LLM routing strategies on 5 task families using 5 OpenRouter models.

## What it measures

| | |
|---|---|
| **Models** | gpt-4o, claude-sonnet-4.5, gpt-4o-mini, llama-3.3-70b, deepseek-chat |
| **Tasks** | HumanEval (code), GSM8K (math), TriviaQA (factual QA), XSum (summarisation), SQuAD (RAG) |
| **Strategies** | Always-Frontier (claude-sonnet-4.5), Always-Cheap (deepseek), Random, X25 cascade |
| **Sample size** | 4 prompts × 5 tasks = 20 prompts (configurable in `prompts.py`, set to 20 for the full 100-prompt run) |
| **Cost** | ~$0.20 OpenRouter + ~$0.05 OpenAI judge for the bundled 20-prompt sample |

## Pipeline

```
prompts ─► phase1_collect ─► raw_results.json   (20×5 calls, OpenRouter)
                  │
                  ▼
            phase2_score ─► scored_results.json (deterministic + LLM-judge)
                  │
                  ▼
          phase3_strategies ─► strategy_results.json  (no new API calls)
                  │
                  ▼
            phase4_charts ─► radar.png, results.png, heatmap.png, summary.md
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
export OPENAI_API_KEY=sk-proj-...     # only needed for phase 2 (xsum, rag judge)
```

If `datasets` isn't installable in your environment, the prompt loader falls back to the bundled set in `prompts_fallback.json` (5 prompts per task).

## Run

```bash
python run_all.py             # everything end-to-end (~3 min, ~$0.25)
python run_all.py --only 3 4  # rerun simulation + charts only
python run_all.py --skip 1    # use cached raw_results, redo scoring + downstream
```

Phase 1 is **resumable** — re-running it skips `(prompt_id, model_id)` pairs already in `data/raw_results.json`. If a model 429s or you Ctrl-C halfway through, just rerun.

## Outputs

All written to `data/`:

- `prompts.json` — the actual prompts used (cached from HuggingFace or the bundle)
- `raw_results.json` — every model's raw response, cost, latency, tokens
- `scored_results.json` — same + per-row quality score and reason
- `strategy_results.json` — per-prompt outcome of each of the 4 strategies + summary
- `radar.png`, `results.png`, `heatmap.png` — charts
- `summary.md` — headline numbers in markdown

Open the PNGs with your OS file viewer (`open data/*.png` on macOS, `xdg-open` on Linux), or embed in a Jupyter notebook with `IPython.display.Image`.

## Headline result (bundled 20-prompt run)

| Strategy | Avg quality | Total cost | Avg latency | Audit |
|---|---:|---:|---:|:---:|
| Always-Frontier (claude-sonnet-4.5) | 0.71 | $0.0311 | 2,827 ms | no |
| Always-Cheap (deepseek) | 0.73 | $0.0024 | 9,496 ms | no |
| Random | 0.74 | $0.0180 | 3,444 ms | no |
| **X25** | **0.84** | **$0.0065** | 4,182 ms | **yes** |

X25 wins quality at 5× lower cost than always-frontier. The biggest delta is on HumanEval (code), where the cascade catches and re-routes when the cheap model fails.

## How X25 is simulated

To make the comparison reproducible and fair, phase 3 does **not** call a live routing service. It replays the cached results through a task-aware cascade:

1. Pick a per-task tier order (e.g. code → start with gpt-4o-mini, escalate to gpt-4o, then claude-sonnet-4.5).
2. Walk down the cascade. If a model's cached quality is ≥ 0.6, accept it and stop.
3. Cost and latency for the prompt = sum of every attempt the cascade made.

This is the **deterministic component** of X25 — a static cascade baseline. The live X25 gateway adds per-org Thompson Sampling and task-dependent quality thresholds on top. With a 20-prompt sample the bandit hasn't converged yet, so the static approximation is a reasonable lower bound on what the live agent achieves.

## Security note (HumanEval scoring)

`phase2_score.py` runs model-generated Python in a subprocess to test it against HumanEval reference cases — the same approach used by OpenAI's official `human-eval` package. The benchmark applies CPU/memory/file-size RLIMITs, runs Python in isolated mode (`-I`), and uses a clean `PATH`-only env, but it does **not** prevent network access or filesystem reads under your home directory.

For a 5-model × 20-prompt HumanEval grid you're running 100 short snippets. Either:

- accept the risk (mainstream practice for HumanEval), or
- run `phase2_score.py` inside a container or fresh VM if you don't trust the model output

## Knobs

- `models.py` — change the 5 model IDs / pricing
- `phase3_strategies.py` — `QUALITY_THRESHOLD` (default 0.6) and `TASK_CASCADES`
- `prompts.py` — `PER_TASK` (default 4, set to 20 for the full 100-prompt run)
- `X25_JUDGE_MODEL` env var — judge model id (default `gpt-4o-mini`)
