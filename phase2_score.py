"""
Phase 2 — Quality scoring.

Per-task scorers:
    humaneval : run the official test harness in a subprocess, pass = 1.0
    gsm8k     : extract last numeric answer, exact match = 1.0
    triviaqa  : substring match against any acceptable answer (alias-aware)
    xsum      : LLM-as-judge (OpenAI, default gpt-4o-mini)
    rag       : LLM-as-judge

Errors → quality 0.0.
Output: benchmark/data/scored_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from typing import Optional

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from prompts import load_prompts

RAW_PATH    = os.path.join(HERE, "data", "raw_results.json")
SCORED_PATH = os.path.join(HERE, "data", "scored_results.json")

JUDGE_MODEL = os.environ.get("X25_JUDGE_MODEL", "gpt-4o-mini")


def _strip_md_fences(text: str) -> str:
    """Strip ``` code fences if present, but preserve internal indentation."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        return stripped.rstrip("\n") + "\n"
    return text.rstrip("\n") + "\n"


def _humaneval_preexec() -> None:
    """Apply RLIMITs to the candidate-code subprocess.

    SECURITY NOTE: the official `human-eval` harness runs untrusted model output
    too. These limits constrain runaway resource use but do NOT sandbox the
    filesystem or network — only run this on a host where you accept that risk
    (or run `phase2_score.py` inside a container / VM). See benchmark/README.md.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except (ValueError, OSError):
        pass


def score_humaneval(response: str, ref: dict, timeout: float = 8.0) -> tuple[float, str]:
    code = _strip_md_fences(response)
    if "def " + ref["entry_point"] in code:
        body = code
    else:
        body = ref["prompt"] + code
    program = (
        body
        + "\n"
        + ref["test"]
        + "\n"
        + f"check({ref['entry_point']})\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name

    try:
        res = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_humaneval_preexec,
        )
    except subprocess.TimeoutExpired:
        return 0.0, "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if res.returncode == 0:
        return 1.0, "all tests passed"
    return 0.0, (res.stderr or res.stdout or "failed")[-200:]


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def score_gsm8k(response: str, ref: dict) -> tuple[float, str]:
    expected = ref["answer"].replace(",", "").strip()
    m = re.search(r"answer\s*[:=]?\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if not m:
        nums = _NUMBER_RE.findall(response.replace(",", ""))
        if not nums:
            return 0.0, "no number found"
        got = nums[-1]
    else:
        got = m.group(1)
    try:
        if abs(float(got) - float(expected)) < 1e-3:
            return 1.0, f"got={got}"
    except ValueError:
        pass
    return 0.0, f"got={got} expected={expected}"


def score_triviaqa(response: str, ref: dict) -> tuple[float, str]:
    answers = [a.strip() for a in ref["answers"] if a and a.strip()]
    text    = response.strip().lower()
    for a in answers:
        a_low = a.lower()
        if not a_low:
            continue
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(a_low)}(?![A-Za-z0-9_])", text):
            return 1.0, f"matched '{a}'"
    return 0.0, f"no match (expected one of {answers[:3]})"


_JUDGE_PROMPT = """You are an evaluator. Score the candidate answer from 0.0 to 1.0
for correctness and completeness against the reference. Return ONLY JSON of the form
{{"score": <float 0-1>, "reason": "<short reason>"}}. Be strict but fair.

QUESTION/TASK:
{task_text}

REFERENCE:
{reference}

CANDIDATE:
{candidate}
"""


def _judge(task_text: str, reference: str, candidate: str,
           judge_client) -> tuple[float, str]:
    if not candidate.strip():
        return 0.0, "empty response"
    msg = _JUDGE_PROMPT.format(task_text=task_text[:2000],
                               reference=reference[:1000],
                               candidate=candidate[:2000])
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": msg}],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score)), str(data.get("reason", ""))[:200]
    except Exception as exc:
        return 0.0, f"judge error: {str(exc)[:120]}"


def _get_judge_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. The xsum and rag scorers use OpenAI as a "
            "judge. Export the key or change X25_JUDGE_MODEL."
        )
    return OpenAI()


def score_all() -> str:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found — run phase1_collect.py first."
        )
    with open(RAW_PATH, "r") as f:
        raw = json.load(f)

    prompts = {p.id: p for p in load_prompts()}

    judge_client = None
    needs_judge  = any(r.get("task") in ("xsum", "rag") for r in raw)
    if needs_judge:
        judge_client = _get_judge_client()

    scored: list[dict] = []
    for n, r in enumerate(raw, 1):
        p = prompts.get(r["prompt_id"])
        if p is None:
            continue
        if r.get("error"):
            quality, reason = 0.0, f"call error: {r['error']}"
        elif p.task == "humaneval":
            quality, reason = score_humaneval(r["text"], p.reference)
        elif p.task == "gsm8k":
            quality, reason = score_gsm8k(r["text"], p.reference)
        elif p.task == "triviaqa":
            quality, reason = score_triviaqa(r["text"], p.reference)
        elif p.task == "xsum":
            quality, reason = _judge(
                "Summarise the article in ONE sentence.",
                p.reference.get("summary", ""),
                r["text"], judge_client,
            )
        elif p.task == "rag":
            quality, reason = _judge(
                p.prompt,
                ", ".join(p.reference.get("answers", [])),
                r["text"], judge_client,
            )
        else:
            quality, reason = 0.0, f"unknown task {p.task}"

        scored.append({**r, "quality": quality, "score_reason": reason})
        print(f"  [{n}/{len(raw)}] {p.task:10s} {r['model_id'][:30]:30s} q={quality:.2f}  {reason[:60]}")

        with open(SCORED_PATH, "w") as f:
            json.dump(scored, f, indent=2)

    print(f"[phase2] done — wrote {SCORED_PATH}")
    return SCORED_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    score_all()
