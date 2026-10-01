"""Evaluate plain-English search on the seeded event log.

    python scripts/eval_search.py                          # provider from .env (LLM_PROVIDER)
    python scripts/eval_search.py --provider anthropic
    python scripts/eval_search.py --only q07,q11 --verbose
    python scripts/eval_search.py --repeat 3              # every question 3 times: run-to-run variance
    python scripts/eval_search.py --write docs/BENCHMARKS.md

The questions are in tests/search/questions.json and the event log in tests/search/seed.py
(a fresh temporary database each run; "now" is fixed at Wed 2026-09-30 15:00). The real
SearchAgent, tools and LLM client run, the same path /api/search uses.

**Grading is deterministic, with no LLM judge.** One question passes only if every check in
its ``expect`` passes:

- ``count: N``: N appears in the answer as a number (digits, or a word up to twenty), not
  as part of a time or a longer number.
- ``breakdown: {label: N}``: for every label, one line or sentence names the label and N.
- ``events: [keys]``: the [#id] citations left after the citation check equal exactly
  this set.
- ``none: true``: no citations, and the answer says no / none / not / zero.
- ``labels_all: [...]``: every string appears (case-insensitive).
- ``labels_any: [...]``: at least one appears.

A question whose agent run fails (LLM error, step or token budget) counts as wrong.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))
sys.path.append(os.path.join(ROOT, "tests", "search"))

QUESTIONS = os.path.join(ROOT, "tests", "search", "questions.json")
SECTION_START = "<!-- benchmark:search:start -->"
SECTION_END = "<!-- benchmark:search:end -->"
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)


# --- grading (pure; unit-tested) --------------------------------------------------------------

def mentions_number(text: str, n: int) -> bool:
    if re.search(rf"(?<![\d.:#,]){n}(?![\d:]|,\d)", text):
        return True
    return 0 <= n < len(NUMBER_WORDS) and re.search(rf"\b{NUMBER_WORDS[n]}\b", text, re.IGNORECASE) is not None


def _chunks(text: str) -> list[str]:
    """Lines, and sentences within lines (for breakdown checks)."""
    out = []
    for line in text.splitlines():
        out.append(line)
        out += [s for s in re.split(r"(?<=[.;])\s+", line) if s]
    return out


_UNICODE_SPACES = dict.fromkeys(map(ord, "     "), " ")
_UNICODE_SPACES.update(dict.fromkeys(map(ord, "‐‑‒–"), "-"))


def normalize(text: str) -> str:
    """Models often use no-break spaces ("Loading Dock") and Unicode hyphens."""
    return (text or "").translate(_UNICODE_SPACES)


def grade(expect: dict[str, Any], answer: str, citations: list[int], event_ids: dict[str, int]) -> list[str]:
    """Failed checks (empty list = pass)."""
    failures = []
    answer = normalize(answer)
    low = answer.lower()
    if "count" in expect and not mentions_number(answer, int(expect["count"])):
        failures.append(f"count {expect['count']} not in answer")
    for label, n in expect.get("breakdown", {}).items():
        if not any(label.lower() in c.lower() and mentions_number(c, int(n)) for c in _chunks(answer)):
            failures.append(f"breakdown: no line with {label!r} and {n}")
    if "events" in expect:
        want = {event_ids[k] for k in expect["events"]}
        got = set(citations)
        if got != want:
            failures.append(f"events: cited {sorted(got)}, expected {sorted(want)}")
    if expect.get("none"):
        if citations:
            failures.append(f"none: cited {citations}")
        if not re.search(r"\b(no|none|not|zero|0|nothing)\b", low):
            failures.append("none: answer doesn't say there were none")
    for label in expect.get("labels_all", []):
        if label.lower() not in low:
            failures.append(f"label {label!r} missing")
    if expect.get("labels_any") and not any(s.lower() in low for s in expect["labels_any"]):
        failures.append(f"none of {expect['labels_any']} in answer")
    return failures


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    k = sum(r["passed"] for r in rows)
    lo, hi = wilson(k, n)
    lat = [r["latency_s"] for r in rows]
    runs: dict[int, list[dict[str, Any]]] = {}
    by_id: dict[str, list[bool]] = {}
    for r in rows:
        runs.setdefault(r.get("run", 1), []).append(r)
        by_id.setdefault(r["id"], []).append(bool(r["passed"]))
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        by_cat.setdefault(r["category"].split("+")[0], [0, 0])
        by_cat[r["category"].split("+")[0]][0] += r["passed"]
        by_cat[r["category"].split("+")[0]][1] += 1
    return {
        "questions": n, "passed": k, "accuracy": k / n if n else None, "ci95": [lo, hi],
        "latency_p50_s": percentile(lat, 0.5), "latency_p95_s": percentile(lat, 0.95),
        "tokens_mean": sum(r["tokens"] for r in rows) / n if n else None,
        "steps_mean": sum(r["steps"] for r in rows) / n if n else None,
        "failed_runs": sum(r["stop"] != "answered" for r in rows),
        "removed_citations": sum(len(r["removed_citations"]) for r in rows),
        "by_category": {c: {"passed": v[0], "total": v[1]} for c, v in sorted(by_cat.items())},
        "user_questions": len({r["id"] for r in rows if r["source"] == "user"}),
        "runs": len(runs),
        "per_run_accuracy": [sum(r["passed"] for r in rr) / len(rr) for rr in runs.values()],
        "flaky": sorted(i for i, v in by_id.items() if 0 < sum(v) < len(v)),
        "always_failed": sorted(i for i, v in by_id.items() if not any(v)),
    }


# --- report -----------------------------------------------------------------------------------

def markdown(s: dict[str, Any], provider: str, model: str, rows: list[dict[str, Any]]) -> str:
    acc = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "n/a"
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/eval_search.py` using **{provider}** "
        f"`{model}`._ The questions are in `tests/search/questions.json`, asked over the fixed event log in "
        "`tests/search/seed.py` (45 events; now = Wed 30 Sep 2026, 15:00). Grading is deterministic "
        "(numbers, cited event ids, labels), with no LLM judge.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Accuracy** ({s['runs']} run{'s' if s['runs'] > 1 else ''} of every question) | "
        f"**{s['passed']} / {s['questions']} ({acc})**, 95% CI {s['ci95'][0]:.0%}–{s['ci95'][1]:.0%} |",
        f"| Accuracy per run | {', '.join(f'{a:.1%}' for a in s['per_run_accuracy'])} |",
        f"| Questions that passed in some runs only (flaky) | {', '.join(s['flaky']) or 'none'} |",
        f"| Questions that failed in every run | {', '.join(s['always_failed']) or 'none'} |",
        f"| Questions written by the user | {s['user_questions']} |",
        f"| Latency per question p50 / p95 | {s['latency_p50_s']:.1f} s / {s['latency_p95_s']:.1f} s |",
        f"| Model turns per question (mean) | {s['steps_mean']:.1f} |",
        f"| Tokens per question (mean, input + output) | {s['tokens_mean']:,.0f} |",
        f"| Runs that ended without an answer (error or budget) | {s['failed_runs']} |",
        f"| Made-up citations removed | {s['removed_citations']} |",
        "",
        "| Category | Passed |",
        "|---|---|",
    ]
    lines += [f"| {c} | {v['passed']} / {v['total']} |" for c, v in s["by_category"].items()]
    failed = [r for r in rows if not r["passed"]]
    if failed:
        lines += ["", "Failures: " + "; ".join(
            f"`{r['id']}`{' run ' + str(r.get('run', 1)) if s['runs'] > 1 else ''} ({r['failures'][0]})"
            for r in failed)]
    lines += ["", "The target is ≥ 90%. With 30 or fewer questions, the confidence interval is wide.",
              SECTION_END]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Search accuracy\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --- running ----------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", help="nvidia | anthropic (default: LLM_PROVIDER from .env)")
    ap.add_argument("--model", help="override the provider's default model")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--repeat", type=int, default=1, help="ask every question this many times (variance)")
    ap.add_argument("--verbose", action="store_true", help="print every answer and tool call")
    ap.add_argument("--write", metavar="MD", help="replace the search section of this markdown file")
    args = ap.parse_args()

    from seed import EVENT_IDS, ZONES, build_store, now_ts

    from config.settings import SentinelConfig
    from llm import build_llm
    from search import SearchAgent, SearchTools

    cfg = SentinelConfig.from_env(os.path.join(ROOT, ".env"))
    if args.provider:
        cfg.llm.provider = args.provider
    if args.model:
        cfg.llm.model = args.model
    llm = build_llm(cfg.llm)

    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        store = build_store(os.path.join(tmp, "events.db"))
        for run, q in [(k, q) for k in range(1, max(1, args.repeat) + 1) for q in questions]:
            agent = SearchAgent(llm, SearchTools(store, ZONES, now=now_ts), max_steps=cfg.search.max_steps,
                                max_tokens=cfg.search.max_tokens_per_question)
            t0 = time.time()
            r = agent.ask(q["question"])
            failures = grade(q["expect"], r.answer, r.citations, EVENT_IDS)
            if r.stop != "answered":
                failures.insert(0, f"run ended: {r.stop} {r.error or ''}".strip())
            row = {"id": q["id"], "run": run, "source": q.get("source", "claude"), "category": q["category"],
                   "question": q["question"], "answer": r.answer, "citations": r.citations,
                   "removed_citations": r.removed_citations, "tool_calls": r.tool_calls, "stop": r.stop,
                   "error": r.error, "steps": r.steps, "tokens": r.usage.total,
                   "latency_s": round(time.time() - t0, 2), "passed": not failures, "failures": failures}
            rows.append(row)
            mark = "PASS" if row["passed"] else "FAIL"
            print(f"{mark} {q['id']} r{run} [{row['latency_s']:.1f}s, {r.steps} turns] {q['question']}", flush=True)
            if args.verbose or not row["passed"]:
                for c in r.tool_calls:
                    err = f" -> {c['error']}" if c["error"] else ""
                    print(f"     tool {c['name']} {json.dumps(c['arguments'])}{err}")
                print("     answer: " + (r.answer or "").replace("\n", "\n             "))
                for f_ in failures:
                    print(f"     x {f_}")

    s = summarize(rows)
    out = os.path.join(ROOT, "outputs", f"eval_search_{llm.provider}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"provider": llm.provider, "model": llm.model, "summary": s, "questions": rows}, f, indent=2)
    section = markdown(s, llm.provider, llm.model, rows)
    print("\n" + section + f"\n\nper-question results: {out}")
    if args.write:
        if args.only:
            print("not writing: --only runs a subset")
        else:
            write_section(args.write, section)
            print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
