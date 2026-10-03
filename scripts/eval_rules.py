"""Evaluate the plain-English rule compiler.

    python scripts/eval_rules.py                          # provider from .env (LLM_PROVIDER)
    python scripts/eval_rules.py --only w01,r03 --verbose
    python scripts/eval_rules.py --repeat 2               # run-to-run variance
    python scripts/eval_rules.py --write docs/BENCHMARKS.md

The cases are in tests/rules/compile_cases.json, with a fixed zone list. The real
RuleCompiler and LLM client run, the same path /api/rules/compile uses.

**Grading is deterministic, with no LLM judge.**
- A rule case is **exact** when the compiled conditions equal ``expect.conditions`` (as a
  set, with defaults filled in, ignoring the case's ``loose`` parameters) and ``duration_s``
  is equal; **semantic** when that holds for ``expect`` or for one of the ``accept``
  alternatives. Pass = semantic.
- A refusal case passes only if the compiler refused. Compiling a rule for it is a failure
  (the worst kind: it would have run).
- ``severity`` and ``notify`` are scored separately where a case states them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "backend"))

CASES = os.path.join(ROOT, "tests", "rules", "compile_cases.json")
SECTION_START = "<!-- benchmark:rules:start -->"
SECTION_END = "<!-- benchmark:rules:end -->"


# --- grading (pure; unit-tested) -----------------------------------------------------------------

def _norm_conditions(conditions: list[dict], loose: list[str]) -> list[str]:
    from rules.dsl import RuleBody

    body = RuleBody(name="x", conditions=conditions)  # fills in defaults, validates
    out = []
    for c in body.conditions:
        d = c.model_dump(mode="json")
        for k in loose:
            d.pop(k, None)
        out.append(json.dumps(d, sort_keys=True))
    return sorted(out)


def matches(expect: dict, rule: dict, loose: list[str]) -> bool:
    try:
        want = _norm_conditions(expect["conditions"], loose)
        got = _norm_conditions(rule["conditions"], loose)
    except ValueError:
        return False
    return want == got and float(expect.get("duration_s", 0)) == float(rule.get("duration_s", 0))


def grade(case: dict, result: dict) -> dict:
    """{"passed", "exact", "kind", "failures", "severity_ok", "notify_ok"} for one compile result."""
    expect = case["expect"]
    status = result.get("status")
    out = {"passed": False, "exact": False, "failures": [], "severity_ok": None, "notify_ok": None}
    if expect.get("refuse"):
        out["kind"] = "refusal"
        out["passed"] = out["exact"] = status == "refusal"
        if status == "rule":
            out["failures"].append("compiled a rule that should have been refused")
        elif status != "refusal":
            out["failures"].append(f"no refusal: {result.get('error') or status}")
        return out
    out["kind"] = "rule"
    if status != "rule":
        out["failures"].append(f"{status}: {result.get('refusal') or result.get('error')}")
        return out
    rule = result["rule"]
    loose = case.get("loose", [])
    out["exact"] = matches(expect, rule, loose)
    out["passed"] = out["exact"] or any(matches(a, rule, loose) for a in case.get("accept", []))
    if not out["passed"]:
        got = ", ".join(f"{c['type']}({', '.join(f'{k}={v}' for k, v in c.items() if k != 'type')})"
                        for c in rule["conditions"])
        out["failures"].append(f"got {got} for {rule.get('duration_s', 0):g} s")
    if "severity" in expect:
        out["severity_ok"] = rule.get("severity") == expect["severity"]
    if "notify" in expect:
        out["notify_ok"] = ("notify" in rule.get("actions", [])) == expect["notify"]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["passed"] for r in rows)
    lo, hi = wilson(k, n)
    refusals = [r for r in rows if r["kind"] == "refusal"]
    rules = [r for r in rows if r["kind"] == "rule"]
    sev = [r["severity_ok"] for r in rows if r["severity_ok"] is not None]
    notify = [r["notify_ok"] for r in rows if r["notify_ok"] is not None]
    by_domain: dict[str, list[int]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], [0, 0])
        by_domain[r["domain"]][0] += r["passed"]
        by_domain[r["domain"]][1] += 1
    lat = sorted(r["latency_s"] for r in rows)
    by_id: dict[str, list[bool]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(bool(r["passed"]))
    return {
        "cases": n, "passed": k, "accuracy": k / n if n else None, "ci95": [lo, hi],
        "exact": sum(r["exact"] for r in rows),
        "rules": len(rules), "rules_passed": sum(r["passed"] for r in rules),
        "refusals": len(refusals), "refusals_passed": sum(r["passed"] for r in refusals),
        "false_rules": sum(r["status"] == "rule" for r in refusals),
        "wrong_refusals": sum(r["status"] == "refusal" for r in rules),
        "errors": sum(r["status"] == "error" for r in rows),
        "repairs": sum(r["attempts"] > 1 for r in rows),
        "severity": [sum(sev), len(sev)], "notify": [sum(notify), len(notify)],
        "by_domain": {d: {"passed": v[0], "total": v[1]} for d, v in sorted(by_domain.items())},
        "user_cases": len({r["id"] for r in rows if r["source"] == "user"}),
        "latency_p50_s": lat[len(lat) // 2] if lat else None,
        "tokens_mean": sum(r["tokens"] for r in rows) / n if n else None,
        "flaky": sorted(i for i, v in by_id.items() if 0 < sum(v) < len(v)),
    }


def markdown(s: dict, provider: str, model: str, rows: list[dict], runs: int) -> str:
    acc = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "n/a"
    lines = [
        SECTION_START,
        f"_Measured {datetime.now():%Y-%m-%d} with `python scripts/eval_rules.py` using **{provider}** `{model}`"
        f"{f', {runs} runs of every case' if runs > 1 else ''}._ Cases are in `tests/rules/compile_cases.json` "
        "(fixed zones: Loading Dock, Forklift Lane, Chemical Storage, Yard Gate). Grading is deterministic: the "
        "compiled conditions and duration must match the expected rule (or an accepted equivalent), and "
        "must-refuse cases must be refused.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Pass rate** (semantic match or correct refusal) | **{s['passed']} / {s['cases']} ({acc})**, "
        f"95% CI {s['ci95'][0]:.0%}–{s['ci95'][1]:.0%} |",
        f"| Exact match (no alternative needed) | {s['exact']} / {s['cases']} |",
        f"| Rules compiled correctly | {s['rules_passed']} / {s['rules']} |",
        f"| Must-refuse cases refused | {s['refusals_passed']} / {s['refusals']} |",
        f"| **Rules compiled that should have been refused** | **{s['false_rules']}** |",
        f"| Valid requests wrongly refused | {s['wrong_refusals']} |",
        f"| Errors (invalid after the repair turn, or provider errors) | {s['errors']} |",
        f"| Needed the repair turn | {s['repairs']} |",
        f"| Severity as stated | {s['severity'][0]} / {s['severity'][1]} |",
        f"| Notify when asked (and only then) | {s['notify'][0]} / {s['notify'][1]} |",
        f"| Cases written by the user | {s['user_cases']} |",
        f"| Latency p50 / tokens per case (mean) | {s['latency_p50_s']:.1f} s / {s['tokens_mean']:,.0f} |",
        f"| Flaky cases (passed in some runs only) | {', '.join(s['flaky']) or 'none'} |",
        "",
        "| Domain | Passed |",
        "|---|---|",
    ]
    lines += [f"| {d} | {v['passed']} / {v['total']} |" for d, v in s["by_domain"].items()]
    failed = [r for r in rows if not r["passed"]]
    if failed:
        lines += ["", "Failures: " + "; ".join(f"`{r['id']}` ({r['failures'][0] if r['failures'] else r['status']})"
                                               for r in failed)]
    if s["user_cases"] == 0:
        lines += ["", "All cases so far were written by the developer (Claude); treat this as an upper bound until "
                      "the user's own sentences (`tests/rules/user_rules.md`) are in."]
    lines += ["", "The target is ≥ 90%.", SECTION_END]
    return "\n".join(lines)


def write_section(path: str, section: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        text = before + section + rest.split(SECTION_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Rule compiler accuracy\n\n" + section + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --- running -------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", help="nvidia | anthropic (default: LLM_PROVIDER from .env)")
    ap.add_argument("--model", help="override the provider's default model")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--write", metavar="MD", help="replace the rules section of this markdown file")
    ap.add_argument("--json", metavar="FILE", help="also save every result as JSON")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # previews contain arrows; the Windows console is cp1252

    from config.settings import SentinelConfig
    from llm import build_llm
    from rules.compiler import RuleCompiler

    cfg = SentinelConfig.from_env(os.path.join(ROOT, ".env"))
    if args.provider:
        cfg.llm.provider = args.provider
    if args.model:
        cfg.llm.model = args.model
    llm = build_llm(cfg.llm)
    with open(CASES, encoding="utf-8") as f:
        doc = json.load(f)
    zones = {k: tuple(v) for k, v in doc["zones"].items()}
    cases = doc["cases"]
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    compiler = RuleCompiler(llm)
    rows = []
    for run, case in [(k, c) for k in range(1, max(1, args.repeat) + 1) for c in cases]:
        res = compiler.compile(case["text"], zones, now="Monday 2026-10-05 12:00").to_dict()
        g = grade(case, res)
        row = {"id": case["id"], "run": run, "source": case.get("source", "claude"), "domain": case["domain"],
               "text": case["text"], "status": res["status"], "attempts": res["attempts"],
               "latency_s": res["latency_s"], "tokens": res["usage"]["input_tokens"] + res["usage"]["output_tokens"],
               "result": res, **g}
        rows.append(row)
        print(f"{'PASS' if g['passed'] else 'FAIL'} {case['id']} r{run} [{res['latency_s']:.1f}s] {case['text']}",
              flush=True)
        if args.verbose or not g["passed"]:
            print(f"     -> {res['status']}: {res.get('preview') or res.get('refusal') or res.get('error')}")
            for msg in g["failures"]:
                print(f"     ! {msg}")

    s = summarize(rows)
    print(json.dumps({k: v for k, v in s.items() if k != "by_domain"}, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": s, "rows": rows}, f, indent=1, default=str)
    if args.write:
        write_section(args.write, markdown(s, llm.provider, llm.model, rows, max(1, args.repeat)))
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
