#!/usr/bin/env python
"""Answer-key validator. Run after any change to prices.json or scenarios/.

Catches the failure mode where the eval scores a *correct* agent as wrong because
the answer key expects something the price table cannot afford.

  python validate_scenarios.py
"""
import json
import sys
from pathlib import Path

from agent import (
    FULL_BUY_KIT, PRICES, PRICES_AS_OF, PRICES_SOURCE, budget_check, build_doer_prompt,
)

ROOT = Path(__file__).parent
BUY_TYPES = {"full_buy", "force_buy", "half_buy", "eco"}

# FULL_BUY_KIT is imported from agent (single source). Only full_buy has a hard economic
# floor: it means a rifle plus armour plus utility, and below that price the label is a
# lie. eco / force_buy / half_buy are spend-level labels with no floor -- testing those
# against a fixed kit would encode the very assumption this validator exists to protect
# against, so only full_buy is floored here.
FLOORED_LABELS = {"full_buy": FULL_BUY_KIT}


def main() -> int:
    scen = sorted((json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "scenarios").glob("*.json")),
                  key=lambda s: s["id"])
    print(f"prices: {PRICES_SOURCE} (as_of {PRICES_AS_OF}) · {len(PRICES)} items")
    print(f"full_buy floor: ${sum(PRICES[i] for i in FULL_BUY_KIT)} "
          f"({'+'.join(FULL_BUY_KIT)}) · eco/force/half have no floor")
    for p in ("glock", "usp", "p2000"):
        if p in PRICES:
            print(f"  WARNING: default pistol '{p}' is priced; it should be unknown to the gate")
    problems = []

    for s in scen:
        sid, avg, exp = s["id"], s["economy"]["us_avg"], s["expected"]
        for key in ("map", "side", "score", "economy", "recent_rounds", "expected"):
            if key not in s:
                problems.append(f"{sid}: missing field {key}")
        if not exp["buy_type"] or not set(exp["buy_type"]) <= BUY_TYPES:
            problems.append(f"{sid}: bad buy_type {exp['buy_type']}")
        for group in exp.get("reason_tags", []):
            if not isinstance(group, list) or not group:
                problems.append(f"{sid}: reason_tags must be lists of aliases")

        # A floored label must be affordable at this scenario's economy, or the answer
        # key rewards something the deterministic gate would reject.
        for label, kit in FLOORED_LABELS.items():
            if label not in exp["buy_type"]:
                continue
            cost, probs = budget_check(
                {"buy": kit, "per_player_spend": sum(PRICES[i] for i in kit)}, avg)
            if probs:
                problems.append(f"{sid}: expects {label} but ${avg} cannot afford it (${cost})")

        # The answer key must never leak into the model's prompt.
        match = {k: v for k, v in s.items() if k not in ("id", "expected")}
        prompt = build_doer_prompt(match)
        for leak in ("expected", "reason_tags", "notes", "buy_type"):
            if leak in prompt:
                problems.append(f"{sid}: '{leak}' leaked into the doer prompt")

    counts = {g: sum(1 for s in scen if s["id"].startswith(g)) for g in ("normal", "eco", "caus", "edge")}
    print(f"scenarios: {len(scen)} " + " ".join(f"{k}={v}" for k, v in counts.items()))
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("all answer keys affordable, well-formed, and non-leaking")
    return 0


if __name__ == "__main__":
    sys.exit(main())
