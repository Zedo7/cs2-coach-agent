"""CS2 round coach: a Doer proposes the next-round call, a Verifier audits it.

Run: python agent.py [match.json]
"""

import json
import os
import sys

import anthropic

MODEL = "claude-opus-4-8"

# Rough CS2 buy costs, used for the deterministic budget check.
PRICES = {
    "ak47": 2700, "m4a1s": 2900, "m4a4": 3100, "awp": 4750,
    "galil": 1800, "famas": 2050, "ssg08": 1700, "mac10": 1050, "mp9": 1250,
    "deagle": 700, "p250": 300, "glock": 0, "usp": 0,
    "kevlar": 650, "kevlar_helmet": 1000, "defuse_kit": 400,
    "he": 300, "flash": 200, "smoke": 300, "molotov": 400, "incendiary": 600,
}

DOER_SYSTEM = """You are the in-game leader of a CS2 team. You get the match state and the
last few rounds. Give the call for the NEXT round only.

You must:
- name the concrete reason the previous rounds were lost (cite the round detail, do not generalize),
- pick a buy that fits the team's average money,
- give one clear tactical instruction the team can execute.

Buy items must come from this list only:
ak47, m4a1s, m4a4, awp, galil, famas, ssg08, mac10, mp9, deagle, p250,
kevlar, kevlar_helmet, defuse_kit, he, flash, smoke, molotov, incendiary.
per_player_spend is the average money each player spends this round."""

VERIFIER_SYSTEM = """You are an independent CS2 match auditor. You did NOT write the plan and you
have no stake in it. You see only the raw match facts and a proposed plan.

Judge two things and nothing else:
1. ECONOMY: does the buy plan actually fit the team's average money? A full buy on
   ~2400 average is not feasible. Force-buys and ecos must be labeled honestly.
2. CAUSALITY: does loss_reason point at a specific, named failure from the recent
   rounds, and does the instruction actually address that failure? A generic reason
   ("we played badly", "poor execution") fails.

Be adversarial. Do not restate the plan or add tactics of your own. Approve only if
both checks pass."""

DOER_SCHEMA = {
    "type": "object",
    "properties": {
        "loss_reason": {"type": "string", "description": "Specific cause of the recent losses, citing round details."},
        "buy_type": {"type": "string", "enum": ["full_buy", "force_buy", "half_buy", "eco"]},
        "buy": {"type": "array", "items": {"type": "string"}, "description": "Item keys from the allowed list."},
        "per_player_spend": {"type": "integer", "description": "Average money spent per player."},
        "instruction": {"type": "string", "description": "The tactical call for next round."},
    },
    "required": ["loss_reason", "buy_type", "buy", "per_player_spend", "instruction"],
    "additionalProperties": False,
}

VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "econ_feasible": {"type": "boolean"},
        "reason_is_specific": {"type": "boolean"},
        "instruction_addresses_reason": {"type": "boolean"},
        "approved": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["econ_feasible", "reason_is_specific", "instruction_addresses_reason", "approved", "issues"],
    "additionalProperties": False,
}


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def parse(response):
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def budget_check(plan, avg_money):
    """Deterministic cost math, so the Verifier never has to trust the Doer's arithmetic."""
    unknown = [i for i in plan["buy"] if i not in PRICES]
    cost = sum(PRICES.get(i, 0) for i in plan["buy"])
    problems = []
    if unknown:
        problems.append(f"unknown items in buy: {unknown}")
    if cost > avg_money:
        problems.append(f"buy costs {cost} vs {avg_money} average money")
    if abs(cost - plan["per_player_spend"]) > 400:
        problems.append(f"claimed spend {plan['per_player_spend']} but items cost {cost}")
    return cost, problems


def doer(client, match, feedback=None):
    content = f"Match state:\n{json.dumps(match, ensure_ascii=False, indent=2)}"
    if feedback:
        content += (
            "\n\nYour previous plan was rejected by the auditor for these reasons:\n"
            + "\n".join(f"- {p}" for p in feedback)
            + "\n\nFix them and give the call again."
        )
    r = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=DOER_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": DOER_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    return parse(r)


def verifier(client, match, plan, computed_cost):
    # Decoupling: the Verifier gets facts + the plan only — never the Doer's system
    # prompt, thinking, or narration. Different persona, different effort, no thinking,
    # so it does not simply re-derive the Doer's conclusion.
    facts = {
        "map": match["map"],
        "side": match["side"],
        "score": match["score"],
        "our_average_money": match["economy"]["us_avg"],
        "their_average_money": match["economy"]["them_avg"],
        "recent_rounds": match["recent_rounds"],
    }
    r = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=VERIFIER_SYSTEM,
        thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERIFIER_SCHEMA}},
        messages=[{
            "role": "user",
            "content": (
                f"MATCH FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
                f"PROPOSED PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
                f"Independently computed cost of that buy: {computed_cost} per player."
            ),
        }],
    )
    return parse(r)


def main():
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example).")

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_match.json"
    with open(path, encoding="utf-8") as f:
        match = json.load(f)

    client = anthropic.Anthropic()
    feedback = None

    for attempt in range(1, 4):
        plan = doer(client, match, feedback)
        cost, hard_problems = budget_check(plan, match["economy"]["us_avg"])
        audit = verifier(client, match, plan, cost)

        issues = hard_problems + (audit["issues"] if not audit["approved"] else [])
        approved = audit["approved"] and not hard_problems

        print(f"\n--- attempt {attempt} ---")
        print(f"loss reason : {plan['loss_reason']}")
        print(f"buy         : {plan['buy_type']} — {', '.join(plan['buy'])} (~${cost}/player)")
        print(f"instruction : {plan['instruction']}")
        print(f"verifier    : {'APPROVED' if approved else 'REJECTED'}")
        for i in issues:
            print(f"  - {i}")

        if approved:
            print("\n=== NEXT ROUND CALL ===")
            print(f"{plan['buy_type'].replace('_', ' ').upper()}: {', '.join(plan['buy'])}")
            print(plan["instruction"])
            return

        feedback = issues

    print("\nNo plan passed the verifier after 3 attempts.")


if __name__ == "__main__":
    main()
