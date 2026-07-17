"""CS2 round coach: a Doer proposes the next-round call, a Verifier audits it.

Run: python agent.py [match.json]
"""

import json
import os
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-8"

# The deterministic gate is only as trustworthy as this table, so it is pinned to a
# file with a source URL and a date rather than hardcoded from memory.
_PRICES_FILE = json.loads((Path(__file__).parent / "prices.json").read_text(encoding="utf-8"))
PRICES = _PRICES_FILE["prices"]
PRICES_SOURCE, PRICES_AS_OF = _PRICES_FILE["source"], _PRICES_FILE["as_of"]

# NOTE: the ~$4200 full_buy floor below must stay in sync with FULL_BUY_KIT in
# validate_scenarios.py (ak47 + kevlar_helmet + flash + smoke). If prices.json changes
# that kit's cost, update both — the prompt teaches the model the floor, the validator
# enforces the same floor on the answer keys.
DOER_SYSTEM = """You are the in-game leader of a CS2 team. You get the match state and the
last few rounds. Give the call for the NEXT round only.

You have tools. Use them before committing to a plan:
- get_round_details to read exactly how specific rounds were lost,
- get_item_prices to look up what things cost,
- check_budget to confirm a candidate buy actually fits the team's money.

Do not guess at prices or affordability — check. Iterate on the buy until check_budget
says it is feasible.

You must:
- name the concrete reason the previous rounds were lost (cite the round detail, do not generalize),
- pick a buy that fits the team's average money,
- give one clear tactical instruction the team can execute.

Buy items must come from this list only:
ak47, m4a1s, m4a4, awp, galil, famas, ssg08, mac10, mp9, deagle, p250,
kevlar, kevlar_helmet, defuse_kit, he, flash, smoke, molotov, incendiary.
per_player_spend is the average money each player spends this round.

Default pistols (glock/usp/p2000) are not buy items.

buy_type is ABSOLUTE — judged by what the buy actually contains and costs, never
relative to what you happen to be able to afford this round:
- full_buy: rifle + armour + utility, roughly $4200+ per player. If you cannot afford
  that, it is NOT a full buy no matter how much of your money you spend.
- force_buy: spending most of what you have on an under-equipped kit, going for the
  round anyway.
- half_buy: a partial kit, deliberately keeping money back.
- eco: minimal spend to save for a future round. The buy list may be empty or contain
  cheap purchases like a p250 or a deagle.
Spending everything you own is not a full buy if everything you own is $800.
Label by intent and absolute contents, not by whether the list is empty."""

DOER_TOOLS = [
    {
        "name": "check_budget",
        "description": (
            "Check whether a candidate buy fits a per-player budget. Returns the total cost, "
            "whether it is feasible, and any problems (unknown items, overspend)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}, "description": "Item keys for one player's buy."},
                "avg_money": {"type": "integer", "description": "Average money available per player."},
            },
            "required": ["items", "avg_money"],
        },
    },
    {
        "name": "get_item_prices",
        "description": "Look up the price of specific items. Returns a price per item; unknown items come back as null.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}, "description": "Item keys to price."},
            },
            "required": ["items"],
        },
    },
    {
        "name": "get_round_details",
        "description": "Get the full recorded detail for specific recent rounds, so you can cite how they were actually lost.",
        "input_schema": {
            "type": "object",
            "properties": {
                "round_numbers": {"type": "array", "items": {"type": "integer"}, "description": "Round numbers to fetch."},
            },
            "required": ["round_numbers"],
        },
    },
]

VERIFIER_SYSTEM = """You are an independent CS2 match auditor. You did NOT write the plan and you
have no stake in it. You see only the raw match facts and a proposed plan.

Judge two things and nothing else:
1. ECONOMY: does the buy plan actually fit the team's average money? A full buy on
   ~2400 average is not feasible. Force-buys and ecos must be labeled honestly —
   honesty here is about spend level matching the label, NOT about the buy list being
   empty. An eco with a p250 or a deagle on it is normal, correct play; do not fault a
   non-empty eco. Fault a plan only when the money it spends contradicts the label it
   claims.
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


def tool_check_budget(match, items, avg_money):
    cost, problems = budget_check({"buy": items, "per_player_spend": sum(PRICES.get(i, 0) for i in items)}, avg_money)
    return {"cost": cost, "feasible": not problems, "problems": problems}


def tool_get_item_prices(match, items):
    return {i: PRICES.get(i) for i in items}


def tool_get_round_details(match, round_numbers):
    wanted = set(round_numbers)
    found = [r for r in match["recent_rounds"] if r["round"] in wanted]
    missing = sorted(wanted - {r["round"] for r in found})
    out = {"rounds": found}
    if missing:
        out["unavailable"] = missing
    return out


TOOL_IMPLS = {
    "check_budget": tool_check_budget,
    "get_item_prices": tool_get_item_prices,
    "get_round_details": tool_get_round_details,
}

MAX_TOOL_ITERATIONS = 8


def run_tool(match, name, tool_input):
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return impl(match, **tool_input)
    except Exception as e:  # surface the failure to the model rather than crashing the loop
        return {"error": f"{type(e).__name__}: {e}"}


def summarize(result):
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= 160 else s[:157] + "..."


def doer(client, match, feedback=None):
    """Agentic loop: the model calls tools until it is ready, then emits the plan."""
    messages = [{"role": "user", "content": build_doer_prompt(match, feedback)}]

    for i in range(MAX_TOOL_ITERATIONS):
        r = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=DOER_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            tools=DOER_TOOLS,
            # Auto-caches the last cacheable block, i.e. the end of the conversation
            # so far. Each turn reads the prefix the previous turn wrote — this is
            # where the cost lives, since the loop resends a growing history 5x.
            # tools+system alone is only ~1.1k tokens, under the 4096 minimum, so a
            # separate static breakpoint would silently never cache.
            cache_control={"type": "ephemeral"},
            messages=messages,
        )
        # Echo the whole assistant turn back, thinking blocks included.
        messages.append({"role": "assistant", "content": r.content})

        tool_uses = [b for b in r.content if b.type == "tool_use"]
        if not tool_uses:
            break

        results = []
        for tu in tool_uses:
            result = run_tool(match, tu.name, tu.input)
            print(f"  [tool] {tu.name}({json.dumps(tu.input, ensure_ascii=False)}) -> {summarize(result)}")
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        messages.append({"role": "user", "content": results})
    else:
        print(f"  [tool] iteration cap ({MAX_TOOL_ITERATIONS}) reached — forcing final plan")

    # Final turn: same conversation, now constrained to the plan schema.
    messages.append({"role": "user", "content": "Now output the final plan for the next round."})
    final = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=DOER_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": DOER_SCHEMA}},
        tools=DOER_TOOLS,
        cache_control={"type": "ephemeral"},
        messages=messages,
    )
    return parse(final)


def build_verifier_facts(match):
    # Decoupling: the Verifier gets facts + the plan only — never the Doer's system
    # prompt, thinking, or narration. Different persona, different effort, no thinking,
    # so it does not simply re-derive the Doer's conclusion.
    return {
        "map": match["map"],
        "side": match["side"],
        "score": match["score"],
        "our_average_money": match["economy"]["us_avg"],
        "their_average_money": match["economy"]["them_avg"],
        "recent_rounds": match["recent_rounds"],
    }


def build_verifier_prompt(match, plan, computed_cost):
    return (
        f"MATCH FACTS:\n{json.dumps(build_verifier_facts(match), ensure_ascii=False, indent=2)}\n\n"
        f"PROPOSED PLAN:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
        f"Independently computed cost of that buy: {computed_cost} per player."
    )


def build_doer_prompt(match, feedback=None):
    content = f"Match state:\n{json.dumps(match, ensure_ascii=False, indent=2)}"
    if feedback:
        content += (
            "\n\nYour previous plan was rejected by the auditor for these reasons:\n"
            + "\n".join(f"- {p}" for p in feedback)
            + "\n\nFix them and give the call again."
        )
    return content


def verifier(client, match, plan, computed_cost):
    r = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=VERIFIER_SYSTEM,
        thinking={"type": "disabled"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": VERIFIER_SCHEMA}},
        messages=[{"role": "user", "content": build_verifier_prompt(match, plan, computed_cost)}],
    )
    return parse(r)


def main():
    # Round details and instructions are Chinese; a cp1252 console would die on them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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