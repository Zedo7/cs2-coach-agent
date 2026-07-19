"""Shared async run loop for cs2-coach-agent.

The single source of the agent trace: emits one TraceEvent per meaningful step
(tool call, plan, budget check, verifier verdict, retry, final). Both the CLI
(agent.main) and the HTTP service (service.py) consume this generator, so there is
exactly one place where the run's control flow lives.

The decision logic here is a faithful async port of the original sync loop -- same
prompts, same tool schemas, same deterministic gate, same 3-attempt retry policy.
Only the I/O changed: what used to be print() is now a yielded event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import anthropic

from agent import (
    DOER_SCHEMA, DOER_SYSTEM, DOER_TOOLS, MAX_TOOL_ITERATIONS, MODEL,
    VERIFIER_SCHEMA, VERIFIER_SYSTEM, budget_check, build_doer_prompt,
    build_verifier_prompt, run_tool, summarize,
)

CONFIGS = {
    "full":         {"hard_gate": True,  "verifier": True},
    "no_verifier":  {"hard_gate": True,  "verifier": False},
    "no_hard_gate": {"hard_gate": False, "verifier": True},
    "doer_only":    {"hard_gate": False, "verifier": False},
}

# Cache-aware pricing, mirroring eval.py. A cache write is 1.25x base input, a read 0.1x.
PRICING = {
    "claude-opus-4-8": (5 / 1e6, 25 / 1e6),
    "claude-haiku-4-5": (1 / 1e6, 5 / 1e6),
    "claude-sonnet-5": (3 / 1e6, 15 / 1e6),
}
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.10


@dataclass
class TraceEvent:
    event: str
    data: dict = field(default_factory=dict)

    def sse(self) -> str:
        """Server-Sent Events wire format: a named event plus a JSON data line."""
        return f"event: {self.event}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


def _price_of(model: str):
    for prefix, rates in PRICING.items():
        if model.startswith(prefix):
            return rates
    return PRICING["claude-opus-4-8"]  # unknown model: price as Opus rather than crash a run


def _model_params(model: str) -> dict:
    # Haiku 4.5 predates adaptive thinking and rejects `effort`; a single param set 400s it.
    if model.startswith("claude-haiku-4-5"):
        return {"thinking": {"type": "enabled", "budget_tokens": 2000}}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}


def _add_usage(acc: dict, model: str, resp) -> None:
    u = resp.usage
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    ci, co = _price_of(model)
    acc["input_tokens"] += u.input_tokens + write + read
    acc["output_tokens"] += u.output_tokens
    acc["cache_write_tokens"] += write
    acc["cache_read_tokens"] += read
    acc["cost_usd"] += (u.input_tokens * ci + write * ci * CACHE_WRITE_MULT
                        + read * ci * CACHE_READ_MULT + u.output_tokens * co)


def _parse_plan(resp) -> dict:
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        kinds = [b.type for b in resp.content]
        raise ValueError(f"doer returned no text block; got {kinds}, stop_reason={resp.stop_reason}")
    return json.loads(text)


async def run_agent(
    match: dict,
    config: str = "full",
    request_id: str = "-",
    client: Optional[anthropic.AsyncAnthropic] = None,
    model: str = MODEL,
    max_attempts: int = 3,
    extra_context: Optional[str] = None,
    verifier_model: Optional[str] = None,
) -> AsyncIterator[TraceEvent]:
    """Drive one full coaching run, yielding trace events. Terminal event is `final`.

    extra_context (session layer only) is appended to the doer prompt every attempt --
    compaction synopsis + adaptation pressure. None by default, so standalone runs are
    unchanged. The agent's own system prompts are never modified.

    verifier_model defaults to MODEL (Opus), NOT to `model`: the verifier is the
    independent auditor, so it stays a strong model even when the doer is weak -- pinning
    it is the whole point of the decoupling (mirrors eval.py). A weak doer + weak verifier
    would just rubber-stamp."""
    cfg = CONFIGS[config]
    verifier_model = verifier_model or MODEL
    own_client = client is None
    client = client or anthropic.AsyncAnthropic()
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_write_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.0}
    params = _model_params(model)
    avg_money = match["economy"]["us_avg"]

    try:
        yield TraceEvent("run_started", {
            "request_id": request_id, "config": config,
            "match_summary": {"map": match.get("map"), "side": match.get("side"),
                              "score": match.get("score"), "us_avg": avg_money},
        })

        feedback, plan, cost, approved, attempt = None, None, 0, False, 0
        for attempt in range(1, max_attempts + 1):
            yield TraceEvent("attempt_started", {"attempt": attempt})

            # --- doer tool loop -------------------------------------------------
            prompt = build_doer_prompt(match, feedback)
            if extra_context:
                prompt += "\n\n" + extra_context
            messages = [{"role": "user", "content": prompt}]
            for _ in range(MAX_TOOL_ITERATIONS):
                r = await client.messages.create(
                    model=model, max_tokens=4000, system=DOER_SYSTEM,
                    tools=DOER_TOOLS, cache_control={"type": "ephemeral"},
                    messages=messages, **params)
                _add_usage(usage, model, r)
                messages.append({"role": "assistant", "content": r.content})
                tool_uses = [b for b in r.content if b.type == "tool_use"]
                if not tool_uses:
                    break
                results = []
                for tu in tool_uses:
                    out = run_tool(match, tu.name, tu.input)
                    yield TraceEvent("tool_call", {
                        "attempt": attempt, "tool": tu.name,
                        "input_summary": summarize(tu.input),
                        "result_summary": summarize(out)})
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": json.dumps(out, ensure_ascii=False)})
                messages.append({"role": "user", "content": results})
            else:
                yield TraceEvent("doer_cap", {"attempt": attempt, "cap": MAX_TOOL_ITERATIONS})

            # --- final plan turn (may itself call tools on a weak model) ---------
            messages.append({"role": "user", "content": "Now output the final plan for the next round."})
            final_params = dict(params)
            oc = dict(final_params.get("output_config", {}))
            oc["format"] = {"type": "json_schema", "schema": DOER_SCHEMA}
            final_params["output_config"] = oc
            for _ in range(3):
                fr = await client.messages.create(
                    model=model, max_tokens=4000, system=DOER_SYSTEM,
                    tools=DOER_TOOLS, cache_control={"type": "ephemeral"},
                    messages=messages, **final_params)
                _add_usage(usage, model, fr)
                tus = [b for b in fr.content if b.type == "tool_use"]
                if not tus:
                    break
                messages.append({"role": "assistant", "content": fr.content})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tu.id,
                     "content": json.dumps(run_tool(match, tu.name, tu.input), ensure_ascii=False)}
                    for tu in tus]})
            plan = _parse_plan(fr)
            yield TraceEvent("doer_plan", {"attempt": attempt, "plan": plan})

            # --- deterministic gate --------------------------------------------
            cost, hard_problems = budget_check(plan, avg_money)
            yield TraceEvent("budget_check", {
                "attempt": attempt, "cost": cost,
                "feasible": not hard_problems, "problems": hard_problems})
            issues = list(hard_problems) if cfg["hard_gate"] else []

            # --- verifier (pinned to verifier_model, effort dropped if unsupported) -----
            if cfg["verifier"]:
                v_oc = {"format": {"type": "json_schema", "schema": VERIFIER_SCHEMA}}
                if not verifier_model.startswith("claude-haiku-4-5"):
                    v_oc["effort"] = "low"   # Haiku 4.5 rejects the effort parameter (400)
                vr = await client.messages.create(
                    model=verifier_model, max_tokens=2000, system=VERIFIER_SYSTEM,
                    thinking={"type": "disabled"}, output_config=v_oc,
                    messages=[{"role": "user", "content": build_verifier_prompt(match, plan, cost)}])
                _add_usage(usage, verifier_model, vr)
                verdict = _parse_plan(vr)
                yield TraceEvent("verifier_verdict", {"attempt": attempt, **verdict})
                if not verdict["approved"]:
                    issues += verdict["issues"]

            approved = not issues
            if approved:
                break
            yield TraceEvent("attempt_rejected", {"attempt": attempt, "issues": issues})
            feedback = issues

        yield TraceEvent("final", {
            "request_id": request_id, "config": config, "approved": approved,
            "plan": plan, "cost": cost, "attempts": attempt,
            "issues": [] if approved else issues,
            "usage": {**usage, "cost_usd": round(usage["cost_usd"], 4)},
        })
    finally:
        if own_client:
            await client.close()


def format_cli(ev: TraceEvent) -> str:
    """Render an event as a human trace line, preserving the old CLI look."""
    e, d = ev.event, ev.data
    if e == "run_started":
        m = d["match_summary"]
        return f"=== {m['map']} {m['side']} {m['score']} (us_avg {m['us_avg']}) · {d['config']} ==="
    if e == "attempt_started":
        return f"\n--- attempt {d['attempt']} ---"
    if e == "tool_call":
        return f"  [tool] {d['tool']}({d['input_summary']}) -> {d['result_summary']}"
    if e == "doer_cap":
        return f"  [tool] iteration cap ({d['cap']}) reached — forcing final plan"
    if e == "doer_plan":
        p = d["plan"]
        return (f"  plan: {p['buy_type']} — {', '.join(p['buy'])}\n"
                f"  reason: {p['loss_reason']}")
    if e == "budget_check":
        tag = "ok" if d["feasible"] else "VIOLATION " + "; ".join(d["problems"])
        return f"  budget: ${d['cost']}/player [{tag}]"
    if e == "verifier_verdict":
        return f"  verifier: {'APPROVED' if d['approved'] else 'REJECTED'}"
    if e == "attempt_rejected":
        return "  rejected:\n" + "\n".join(f"    - {i}" for i in d["issues"])
    if e == "final":
        if d["approved"]:
            p = d["plan"]
            return (f"\n=== NEXT ROUND CALL ({d['attempts']} attempt(s), "
                    f"${d['usage']['cost_usd']}) ===\n"
                    f"{p['buy_type'].replace('_', ' ').upper()}: {', '.join(p['buy'])}\n"
                    f"{p['instruction']}")
        return f"\nNo plan approved after {d['attempts']} attempts: {'; '.join(d['issues'])}"
    return f"  [{e}] {json.dumps(d, ensure_ascii=False)}"
