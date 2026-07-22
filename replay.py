#!/usr/bin/env python
"""Replay a parsed match through a coaching session -- the closed-loop demo artifact.

For each round, in order: ask the agent for its call for THIS round (grounded only in the
rounds before it), then reveal the real outcome and submit it to the ledger. The report
shows, per round, what the agent would have called vs what the team actually ran and how
the round actually went.

Uses the same SessionStore + build_grounding + run_agent path the /v1/sessions/{id}/coach
endpoint uses, so the replay exercises real session behaviour (K-window, compaction,
adaptation pressure), just driven locally instead of over HTTP.

  python replay.py matches/<id>/rounds.jsonl --rounds 8 --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import anthropic

import agent
from ingest import load_rounds
from runner import run_agent
from sessions import Call, InMemoryStore, RoundOutcome, build_grounding


async def _run_one(client, match, config, model, extra_context):
    """Consume run_agent to completion, returning the final event's data."""
    final = None
    async for ev in run_agent(match, config=config, request_id="replay", client=client,
                              model=model, max_attempts=3, extra_context=extra_context):
        if ev.event == "final":
            final = ev.data
    return final


async def run_replay(rounds: list, client, *, config="full", model=None,
                     map_name="unknown", us_team="us", window=5, compact_batch=5,
                     max_rounds: Optional[int] = None, on_event=None):
    """Drive rounds through a session. Returns (report_rows, session)."""
    model = model or agent.MODEL
    store = InMemoryStore(window=window, compact_batch=compact_batch)
    s = await store.create(map=map_name, us_team=us_team, window=window)
    sid = s.id
    rows = []
    todo = rounds if max_rounds is None else rounds[:max_rounds]

    for r in todo:
        # get() returns a snapshot; re-read each round rather than holding a stale object
        s = await store.get(sid)
        recent, extra_context, compaction = build_grounding(s, compact_batch)
        if compaction and await store.claim_compaction(sid, compaction["batch"]) and on_event:
            on_event("compaction", {"round": r["round"], **compaction})
        score = s.outcomes[-1].score_after if s.outcomes else {"us": 0, "them": 0}
        match = {"map": map_name, "side": r["our_side"], "score": score,
                 "economy": r["economy"], "recent_rounds": recent}

        final = await _run_one(client, match, config, model, extra_context)
        plan = final["plan"]
        await store.record_call(sid, Call(round=r["round"], buy_type=plan["buy_type"],
                                          buy=plan["buy"], cost=final["cost"],
                                          approved=final["approved"], attempts=final["attempts"]))

        actual_buy = r["buy"].get("us")
        row = {
            "round": r["round"], "our_side": r["our_side"],
            "money": r["economy"]["us_avg"],
            "agent_buy_type": plan["buy_type"], "agent_buy": plan["buy"],
            "agent_cost": final["cost"], "approved": final["approved"],
            "attempts": final["attempts"],
            "actual_buy_type": actual_buy,
            "match": (plan["buy_type"] == actual_buy),
            "result": r["result"], "detail": r["detail"],
            "cost_usd": final["usage"]["cost_usd"],
        }
        rows.append(row)
        if on_event:
            on_event("round", row)
        # reveal the real outcome to the ledger AFTER the agent has committed its call
        await store.append_outcome(sid, RoundOutcome(**{k: r[k] for k in (
            "round", "our_side", "their_side", "result", "winner",
            "score_after", "economy", "buy", "detail")}))

    return rows, await store.get(sid)


def render_report(rows, meta) -> str:
    lines = [f"# Match replay — {meta.get('map','?')} · us = {meta.get('us_team','?')}", "",
             f"Doer: `{meta.get('model')}` · config: `{meta.get('config')}` · "
             f"rounds replayed: {len(rows)}", "",
             "| rd | side | $ | agent call | actual buy | match? | result |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        call = f"{r['agent_buy_type']} ({', '.join(r['agent_buy'])}, ${r['agent_cost']})"
        lines.append(f"| {r['round']} | {r['our_side']} | {r['money']} | {call} | "
                     f"{r['actual_buy_type']} | {'✓' if r['match'] else '✗'} | {r['result']} |")
    matched = sum(1 for r in rows if r["match"])
    wins = sum(1 for r in rows if r["result"] == "win")
    total_cost = sum(r["cost_usd"] for r in rows)
    lines += ["",
              f"buy_type agreement with the team's real buy: {matched}/{len(rows)}",
              f"rounds the team actually won: {wins}/{len(rows)}",
              f"replay API cost: ${total_cost:.4f}",
              "",
              "*Agreement is not accuracy: the team's real buy is not ground truth for the "
              "optimal call. This is a demo of the closed loop, not an eval.*"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a rounds.jsonl through a coaching session")
    ap.add_argument("rounds_file", help="path to matches/<id>/rounds.jsonl")
    ap.add_argument("--rounds", type=int, help="replay only the first N rounds")
    ap.add_argument("--config", default="full", help="ablation config for the doer")
    ap.add_argument("--model", help="doer model (default: agent.MODEL)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--compact-batch", type=int, default=5,
                    help="fold this many out-of-window rounds per compaction event")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    agent.load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (real replay makes model calls).")

    rounds_path = Path(args.rounds_file)
    rounds = load_rounds(rounds_path)
    meta_path = rounds_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    map_name = meta.get("map", "unknown")
    us_team = meta.get("us_team", "us")
    model = args.model or agent.MODEL

    def on_event(kind, data):
        if kind == "compaction":
            print(f"  [compaction @ r{data['round']}] {data['record']}, "
                  f"trend {data['economy_trend']}")
        else:
            m = "✓" if data["match"] else "✗"
            print(f"  r{data['round']:>2} {data['our_side']:<2} ${data['money']:<5} "
                  f"agent={data['agent_buy_type']:<9} actual={str(data['actual_buy_type']):<9} "
                  f"{m}  -> {data['result']}")

    async def go():
        client = anthropic.AsyncAnthropic()
        try:
            return await run_replay(rounds, client, config=args.config, model=model,
                                    map_name=map_name, us_team=us_team, window=args.window,
                                    compact_batch=args.compact_batch,
                                    max_rounds=args.rounds, on_event=on_event)
        finally:
            await client.close()

    rows, _ = asyncio.run(go())
    meta_out = {**meta, "model": model, "config": args.config}
    report = render_report(rows, meta_out)
    out = rounds_path.parent / "replay_report.md"
    out.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
