#!/usr/bin/env python
"""Ablation harness for cs2-coach-agent.

Runs every scenario under four configs (full / no_verifier / no_hard_gate / doer_only)
and reports whether each guard rail actually earns its cost.

  python eval.py --estimate      # cost estimate, no API calls
  python eval.py                 # full run (uses disk cache)
  python eval.py --no-cache      # force fresh API calls
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from agent import (
    DOER_SCHEMA, DOER_SYSTEM, DOER_TOOLS, MAX_TOOL_ITERATIONS, MODEL, PRICES,
    PRICES_AS_OF, PRICES_SOURCE, VERIFIER_SCHEMA, VERIFIER_SYSTEM, budget_check,
    build_doer_prompt, build_verifier_prompt, load_env, run_tool,
)

ROOT = Path(__file__).parent
SCENARIOS_DIR, RESULTS_DIR, CACHE_DIR = ROOT / "scenarios", ROOT / "results", ROOT / ".cache"

# Haiku for the judge: it answers one narrow yes/no against a fixed rubric. Paying
# Opus rates 96 times for that is pure waste, and a weaker judge is *safer* here —
# it cannot rationalize a vague reason into a specific one the way Opus might.
JUDGE_MODEL = "claude-haiku-4-5"
MAX_ATTEMPTS = 3          # outer retries, matches agent.py
REQUEST_TIMEOUT = 240.0   # a doer turn at high effort can legitimately run minutes
MAX_RETRIES = 5           # per-request retries; 3 was not enough to ride out 529 bursts
PRICING = {  # USD per token, from the model tables
    "claude-opus-4-8": (5 / 1e6, 25 / 1e6),
    "claude-haiku-4-5": (1 / 1e6, 5 / 1e6),
    "claude-sonnet-5": (3 / 1e6, 15 / 1e6),
}


def price_of(model: str):
    for prefix, rates in PRICING.items():
        if model.startswith(prefix):
            return rates
    raise KeyError(f"no pricing for {model}; add it to PRICING")


def model_params(model: str) -> dict:
    """Thinking/effort differ by model generation. Haiku 4.5 predates adaptive thinking
    and rejects `effort` outright, so a single param set would 400 every Haiku run."""
    if model.startswith("claude-haiku-4-5"):
        return {"thinking": {"type": "enabled", "budget_tokens": 2000}}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}


def merge_output_config(base: dict, extra: dict) -> dict:
    """output_config carries both effort and format; on models without effort we must
    still be able to attach the response schema."""
    oc = dict(base.get("output_config", {}))
    oc.update(extra)
    out = {k: v for k, v in base.items() if k != "output_config"}
    out["output_config"] = oc
    return out
# Prompt-caching economics: a cache write costs 1.25x base input, a read 0.1x.
# Break-even is 2 requests against the same prefix; the doer loop resends its
# conversation ~5x per run, so it clears that easily.
CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.10
MIN_CACHEABLE_TOKENS = 4096  # Opus 4.8 / Haiku 4.5. Shorter prefixes silently do not cache.

log = logging.getLogger("eval")

CONFIGS = {
    "full":         {"hard_gate": True,  "verifier": True},
    "no_verifier":  {"hard_gate": True,  "verifier": False},
    "no_hard_gate": {"hard_gate": False, "verifier": True},
    "doer_only":    {"hard_gate": False, "verifier": False},
}

JUDGE_SYSTEM = """You grade ONE field from a CS2 coaching plan: loss_reason. Decide only
whether it is GENERIC.

generic = true if it could be pasted into any losing match without changing meaning.
  Examples: "we played badly", "poor execution", "lost the duels", "bad positioning",
  "the enemy outplayed us", "economy issues".
generic = false ONLY if it names at least one concrete, checkable particular: a map
  location (mid, banana, ramp, A site), a named mechanism (AWP holding window, no
  utility thrown, lurker dying alone, falling for a fake), or an explicit repeated
  pattern across specific rounds.

A reason that names a specific failure is NOT generic even if it is also wordy.
A reason listing round numbers but describing them only in vague terms IS generic.
Judge the text only. Do not consider whether it is tactically correct."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"generic": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["generic", "why"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- infra

def setup_logging(verbose: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.DEBUG if verbose else logging.INFO)
    stderr.setFormatter(fmt)
    # The file always gets DEBUG: when a run misbehaves you want the payloads
    # after the fact, and re-running to reproduce costs real money.
    fileh = logging.FileHandler(RESULTS_DIR / "eval.log", encoding="utf-8")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    log.setLevel(logging.DEBUG)
    log.addHandler(stderr)
    log.addHandler(fileh)
    log.propagate = False


def cache_path(kind: str, material: dict) -> Path:
    # The key hashes the *entire* prompt payload, so any edit to a system prompt,
    # tool schema or scenario silently invalidates stale entries instead of
    # serving results that no longer correspond to the code.
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{kind}_{digest}.json"


async def api_call(client: anthropic.AsyncAnthropic, sem: asyncio.Semaphore, **kwargs):
    """One API call with timeout, bounded concurrency and backoff on transient errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Semaphore acquired per attempt, never held across a sleep — otherwise a
            # backing-off task would keep a concurrency slot idle and stall the pool.
            async with sem:
                return await asyncio.wait_for(client.messages.create(**kwargs), REQUEST_TIMEOUT)
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APITimeoutError, anthropic.APIConnectionError,
                asyncio.TimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            # Exponential backoff clears rate limits; jitter stops 4 workers that
            # hit the same 429 from retrying in lockstep and re-triggering it.
            delay = 2.0 * (2 ** attempt) + random.uniform(0, 1.5)
            log.warning("%s on attempt %d/%d, retrying in %.1fs",
                        type(e).__name__, attempt + 1, MAX_RETRIES, delay)
            await asyncio.sleep(delay)


def parse_json_response(resp) -> dict:
    # next() raising StopIteration inside a coroutine surfaces as an opaque
    # "coroutine raised StopIteration" RuntimeError, which hides the real cause:
    # the model returned no text block at all (e.g. it emitted only tool_use).
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        kinds = [b.type for b in resp.content]
        raise ValueError(f"no text block in response; got {kinds}, stop_reason={resp.stop_reason}")
    return json.loads(text)


def add_usage(acc: dict, model: str, resp) -> None:
    """Price cache reads/writes separately — billing them at the base rate would
    misreport the spend the caching was added to reduce."""
    u = resp.usage
    i, o = u.input_tokens, u.output_tokens
    write = getattr(u, "cache_creation_input_tokens", 0) or 0
    read = getattr(u, "cache_read_input_tokens", 0) or 0
    ci, co = price_of(model)
    acc["in"] += i + write + read       # total prompt tokens actually processed
    acc["out"] += o
    acc["cache_write"] += write
    acc["cache_read"] += read
    acc["cost"] += i * ci + write * ci * CACHE_WRITE_MULT + read * ci * CACHE_READ_MULT + o * co


# --------------------------------------------------------------------------- steps

async def doer_step(client, sem, match, feedback, key_extra, use_cache, doer_model=MODEL):
    """agent.doer's tool loop, async. Returns (plan, usage, seconds, cached)."""
    prompt = build_doer_prompt(match, feedback)
    # doer_model is part of the key: without it a Haiku run would silently serve
    # Opus's cached plans and the whole capability experiment would be a no-op.
    # PRICES is part of the key too, and less obviously: prices never appear in the
    # prompt (the model fetches them through tools), so a price correction would
    # otherwise leave every cached plan intact and the "re-run" would change nothing.
    path = cache_path("doer", {"prompt": prompt, "system": DOER_SYSTEM, "model": doer_model,
                               "tools": DOER_TOOLS, "schema": DOER_SCHEMA, "prices": PRICES,
                               **key_extra})
    if use_cache and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        return c["value"], c["usage"], c["seconds"], True

    usage = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0, "cost": 0.0}
    t0 = time.monotonic()
    messages = [{"role": "user", "content": prompt}]
    params = model_params(doer_model)
    for _ in range(MAX_TOOL_ITERATIONS):
        # cache_control auto-places on the last cacheable block = the end of the
        # conversation so far, so each turn reads the prefix its predecessor wrote.
        # Only the doer gets this: the verifier (660 tok) and judge (216 tok) prompts
        # are under the 4096-token minimum and would never produce a cache hit.
        r = await api_call(client, sem, model=doer_model, max_tokens=4000, system=DOER_SYSTEM,
                           tools=DOER_TOOLS, cache_control={"type": "ephemeral"},
                           messages=messages, **params)
        add_usage(usage, doer_model, r)
        messages.append({"role": "assistant", "content": r.content})
        tool_uses = [b for b in r.content if b.type == "tool_use"]
        if not tool_uses:
            break
        results = []
        for tu in tool_uses:
            out = run_tool(match, tu.name, tu.input)
            log.debug("tool %s(%s) -> %s", tu.name, tu.input, out)
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": json.dumps(out, ensure_ascii=False)})
        messages.append({"role": "user", "content": results})
    else:
        log.warning("doer hit the %d-iteration cap", MAX_TOOL_ITERATIONS)

    messages.append({"role": "user", "content": "Now output the final plan for the next round."})
    final_params = merge_output_config(params, {"format": {"type": "json_schema",
                                                           "schema": DOER_SCHEMA}})
    # A weaker doer may answer the "give me the plan" turn with yet another tool call
    # instead of the plan. Serve the tool, ask again. Without this the turn yields no
    # text block and the scenario dies -- and it dies precisely on the hardest cases,
    # which would silently bias the surviving sample.
    for _ in range(3):
        final = await api_call(client, sem, model=doer_model, max_tokens=4000, system=DOER_SYSTEM,
                               tools=DOER_TOOLS, cache_control={"type": "ephemeral"},
                               messages=messages, **final_params)
        add_usage(usage, doer_model, final)
        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if not tool_uses:
            break
        log.debug("doer called tools on the final turn; serving them and re-asking")
        messages.append({"role": "assistant", "content": final.content})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tu.id,
             "content": json.dumps(run_tool(match, tu.name, tu.input), ensure_ascii=False)}
            for tu in tool_uses]})
    plan = parse_json_response(final)
    seconds = time.monotonic() - t0
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"value": plan, "usage": usage, "seconds": seconds},
                               ensure_ascii=False), encoding="utf-8")
    return plan, usage, seconds, False


async def verifier_step(client, sem, match, plan, cost, key_extra, use_cache):
    prompt = build_verifier_prompt(match, plan, cost)
    path = cache_path("verifier", {"prompt": prompt, "system": VERIFIER_SYSTEM,
                                   "schema": VERIFIER_SCHEMA, **key_extra})
    if use_cache and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        return c["value"], c["usage"], c["seconds"], True
    usage = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0, "cost": 0.0}
    t0 = time.monotonic()
    r = await api_call(client, sem, model=MODEL, max_tokens=2000, system=VERIFIER_SYSTEM,
                       thinking={"type": "disabled"},
                       output_config={"effort": "low",
                                      "format": {"type": "json_schema", "schema": VERIFIER_SCHEMA}},
                       messages=[{"role": "user", "content": prompt}])
    add_usage(usage, MODEL, r)
    verdict, seconds = parse_json_response(r), time.monotonic() - t0
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"value": verdict, "usage": usage, "seconds": seconds},
                               ensure_ascii=False), encoding="utf-8")
    return verdict, usage, seconds, False


async def judge_step(client, sem, loss_reason, use_cache):
    """Metric only — never gates the pipeline, so it cannot contaminate the ablation."""
    path = cache_path("judge", {"reason": loss_reason, "system": JUDGE_SYSTEM})
    if use_cache and path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
        return c["value"], c["usage"], c["seconds"], True
    usage = {"in": 0, "out": 0, "cache_write": 0, "cache_read": 0, "cost": 0.0}
    t0 = time.monotonic()
    r = await api_call(client, sem, model=JUDGE_MODEL, max_tokens=500, system=JUDGE_SYSTEM,
                       output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                       messages=[{"role": "user", "content": f"loss_reason:\n{loss_reason}"}])
    add_usage(usage, JUDGE_MODEL, r)
    verdict, seconds = parse_json_response(r), time.monotonic() - t0
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"value": verdict, "usage": usage, "seconds": seconds},
                               ensure_ascii=False), encoding="utf-8")
    return verdict, usage, seconds, False


# --------------------------------------------------------------------------- scoring

def tag_coverage(loss_reason: str, tag_groups) -> float:
    """Fraction of required concepts named. Each group is a list of accepted aliases
    (English + Chinese), because the model answers in either language."""
    if not tag_groups:
        return 1.0
    low = f" {loss_reason.lower()} "
    hits = sum(1 for group in tag_groups if any(a.lower() in low for a in group))
    return hits / len(tag_groups)


async def run_scenario(client, sem, scen, config_name, cfg, use_cache, doer_model=MODEL):
    sid = scen["id"]
    # The "expected" block is the answer key — it must never reach the model.
    match = {k: v for k, v in scen.items() if k not in ("id", "expected")}
    expected = scen["expected"]
    total = {"in": 0, "out": 0, "cost": 0.0}
    seconds, attempts, feedback, plan, verdict = 0.0, 0, None, None, None
    key_extra = {"scenario": sid, "config": config_name}

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            plan, u, s, cached = await doer_step(
                client, sem, match, feedback, {**key_extra, "attempt": attempt}, use_cache,
                doer_model)
            total = {k: total[k] + u[k] for k in total}
            seconds += s
            cost, hard_problems = budget_check(plan, match["economy"]["us_avg"])

            issues = list(hard_problems) if cfg["hard_gate"] else []
            if cfg["verifier"]:
                verdict, u, s, _ = await verifier_step(
                    client, sem, match, plan, cost, {**key_extra, "attempt": attempt}, use_cache)
                total = {k: total[k] + u[k] for k in total}
                seconds += s
                if not verdict["approved"]:
                    issues += verdict["issues"]
            if not issues:
                break
            log.info("[%s/%s] attempt %d rejected: %s", config_name, sid, attempt, issues)
            feedback = issues

        judged, u, s, _ = await judge_step(client, sem, plan["loss_reason"], use_cache)
        total = {k: total[k] + u[k] for k in total}
        seconds += s
        final_cost, final_problems = budget_check(plan, match["economy"]["us_avg"])

        row = {
            "scenario": sid, "group": sid.split("_")[0], "config": config_name,
            "doer_model": doer_model, "status": "OK",
            "buy_type": plan["buy_type"], "buy_ok": plan["buy_type"] in expected["buy_type"],
            "cost": final_cost, "budget_ok": not final_problems, "problems": final_problems,
            "generic": judged["generic"], "tag_cov": tag_coverage(plan["loss_reason"],
                                                                 expected.get("reason_tags", [])),
            "attempts": attempts, "tokens": total["in"] + total["out"], "usd": total["cost"],
            "seconds": seconds, "loss_reason": plan["loss_reason"],
        }
        log.info("[%s/%s] %s buy=%s ok=%s budget_ok=%s generic=%s tags=%.0f%% attempts=%d",
                 config_name, sid, "OK", row["buy_type"], row["buy_ok"], row["budget_ok"],
                 row["generic"], row["tag_cov"] * 100, attempts)
        return row
    except Exception as e:
        # One bad scenario must not lose the other 95 runs' worth of API spend.
        log.exception("[%s/%s] FAILED: %s", config_name, sid, e)
        return {"scenario": sid, "group": sid.split("_")[0], "config": config_name,
                "doer_model": doer_model, "status": "FAILED",
                "error": f"{type(e).__name__}: {e}", "attempts": attempts,
                "tokens": total["in"] + total["out"], "usd": total["cost"], "seconds": seconds}


def aggregate(rows):
    ok = [r for r in rows if r["status"] == "OK"]
    n = len(ok)
    if not n:
        return {"n": 0, "failed": len(rows)}
    return {
        "n": n, "failed": sum(1 for r in rows if r["status"] == "FAILED"),
        "buy_acc": sum(r["buy_ok"] for r in ok) / n,
        "budget_viol": sum(not r["budget_ok"] for r in ok),
        "generic_rate": sum(r["generic"] for r in ok) / n,
        "tag_cov": sum(r["tag_cov"] for r in ok) / n,
        "retries": sum(r["attempts"] - 1 for r in ok) / n,
        "tokens": sum(r["tokens"] for r in ok) / n,
        "seconds": sum(r["seconds"] for r in ok) / n,
        "usd": sum(r["usd"] for r in rows),
    }


def render_markdown(all_rows, started, doer_model=MODEL):
    lines = [f"# cs2-coach-agent eval — {started:%Y-%m-%d %H:%M}", "",
             f"Doer: `{doer_model}` · verifier: `{MODEL}` · judge: `{JUDGE_MODEL}`", "",
             f"Prices: {PRICES_SOURCE} (as_of {PRICES_AS_OF}) · scenarios: "
             f"{len({r['scenario'] for r in all_rows})}", "",
             "| config | n | buy_type acc | budget viol. | generic reason | tag cov. | "
             "retries/scen | tokens/scen | sec/scen | failed |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        rows = [r for r in all_rows if r["config"] == name]
        if not rows:
            continue
        a = aggregate(rows)
        if not a["n"]:
            lines.append(f"| `{name}` | 0 | – | – | – | – | – | – | – | {a['failed']} |")
            continue
        lines.append(
            f"| `{name}` | {a['n']} | {a['buy_acc']:.0%} | {a['budget_viol']} | "
            f"{a['generic_rate']:.0%} | {a['tag_cov']:.0%} | {a['retries']:.2f} | "
            f"{a['tokens']:,.0f} | {a['seconds']:.0f} | {a['failed']} |")
    lines += ["", "*budget viol. = scenarios whose final plan fails the deterministic price "
              "check. sec/scen = summed step latency, not wall-clock (runs are concurrent).*", ""]

    lines += ["## Per-scenario", "", "| scenario | config | buy_type | exp? | budget | generic | "
              "tags | tries |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(all_rows, key=lambda r: (r["scenario"], r["config"])):
        if r["status"] != "OK":
            lines.append(f"| {r['scenario']} | `{r['config']}` | FAILED | – | – | – | – | "
                         f"{r['attempts']} |")
            continue
        lines.append(f"| {r['scenario']} | `{r['config']}` | {r['buy_type']} | "
                     f"{'yes' if r['buy_ok'] else 'NO'} | {'ok' if r['budget_ok'] else 'VIOL'} | "
                     f"{'GENERIC' if r['generic'] else 'ok'} | {r['tag_cov']:.0%} | "
                     f"{r['attempts']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- driver

def estimate(scenarios, configs, doer_model=MODEL):
    """Cost model. Assumptions are printed, not hidden, so the bill is checkable."""
    oi, oo = price_of(doer_model)
    vi, vo = price_of(MODEL)          # verifier is pinned to Opus regardless of doer
    ji, jo = price_of(JUDGE_MODEL)
    # Measured from one real sample_match.json run: 5 doer turns whose inputs grow as
    # the conversation is resent (~1.3k, 2.5k, 4k, 6k, 8k) = ~22k prompt tokens total.
    turns = [1300, 2500, 4000, 6000, 8000]
    d_out, v_in, v_out, j_in, j_out = 2600, 660, 250, 400, 80

    uncached = sum(turns) * oi + d_out * oo
    # With caching: turn N's prefix is turn N-1's total. A prefix is only cacheable at
    # >= MIN_CACHEABLE_TOKENS; below that the marker is inert and we pay base rate.
    cached_cost, writes, reads, base = 0.0, 0, 0, 0
    for n, total_in in enumerate(turns):
        prefix = turns[n - 1] if n > 0 else 0
        if prefix >= MIN_CACHEABLE_TOKENS:
            reads += prefix
            writes += total_in - prefix        # the new suffix gets written for next turn
        else:
            base += total_in
    cached_cost = (base * oi + writes * oi * CACHE_WRITE_MULT
                   + reads * oi * CACHE_READ_MULT + d_out * oo)
    ver = v_in * vi + v_out * vo               # 660 tok < 4096 -> not cacheable, base rate
    judge = j_in * ji + j_out * jo             # 216 tok system -> not cacheable, base rate

    print(f"\nDoer: {doer_model} · verifier: {MODEL} · judge: {JUDGE_MODEL}")
    print(f"Pricing: input ${oi*1e6:.2f}/Mtok · cache WRITE {CACHE_WRITE_MULT}x (${oi*1e6*CACHE_WRITE_MULT:.2f})"
          f" · cache READ {CACHE_READ_MULT}x (${oi*1e6*CACHE_READ_MULT:.2f}) · output ${oo*1e6:.2f}/Mtok")
    print(f"Min cacheable prefix: {MIN_CACHEABLE_TOKENS:,} tok. Measured prefixes: doer tools+system "
          f"1,061 (inert alone) · verifier 660 (inert) · judge 216 (inert).")
    print(f"Doer conversation crosses the minimum at turn 4, so only turns 4-5 read cache.")
    print(f"\nPer doer run: ${uncached:.3f} uncached -> ${cached_cost:.3f} cached "
          f"({base:,} base + {writes:,} write + {reads:,} read tok)")
    print(f"Verifier ${ver:.4f}/call · judge ${judge:.4f}/call (both uncacheable)\n")

    print(f"{'config':<14} {'scen':>5} {'$/scen':>9} {'$ total':>9}   {'(was)':>8}")
    total = was_total = 0.0
    for name in configs:
        cfg = CONFIGS[name]
        retry = 1.25 if (cfg["verifier"] or cfg["hard_gate"]) else 1.0
        per = (cached_cost + judge + (ver if cfg["verifier"] else 0)) * retry
        old = (uncached + judge + (ver if cfg["verifier"] else 0)) * retry
        total += per * len(scenarios)
        was_total += old * len(scenarios)
        print(f"{name:<14} {len(scenarios):>5} {per:>8.2f} {per*len(scenarios):>9.2f}   "
              f"{old*len(scenarios):>8.2f}")
    print(f"\n{'TOTAL':<14} {'':>5} {'':>9} {total:>9.2f}   {was_total:>8.2f}")
    print(f"projected saving: ${was_total-total:.2f} ({(1-total/was_total):.0%}). "
          f"Cached disk entries cost $0.\n")


async def main_async(args):
    scenarios = sorted((json.loads(p.read_text(encoding="utf-8")) for p in SCENARIOS_DIR.glob("*.json")),
                       key=lambda s: s["id"])
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep or s["id"].split("_")[0] in keep]
    configs = args.configs.split(",") if args.configs else list(CONFIGS)
    if not scenarios:
        sys.exit("no scenarios matched")

    if args.estimate:
        estimate(scenarios, configs, args.doer_model)
        return

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.AsyncAnthropic()
    # One semaphore shared by every task: it bounds *in-flight requests*, which is what
    # the rate limit actually counts — not tasks, not scenarios.
    sem = asyncio.Semaphore(args.concurrency)
    started, all_rows = datetime.now(), []
    for name in configs:
        # Configs run sequentially so they never contend for the same rate-limit budget;
        # a config that ran while another saturated the pool would show inflated latency.
        log.info("=== config %s (%d scenarios, concurrency %d) ===", name, len(scenarios), args.concurrency)
        t0 = time.monotonic()
        rows = await asyncio.gather(*[
            run_scenario(client, sem, s, name, CONFIGS[name], not args.no_cache, args.doer_model)
            for s in scenarios])
        all_rows += rows
        a = aggregate(rows)
        log.info("=== %s done in %.0fs: buy_acc=%.0f%% budget_viol=%d generic=%.0f%% $%.2f ===",
                 name, time.monotonic() - t0, a.get("buy_acc", 0) * 100, a.get("budget_viol", 0),
                 a.get("generic_rate", 0) * 100, a.get("usd", 0))
    await client.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    md = render_markdown(all_rows, started, args.doer_model)
    stem = f"eval_{started:%Y%m%d_%H%M%S}" + (f"_{args.tag}" if args.tag else "")
    out = RESULTS_DIR / f"{stem}.md"
    out.write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"{stem}.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + md.split("## Per-scenario")[0])
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description="cs2-coach-agent ablation eval")
    p.add_argument("--estimate", action="store_true", help="print cost estimate, make no API calls")
    p.add_argument("--no-cache", action="store_true", help="ignore the disk cache")
    p.add_argument("--concurrency", type=int, default=4, help="max in-flight API requests")
    p.add_argument("--doer-model", default=MODEL,
                   help="model for the Doer only; verifier and judge are fixed")
    p.add_argument("--tag", default="", help="suffix for the results filename")
    p.add_argument("--configs", help="comma-separated subset of configs")
    p.add_argument("--scenarios", help="comma-separated scenario ids or group prefixes")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG to stderr")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(args.verbose)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
