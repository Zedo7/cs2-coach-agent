#!/usr/bin/env python
"""Parse a CS2 demo (.dem) into our round-stream format.

Parser: demoparser2. Picked over awpy because it is CS2-native, Rust-backed, and
actively maintained (awpy has historically lagged CS2 updates and churned its API);
for "economy + results + details only" it exposes exactly the primitives we need
(parse_event for round_end/bomb_*, parse_ticks for per-player balance + equip value)
with the fewest moving abstractions. awpy is the fallback only if demoparser2 chokes
on a specific demo.

Output: matches/<match_id>/rounds.jsonl  (one round-state per line) + meta.json.

SCOPE: economy + results + details only. No positions, coordinates, heatmaps, or ML
features -- see the RICHER-EXTRACTION HOOK below for where trajectory/utility-usage
extraction would plug in later.

The demo-parsing (parse_demo) needs a real file and is confirmed via `--probe`. The
round-assembly (build_round_record, detail_string, validate_round_record) is pure and
unit-tested against a hand-made fixture -- no demo in the test suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from agent import classify_equip_value

ROOT = Path(__file__).parent
MATCHES = ROOT / "matches"

# demoparser2 field names, centralized so the one confirmation step against a real demo
# (`python ingest.py <demo> --probe`) touches a single place. team_num: 2 = T, 3 = CT.
EV_ROUND_END = "round_end"
EV_ROUND_START = "round_start"
EV_FREEZE_END = "round_freeze_end"
EV_BOMB_PLANTED = "bomb_planted"
EV_BOMB_DEFUSED = "bomb_defused"
EV_BOMB_EXPLODED = "bomb_exploded"
TICK_PROPS = ["balance", "current_equip_value", "team_num", "team_clan_name"]
T_SIDE, CT_SIDE = 2, 3
SIDE_NAME = {T_SIDE: "T", CT_SIDE: "CT"}
# Confirmed via --probe against a real demo: round_end.winner is the string "T"/"CT"
# (not an int team_num), and round_end carries a phantom round 0 (winner=None, warmup)
# that must be dropped or the score and numbering shift by one.
WINNER_SIDE = {"T": T_SIDE, "CT": CT_SIDE}

REQUIRED_FIELDS = ("round", "our_side", "their_side", "result", "winner",
                   "score_after", "economy", "buy", "detail")


# --------------------------------------------------------------------------- pure logic

def detail_string(winner_side: str, bomb: Optional[str], site: Optional[str]) -> str:
    """Short auto-generated summary from the cheap signals a parser exposes: which side
    won and what the bomb did. RICHER-EXTRACTION HOOK: clutch detection (1vN), entry
    trades, and utility usage would extend this -- they need per-death / per-grenade
    tracking, deliberately out of scope here."""
    if bomb == "defused":
        tail = "bomb defused"
    elif bomb == "exploded":
        tail = f"bomb detonated{f' at {site}' if site else ''}"
    elif bomb == "planted":
        # planted but neither defused nor detonated -> round ended by elimination/time with
        # the bomb down. Winner-agnostic: the planting side can win this by trading out the
        # retake, so "then lost" would be wrong.
        tail = f"post-plant{f' at {site}' if site else ''} — elimination / time"
    else:
        tail = "elimination / time"
    return f"{winner_side} win — {tail}"


def build_round_record(round_no: int, us_side: int, winner_side: int,
                       us_balances: list[int], them_balances: list[int],
                       us_equip: list[int], them_equip: list[int],
                       score_after: dict, bomb: Optional[str], site: Optional[str]) -> dict:
    """Assemble one rounds.jsonl record from already-extracted per-player numbers.
    Pure: no parser, no I/O -- this is the unit-tested mapping layer."""
    their_side = CT_SIDE if us_side == T_SIDE else T_SIDE
    won = winner_side == us_side

    def avg(xs):
        return round(mean(xs)) if xs else 0

    return {
        "round": round_no,
        "our_side": SIDE_NAME[us_side],
        "their_side": SIDE_NAME[their_side],
        "result": "win" if won else "loss",
        "winner": "us" if won else "them",
        "score_after": score_after,
        # Economy = money AVAILABLE at round start (balance before buys), which is what a
        # coach reasons about. Buy type is read from equipment fielded at freeze end.
        "economy": {"us_avg": avg(us_balances), "them_avg": avg(them_balances)},
        "buy": {"us": classify_equip_value(avg(us_equip)),
                "them": classify_equip_value(avg(them_equip))},
        "detail": detail_string(SIDE_NAME[winner_side], bomb, site),
    }


def validate_round_record(rec: dict) -> list[str]:
    """Schema check for one rounds.jsonl line. Used by tests and by load_rounds."""
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in rec:
            problems.append(f"missing field: {f}")
    if problems:
        return problems
    if rec["our_side"] not in ("T", "CT") or rec["their_side"] not in ("T", "CT"):
        problems.append(f"bad side: {rec['our_side']}/{rec['their_side']}")
    if rec["our_side"] == rec["their_side"]:
        problems.append("our_side == their_side")
    if rec["result"] not in ("win", "loss"):
        problems.append(f"bad result: {rec['result']}")
    if (rec["winner"] == "us") != (rec["result"] == "win"):
        problems.append("winner/result disagree")
    for k in ("us_avg", "them_avg"):
        if not isinstance(rec["economy"].get(k), int) or rec["economy"][k] < 0:
            problems.append(f"bad economy.{k}")
    for k in ("us", "them"):
        if rec["buy"].get(k) not in (None, "pistol", "eco", "half_buy", "force_buy", "full_buy"):
            problems.append(f"bad buy.{k}: {rec['buy'].get(k)}")
    return problems


def load_rounds(path: Path) -> list[dict]:
    """Load and schema-validate a rounds.jsonl. Raises on the first malformed line."""
    rounds = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        probs = validate_round_record(rec)
        if probs:
            raise ValueError(f"{path}:{i}: {'; '.join(probs)}")
        rounds.append(rec)
    return rounds


def write_match(match_id: str, rounds: list[dict], meta: dict) -> Path:
    out = MATCHES / match_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rounds.jsonl").open("w", encoding="utf-8") as f:
        for r in rounds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- demo parsing

def _load_parser(path: str):
    from demoparser2 import DemoParser  # imported lazily so tests never need the dep
    return DemoParser(path)


def probe(path: str) -> None:
    """Confirm parser field names against a real demo -- the one verification step the
    ingest depends on. Prints map, relevant events, and their columns."""
    p = _load_parser(path)
    print("header map:", p.parse_header().get("map_name"))
    events = p.list_game_events()
    interesting = [e for e in events if e.startswith(("round_", "bomb_"))]
    print("round_/bomb_ events present:", interesting)
    for ev in (EV_ROUND_END, EV_ROUND_START, EV_FREEZE_END, EV_BOMB_PLANTED):
        try:
            df = p.parse_event(ev)
            print(f"\n[{ev}] rows={len(df)} cols={list(df.columns)}")
            if len(df):
                print(df.head(2).to_dict("records"))
        except Exception as e:
            print(f"\n[{ev}] ERROR: {type(e).__name__}: {e}")
    try:
        tdf = p.parse_ticks(TICK_PROPS, ticks=[100000])
        print(f"\n[ticks props {TICK_PROPS}] cols={list(tdf.columns)}")
    except Exception as e:
        print(f"\n[ticks] ERROR: {type(e).__name__}: {e}")


def parse_demo(path: str, us_team: Optional[str] = None, max_rounds: Optional[int] = None):
    """Extract (meta, rounds) from a demo. Needs a real file; confirmed via --probe.

    Kept deliberately linear and defensive: exact column names are verified against the
    target demo before a real run, and each lookup falls back rather than crashing."""
    p = _load_parser(path)
    header = p.parse_header()
    map_name = header.get("map_name", "unknown")

    # Drop the phantom round 0 / any row without a real winner (confirmed via --probe).
    round_ends = [r for r in p.parse_event(EV_ROUND_END).to_dict("records")
                  if r.get("winner") in WINNER_SIDE and int(r.get("round", 0)) >= 1]
    freeze_ends = {int(r.get("round", i + 1)): int(r["tick"])
                   for i, r in enumerate(p.parse_event(EV_FREEZE_END).to_dict("records"))}
    round_starts = {int(r.get("round", i + 1)): int(r["tick"])
                    for i, r in enumerate(p.parse_event(EV_ROUND_START).to_dict("records"))}

    def bomb_events(ev):
        try:
            return p.parse_event(ev).to_dict("records")
        except Exception:
            return []
    planted, defused, exploded = (bomb_events(EV_BOMB_PLANTED),
                                  bomb_events(EV_BOMB_DEFUSED), bomb_events(EV_BOMB_EXPLODED))

    # Per-tick per-player money/equip, pulled once for all freeze-end + round-start ticks.
    wanted_ticks = sorted(set(freeze_ends.values()) | set(round_starts.values()))
    ticks = p.parse_ticks(TICK_PROPS, ticks=wanted_ticks).to_dict("records")
    by_tick: dict[int, list[dict]] = {}
    for row in ticks:
        by_tick.setdefault(int(row["tick"]), []).append(row)

    # Fix "us": the team on CT in round 1, unless --us-team names a clan.
    def side_at(tick, team_clan_name):
        return [r for r in by_tick.get(tick, []) if r.get("team_clan_name") == team_clan_name]

    r1_start = round_starts.get(1) or (wanted_ticks[0] if wanted_ticks else None)
    clans = {r.get("team_clan_name") for r in by_tick.get(r1_start, []) if r.get("team_num") in (T_SIDE, CT_SIDE)}
    clans.discard(None)
    if us_team and us_team in clans:
        us_clan = us_team
    else:
        ct_players = [r for r in by_tick.get(r1_start, []) if r.get("team_num") == CT_SIDE]
        us_clan = ct_players[0]["team_clan_name"] if ct_players else (sorted(clans)[0] if clans else None)
    them_clan = next((c for c in clans if c != us_clan), None)

    rounds, us_score, them_score = [], 0, 0
    total = len(round_ends) if max_rounds is None else min(max_rounds, len(round_ends))
    for i in range(total):
        rno = int(round_ends[i].get("round", i + 1))
        winner_side = WINNER_SIDE[round_ends[i]["winner"]]
        ftick, stick = freeze_ends.get(rno), round_starts.get(rno)

        start_rows = by_tick.get(stick, []) if stick else []
        freeze_rows = by_tick.get(ftick, []) if ftick else []
        us_start = [r for r in start_rows if r.get("team_clan_name") == us_clan]
        them_start = [r for r in start_rows if r.get("team_clan_name") == them_clan]
        us_frz = [r for r in freeze_rows if r.get("team_clan_name") == us_clan]
        them_frz = [r for r in freeze_rows if r.get("team_clan_name") == them_clan]

        # Side is read at FREEZE END, not round start: at the halftime boundary (round 13)
        # the side swap has not yet registered at the round-start tick -- FURIA still reads
        # team_num=2 there and only flips to 3 by freeze end (confirmed via diagnostic).
        # Money still comes from round start (available before buys); clan grouping is
        # swap-independent, so only the side integer needs the settled tick.
        us_side = int(us_frz[0]["team_num"]) if us_frz else (
            int(us_start[0]["team_num"]) if us_start else CT_SIDE)
        if winner_side == us_side:
            us_score += 1
        else:
            them_score += 1

        tick_of = round_ends[i].get("tick")
        bomb, site = _bomb_for_round(rno, ftick, tick_of, planted, defused, exploded)
        rounds.append(build_round_record(
            rno, us_side, winner_side,
            [int(r["balance"]) for r in us_start], [int(r["balance"]) for r in them_start],
            [int(r["current_equip_value"]) for r in us_frz],
            [int(r["current_equip_value"]) for r in them_frz],
            {"us": us_score, "them": them_score}, bomb, site))

    meta = {
        "match_id": None, "map": map_name, "us_team": us_clan, "them_team": them_clan,
        "total_rounds": len(rounds), "source_demo": str(path),
        "parser": "demoparser2", "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    return meta, rounds


def _bomb_for_round(rno, freeze_tick, end_tick, planted, defused, exploded):
    """Cheap bomb outcome + site for the detail string, matched by tick window."""
    if freeze_tick is None or end_tick is None:
        return None, None

    def in_round(rows):
        return [r for r in rows if freeze_tick <= int(r.get("tick", -1)) <= end_tick]
    site = None
    pl = in_round(planted)
    if pl:
        raw = pl[0].get("site")
        site = {0: "A", 1: "B"}.get(raw, raw if isinstance(raw, str) else None)
    if in_round(defused):
        return "defused", site
    if in_round(exploded):
        return "exploded", site
    if pl:
        return "planted", site
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse a CS2 demo into rounds.jsonl")
    ap.add_argument("demo", help="path to a .dem file")
    ap.add_argument("--match-id", help="output dir name under matches/ (default: demo stem)")
    ap.add_argument("--us-team", help="clan name to treat as 'us' (default: team starting CT)")
    ap.add_argument("--max-rounds", type=int, help="stop after N rounds")
    ap.add_argument("--probe", action="store_true", help="print parser fields and exit (verify columns)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.probe:
        probe(args.demo)
        return 0

    match_id = args.match_id or Path(args.demo).stem
    meta, rounds = parse_demo(args.demo, us_team=args.us_team, max_rounds=args.max_rounds)
    meta["match_id"] = match_id
    out = write_match(match_id, rounds, meta)
    print(f"wrote {len(rounds)} rounds to {out / 'rounds.jsonl'}")
    print(f"  map={meta['map']} us={meta['us_team']} them={meta['them_team']}")
    if rounds:
        final = rounds[-1]["score_after"]
        print(f"  final score us {final['us']} : {final['them']} them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
