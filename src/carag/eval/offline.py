"""Offline re-arbitration over the claims + NLI matrices cached in a run's predictions.

Arbitration is cheap once claims and NLI scores exist, so thresholds and reliability tables can be
changed and re-scored without loading any model. Scores are for the templated (arbitration_only) output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..arbitration import EvidenceArbiter
from ..data import load_instances
from ..prompts import parse_output, render_arbitration
from ..schema import Claim, Passage
from .metrics import aggregate, score_instance


def load_traces(run: str | Path) -> tuple[dict[str, dict], list[tuple]]:
    """-> (instances by qid, [(prediction row, claims, entail, contra, relevance | None)]), one per question."""
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    by_qid = {x["qid"]: x for x in load_instances(config["data"])}
    to_np = lambda m: np.array([[np.nan if x is None else x for x in row] for row in m], dtype=float)
    seen, cached = set(), []
    with open(run / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("trace")
            if not t or r["qid"] in seen:
                continue
            seen.add(r["qid"])
            claims = [Claim(Passage(c["pid"], "", c["source"]), c["answer"], c["confidence"], c["has_answer"], "")
                      for c in t["claims"]]
            rel = [c.get("relevance") for c in t["claims"]]
            cached.append((r, claims, to_np(t["entail"]), to_np(t["contra"]), None if None in rel else rel))
    if not cached:
        raise SystemExit(f"{run} has no arbitration traces (run eval with conflict_aware or arbitration_only)")
    return by_qid, cached


def rearbitrate(cached: list[tuple], by_qid: dict[str, dict], arbiter: EvidenceArbiter, name: str = "offline") -> dict:
    scored = []
    for r, claims, E, C, rel in cached:
        out = render_arbitration(arbiter.arbitrate(claims, E, C, rel), claims)
        p = parse_output(out)
        res = {**r, "system": name, "output": out, "answer": p.answer, "flagged_conflict": p.flagged_conflict}
        scored.append(score_instance(by_qid[r["qid"]], res))
    return aggregate(scored)[name]
