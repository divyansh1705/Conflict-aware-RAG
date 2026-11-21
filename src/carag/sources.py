"""Source reliability priors.

Reliability is a prior probability that a statement from that source is correct. In a real
deployment these would come from a domain-trust list or be learned; here they are a fixed table
that the synthetic benchmark samples sources from.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_RELIABILITY: dict[str, float] = {
    # high-reliability tier
    "wikipedia": 0.90,
    "britannica": 0.90,
    "reuters": 0.85,
    "bbc": 0.85,
    # low-reliability tier
    "personal-blog": 0.35,
    "forum": 0.25,
    "social-media": 0.20,
    "content-farm": 0.20,
}
HIGH_TIER = ["wikipedia", "britannica", "reuters", "bbc"]
LOW_TIER = ["personal-blog", "forum", "social-media", "content-farm"]
UNKNOWN_RELIABILITY = 0.5


class SourceReliability:
    def __init__(self, table: dict[str, float] | None = None, default: float = UNKNOWN_RELIABILITY):
        self.table = dict(DEFAULT_RELIABILITY if table is None else table)
        self.default = default

    def __call__(self, source: str) -> float:
        return self.table.get(source, self.default)

    @classmethod
    def from_json(cls, path: str | Path) -> "SourceReliability":
        return cls(json.loads(Path(path).read_text()))
