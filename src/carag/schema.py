"""Core data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Decision = Literal["answer", "disagree", "abstain"]


@dataclass
class Passage:
    pid: str
    text: str
    source: str = "unknown"
    title: str = ""
    meta: dict = field(default_factory=dict)  # e.g. {"qid":..., "role": "gold"|"conflict"|"distractor"}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Passage":
        return cls(**d)


@dataclass
class Retrieved:
    passage: Passage
    score: float
    rank: int


@dataclass
class Claim:
    """The short answer a single passage gives to the question, plus a declarative form for NLI."""

    passage: Passage
    answer: str
    confidence: float  # span probability from the extractor
    has_answer: bool  # False when the extractor prefers "no answer in this passage"
    statement: str  # hypothesis text used for NLI (answer-bearing sentence)


@dataclass
class StanceGroup:
    answer: str
    members: list[int]  # indices into the claim list
    sources: list[str]
    support: float  # noisy-OR of reliability-weighted support
    mean_reliability: float


@dataclass
class Arbitration:
    decision: Decision
    conflict: bool
    margin: float  # normalized support margin between top group and its strongest contradicting rival
    groups: list[StanceGroup]  # sorted by support, descending
    rival: int | None  # index into groups of the strongest contradicting rival of groups[0]
    contradictions: list[tuple[int, int, float]]  # (group_a, group_b, conflict score)
    # one dict per conflicting group pair: which claims clash, the NLI scores, and why it counts
    conflicts: list[dict] = field(default_factory=list)

    @property
    def best(self) -> StanceGroup | None:
        return self.groups[0] if self.groups else None

    def to_dict(self) -> dict:
        return asdict(self)
