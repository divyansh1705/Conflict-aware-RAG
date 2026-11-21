"""Learning source reliability instead of hand-setting it.

Two estimators over the verified, stance-grouped claims of a calibration run (e.g. the dev split):

* supervised - reliability(s) = smoothed fraction of source s's claims that agree with the known
  answer.  Needs a labeled calibration set (questions with gold answers).
* em         - unsupervised truth discovery.  Alternates between (E) the posterior that each stance
  group of a question is the true one given current reliabilities and (M) reliability(s) = expected
  fraction of s's claims that are true.  Needs no labels, but assumes sources err independently, so
  repeated misinformation can fool it.
"""
from __future__ import annotations

from collections import defaultdict

from .arbitration import ArbitrationConfig, EvidenceArbiter
from .sources import SourceReliability
from .text import contains_any, same_answer


def _grouped(cached, config: ArbitrationConfig | None):
    """Per question: [(sources of the group's members, group answer)] for verified stance groups."""
    arbiter = EvidenceArbiter(SourceReliability({}), config)  # uniform reliability: grouping does not depend on it
    for r, claims, E, C, rel in cached:
        arb = arbiter.arbitrate(claims, E, C, rel)
        yield r["qid"], [([claims[m].passage.source for m in g.members], g.answer) for g in arb.groups]


def learn_supervised(cached, by_qid, config: ArbitrationConfig | None = None, prior: float = 1.0):
    right, total = defaultdict(float), defaultdict(float)
    for qid, groups in _grouped(cached, config):
        golds = by_qid[qid]["answers"]
        for sources, answer in groups:
            ok = contains_any(answer, golds) or any(same_answer(answer, g) for g in golds)
            for s in sources:
                total[s] += 1
                right[s] += ok
    table = {s: round((right[s] + prior) / (total[s] + 2 * prior), 3) for s in sorted(total)}
    return table, {s: int(total[s]) for s in sorted(total)}


def learn_em(cached, config: ArbitrationConfig | None = None, iters: int = 50, prior: float = 1.0, init: float = 0.7):
    questions = [g for _, g in _grouped(cached, config) if g]
    sources = sorted({s for groups in questions for srcs, _ in groups for s in srcs})
    rel = {s: init for s in sources}
    for _ in range(iters):
        num, den = defaultdict(float), defaultdict(float)
        for groups in questions:
            # P(group g is the true stance) is proportional to prod_{s in g} r_s * prod_{s elsewhere} (1 - r_s)
            likes = []
            for gi, _ in enumerate(groups):
                p = 1.0
                for gj, (srcs, _) in enumerate(groups):
                    for s in srcs:
                        p *= rel[s] if gi == gj else 1.0 - rel[s]
                likes.append(p)
            z = sum(likes) or 1.0
            for (srcs, _), like in zip(groups, likes):
                for s in srcs:
                    num[s] += like / z
                    den[s] += 1
        new = {s: (num[s] + prior) / (den[s] + 2 * prior) for s in sources}
        delta = max(abs(new[s] - rel[s]) for s in sources) if sources else 0.0
        rel = new
        if delta < 1e-5:
            break
    counts = defaultdict(int)
    for groups in questions:
        for srcs, _ in groups:
            for s in srcs:
                counts[s] += 1
    return {s: round(rel[s], 3) for s in sources}, dict(counts)
