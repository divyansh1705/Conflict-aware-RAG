"""Evidence arbitration: detect contradictions between passages and decide how to answer.

Given per-passage claims and pairwise NLI probabilities (entail[j, i] = P(passage j entails claim i)):

0. Verification.  A claim is kept only if its own passage entails it (entail[i, i] >= tau_v).
   This drops answers the extractor hallucinated from parametric knowledge or from off-topic text.
   With reranker scores, claims from passages far less relevant than the most relevant
   answer-bearing passage are dropped too.
1. Stance grouping.  Verified claims are grouped when their answers match (normalized string
   match), and groups are merged when NLI says they are compatible (one entails the other, neither
   contradicts) - e.g. "1901" vs "December 10, 1901".
2. Support.  Each group's support is a noisy-OR over passages, weighted by source reliability:
   support(g) = 1 - prod_j (1 - r_j * s_jg), with s_jg = entail[j, j] for members and
   s_jg = w * max_i entail[j, i] for non-members that still back the claim.
3. Conflict.  Two groups conflict when some cross pair has P(contradiction) >= tau_c, or - with
   `distinct_is_conflict` - when they give different verified answers that NLI cannot reconcile
   (no cross entailment >= tau_e).
4. Decision.  No verified evidence -> "abstain".  No conflict with the top group -> "answer".
   Otherwise compare the top group with its strongest conflicting rival:
   margin = (s_top - s_rival) / (s_top + s_rival); margin >= delta -> "answer" (and mention the
   dissent), else "disagree" (report the disagreement explicitly).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Arbitration, Claim, StanceGroup
from .sources import SourceReliability
from .text import same_answer


@dataclass
class ArbitrationConfig:
    entail_threshold: float = 0.5
    contra_threshold: float = 0.5
    margin_threshold: float = 0.1   # defaults tuned on the dev split (`carag sweep`)
    verify_threshold: float = 0.3
    outside_support_weight: float = 0.5
    distinct_is_conflict: bool = True
    # How cross-claim entailment between two different answers is aggregated when deciding to merge:
    # "max" = either direction, "mean" = average of both directions (stricter).
    merge_rule: str = "mean"
    # Relevance gate: ignore claims from passages whose reranker score is more than this far below the
    # best retrieved passage (near-topic passages answering a *different* question). None = off.
    relevance_gap: float | None = 2.0
    # A group can only act as the conflicting rival of the top group if its best passage is within this
    # many reranker logits of the top group's best passage. None = off.
    conflict_relevance_gap: float | None = None


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def nli_requests(claims: list[Claim]) -> list[tuple[int, int, str, str]]:
    """(premise_idx j, claim_idx i, premise text, hypothesis) for every passage j vs answer-bearing claim i.

    j == i is the self-verification pair."""
    reqs = []
    for i, ci in enumerate(claims):
        if not ci.has_answer:
            continue
        for j, cj in enumerate(claims):
            reqs.append((j, i, cj.passage.text, ci.statement))
    return reqs


class EvidenceArbiter:
    def __init__(self, reliability: SourceReliability | None = None, config: ArbitrationConfig | None = None):
        self.reliability = reliability or SourceReliability()
        self.cfg = config or ArbitrationConfig()

    def arbitrate(self, claims: list[Claim], entail: np.ndarray, contra: np.ndarray,
                  relevance: list[float] | None = None) -> Arbitration:
        """entail[j, i] / contra[j, i] = P(passage j entails / contradicts claim i). NaN where not computed."""
        cfg = self.cfg
        E = np.nan_to_num(entail, nan=0.0)
        C = np.nan_to_num(contra, nan=0.0)
        verified = np.array([E[i, i] for i in range(len(claims))])
        idx = [i for i, c in enumerate(claims) if c.has_answer and verified[i] >= cfg.verify_threshold]
        if relevance is not None and cfg.relevance_gap is not None and idx:
            # measured from the most relevant passage that states a verified answer, so a top-ranked
            # passage without an answer cannot push every answer-bearing passage below the bar
            floor = max(relevance[i] for i in idx) - cfg.relevance_gap
            idx = [i for i in idx if relevance[i] >= floor]
        if not idx:
            return Arbitration("abstain", False, 0.0, [], None, [])
        rel = np.array([self.reliability(c.passage.source) for c in claims])

        # 1a. group identical answers
        uf = _UnionFind(idx)
        for a in idx:
            for b in idx:
                if a < b and same_answer(claims[a].answer, claims[b].answer):
                    uf.union(a, b)

        def groups_now() -> list[list[int]]:
            g: dict[int, list[int]] = {}
            for i in idx:
                g.setdefault(uf.find(i), []).append(i)
            return list(g.values())

        def cross(ga: list[int], gb: list[int], M: np.ndarray) -> float:
            return max(max(M[a, b], M[b, a]) for a in ga for b in gb)

        def compat(ga: list[int], gb: list[int]) -> float:
            if cfg.merge_rule == "mean":
                return max((E[a, b] + E[b, a]) / 2 for a in ga for b in gb)
            return cross(ga, gb, E)

        def conflict_score(ga: list[int], gb: list[int]) -> float:
            c = cross(ga, gb, C)
            if cfg.distinct_is_conflict and compat(ga, gb) < cfg.entail_threshold:
                # distinct verified answers that NLI cannot reconcile
                c = max(c, cfg.contra_threshold)
            return c

        # 1b. merge NLI-compatible groups (entailment without contradiction)
        changed = True
        while changed:
            changed = False
            gs = groups_now()
            for x in range(len(gs)):
                for y in range(x + 1, len(gs)):
                    if cross(gs[x], gs[y], C) < cfg.contra_threshold and compat(gs[x], gs[y]) >= cfg.entail_threshold:
                        uf.union(gs[x][0], gs[y][0])
                        changed = True
                        break
                if changed:
                    break

        # 2. reliability-weighted noisy-OR support
        raw_groups = groups_now()
        built: list[tuple[float, list[int]]] = []
        for g in raw_groups:
            miss = 1.0
            for j in range(len(claims)):
                s = verified[j] if j in g else cfg.outside_support_weight * max(E[j, i] for i in g)
                miss *= 1.0 - rel[j] * s
            built.append((1.0 - miss, g))
        built.sort(key=lambda t: (-t[0], min(t[1])))

        groups = []
        for support, g in built:
            # canonical answer: from the most reliable member (ties -> best verified)
            rep = max(g, key=lambda i: (rel[i], verified[i]))
            groups.append(StanceGroup(
                answer=claims[rep].answer,
                members=sorted(g),
                sources=[claims[i].passage.source for i in sorted(g)],
                support=float(support),
                mean_reliability=float(np.mean([rel[i] for i in g])),
            ))

        # 3. contradictions between groups
        def best_rel(g: list[int]) -> float | None:
            return None if relevance is None else max(relevance[i] for i in g)

        contradictions, conflicts = [], []
        for x in range(len(built)):
            for y in range(x + 1, len(built)):
                gx, gy = built[x][1], built[y][1]
                p = conflict_score(gx, gy)
                if p < cfg.contra_threshold:
                    continue
                a, b = max(((a, b) for a in gx for b in gy), key=lambda ab: max(C[ab[0], ab[1]], C[ab[1], ab[0]]))
                nli = float(max(C[a, b], C[b, a]))
                eligible = True
                if x == 0 and cfg.conflict_relevance_gap is not None and relevance is not None:
                    eligible = best_rel(gy) >= best_rel(gx) - cfg.conflict_relevance_gap
                contradictions.append((x, y, float(p)))
                conflicts.append({
                    "groups": [x, y], "claims": [a, b], "score": float(p),
                    "nli_contradiction": nli, "compatibility": float(compat(gx, gy)),
                    "reason": "NLI contradiction" if nli >= cfg.contra_threshold else "distinct answers NLI cannot reconcile",
                    "counts": eligible,
                    "note": None if eligible else "rival passage much less relevant than the top answer; not treated as a conflict",
                })

        # 4. decision
        rivals = [(c["groups"][1], c["score"]) for c in conflicts if c["groups"][0] == 0 and c["counts"]]
        if not rivals:
            return Arbitration("answer", False, 1.0, groups, None, contradictions, conflicts)
        rival = max(rivals, key=lambda t: groups[t[0]].support)[0]
        s1, s2 = groups[0].support, groups[rival].support
        margin = (s1 - s2) / (s1 + s2) if s1 + s2 > 0 else 0.0
        decision = "answer" if margin >= cfg.margin_threshold else "disagree"
        return Arbitration(decision, True, float(margin), groups, rival, contradictions, conflicts)
