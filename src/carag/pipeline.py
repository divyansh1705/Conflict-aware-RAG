"""End-to-end RAG systems.

Systems compared (all share the same retrieval results for a controlled comparison):
  vanilla           standard RAG: top-k passages -> LLM
  prompted          standard RAG + source labels + instructions to handle conflicts (prompting-only baseline)
  conflict_aware    claims -> NLI -> stance groups/reliability arbitration -> LLM conditioned on the verdict
  arbitration_only  same arbitration, deterministic templated answer (no LLM; isolates the arbitration module)
"""
from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .arbitration import EvidenceArbiter, nli_requests
from .claims import ExtractiveClaimExtractor
from .generator import Generator
from .nli import NLIModel
from .prompts import arbitrated_prompt, parse_output, prompted_prompt, render_arbitration, vanilla_prompt
from .retriever import RerankRetriever, Retriever
from .schema import Arbitration, Claim, Retrieved

PAIR_SYSTEM = "You compare two documents. You judge only what the documents are about, never whether they are true."
PAIR_PROMPT = """Question: {q}

Document A:
{a_text}

Document B:
{b_text}

Document A gives the answer "{a_ans}"; document B gives "{b_ans}".
Are both documents about the subject the question asks about, so that these are two competing answers to the same question?
Reply SAME if both are about that subject (even if one of them is wrong).
Reply A if only document A is about it and document B is about something else (for example a different film or song with a similar title, a different person or event, or a narrower or broader category such as "personal computer" instead of "computer").
Reply B if only document B is about it.
Reply with one word: SAME, A or B."""

ALL_SYSTEMS = ("vanilla", "prompted", "conflict_aware", "arbitration_only")
LLM_SYSTEMS = {"vanilla", "prompted", "conflict_aware"}


@dataclass
class RAGResult:
    qid: str
    question: str
    system: str
    output: str
    answer: str
    flagged_conflict: bool
    retrieved: list[dict]
    arbitration: dict | None = None
    trace: dict | None = None  # claims + NLI matrices

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Analysis:
    retrieved: list[Retrieved]
    claims: list[Claim]
    arbitration: Arbitration
    entail: np.ndarray | None = None
    contra: np.ndarray | None = None
    off_subject: list[int] = field(default_factory=list)  # claims dropped by the same-question check
    pair_verdict: str | None = None  # SAME / A / B from the same-question check

    def trace(self) -> dict:
        """Serializable claims + NLI matrices (enough to re-run arbitration offline, see `carag sweep`)."""
        r = lambda m: [[None if x != x else round(float(x), 4) for x in row] for row in m]
        return {"claims": [{"pid": c.passage.pid, "source": c.passage.source, "answer": c.answer,
                            "has_answer": c.has_answer, "confidence": round(c.confidence, 4),
                            "relevance": round(float(r_.score), 4), "off_subject": k in self.off_subject}
                           for k, (c, r_) in enumerate(zip(self.claims, self.retrieved))],
                "entail": r(self.entail), "contra": r(self.contra), "pair_verdict": self.pair_verdict}


class ConflictAwareRAG:
    def __init__(self, retriever: Retriever, extractor: ExtractiveClaimExtractor, nli: NLIModel,
                 arbiter: EvidenceArbiter, generator: Generator | None = None, k: int = 5,
                 pair_check: bool = False, reask_none: bool = False):
        self.retriever, self.extractor, self.nli, self.arbiter = retriever, extractor, nli, arbiter
        self.generator, self.k = generator, k
        # On a detected conflict, ask the LLM whether the two clashing passages are about the same subject
        # (NLI accepts "father of the personal computer" as "father of the computer"). It compares the two
        # documents with each other and is never asked which answer is true.
        self.pair_check = pair_check and generator is not None
        # Re-ask the extractor (softer prompt) for highly relevant passages where it found no answer.
        self.reask_none = reask_none and generator is not None

    @property
    def uses_reranker(self) -> bool:
        return isinstance(self.retriever, RerankRetriever)

    # -- evidence analysis (batched across questions) --------------------------------------------
    def analyze(self, questions: list[str], retrieved: list[list[Retrieved]] | None = None) -> list[Analysis]:
        retrieved = retrieved if retrieved is not None else self.retriever.search_batch(questions, self.k)
        flat = [(q, r.passage) for q, rs in zip(questions, retrieved) for r in rs]
        flat_claims = self.extractor.extract(flat)
        if self.reask_none:
            flat_claims = self._reask(flat, flat_claims, retrieved)
        per_q: list[list[Claim]] = []
        pos = 0
        for rs in retrieved:
            per_q.append(flat_claims[pos : pos + len(rs)])
            pos += len(rs)

        reqs = [(qi, j, i, prem, hyp) for qi, cl in enumerate(per_q) for (j, i, prem, hyp) in nli_requests(cl)]
        probs = self.nli.predict([(prem, hyp) for (_, _, _, prem, hyp) in reqs])
        mats = [(np.full((len(c), len(c)), np.nan), np.full((len(c), len(c)), np.nan)) for c in per_q]
        for (qi, j, i, _, _), p in zip(reqs, probs):
            mats[qi][0][j, i] = p[0]
            mats[qi][1][j, i] = p[2]
        rerank = self.uses_reranker  # only cross-encoder scores are comparable
        rel = [[r.score for r in rs] if rerank else None for rs in retrieved]
        analyses = [Analysis(rs, cl, self.arbiter.arbitrate(cl, E, C, rl), E, C)
                    for rs, cl, (E, C), rl in zip(retrieved, per_q, mats, rel)]
        if self.pair_check:
            self._apply_pair_check(questions, analyses, rel)
        return analyses

    def _reask(self, flat, claims: list[Claim], retrieved) -> list[Claim]:
        """Second pass with the softer prompt for NONE claims from passages within 1 logit of the best one."""
        from .claims import LLMClaimExtractor

        if not isinstance(self.extractor, LLMClaimExtractor):
            return claims
        rerank, todo, pos = self.uses_reranker, [], 0
        for rs in retrieved:
            best = max((r.score for r in rs), default=0.0)
            for k, r in enumerate(rs):
                if not claims[pos + k].has_answer and (not rerank or r.score >= best - 1.0):
                    todo.append(pos + k)
            pos += len(rs)
        if todo:
            soft = LLMClaimExtractor(self.extractor.generator, "soft")
            for i, c in zip(todo, soft.extract([flat[i] for i in todo])):
                if c.has_answer:
                    claims[i] = c
        return claims

    def _apply_pair_check(self, questions: list[str], analyses: list[Analysis], rel) -> None:
        """For each counted conflict, ask whether the two clashing passages are about the same subject.
        If one is about something else, drop that side's claims and re-arbitrate (once)."""
        reqs = []
        for qi, a in enumerate(analyses):
            arb = a.arbitration
            if not arb.conflict:
                continue
            ca, cb = next(c["claims"] for c in arb.conflicts if c["groups"] == [0, arb.rival] and c["counts"])
            reqs.append((qi, ca, cb))
        if not reqs:
            return
        outs = self.generator.generate([
            (PAIR_SYSTEM, PAIR_PROMPT.format(
                q=questions[qi], a_text=analyses[qi].claims[ca].passage.text, b_text=analyses[qi].claims[cb].passage.text,
                a_ans=analyses[qi].claims[ca].answer, b_ans=analyses[qi].claims[cb].answer)) for qi, ca, cb in reqs])
        for (qi, ca, cb), o in zip(reqs, outs):
            word = (re.findall(r"[A-Za-z]+", o) or ["SAME"])[0].upper()
            a = analyses[qi]
            a.pair_verdict = word
            if word not in ("A", "B"):
                continue
            off_group = a.arbitration.rival if word == "A" else 0  # ca is in group 0, cb in the rival group
            drop = set(a.arbitration.groups[off_group].members)
            a.claims = [replace(c, has_answer=False) if i in drop else c for i, c in enumerate(a.claims)]
            a.off_subject = sorted(drop)
            a.arbitration = self.arbiter.arbitrate(a.claims, a.entail, a.contra, rel[qi])

    # -- run any subset of systems on the same retrieval -----------------------------------------
    def run(self, items: list[tuple[str, str]], systems: tuple[str, ...] = ALL_SYSTEMS,
            cache_dir: str | Path | None = None, cache_tag: str = "") -> dict[str, list[RAGResult]]:
        """Run systems on shared retrieval. With `cache_dir`, every stage (retrieval+analysis, each system's
        generations) is checkpointed there and re-used on rerun, so an interrupted eval resumes."""
        qids = [q for q, _ in items]
        questions = [x for _, x in items]
        needs_analysis = {"conflict_aware", "arbitration_only"} & set(systems)
        if LLM_SYSTEMS & set(systems) and self.generator is None:
            raise ValueError("an LLM generator is required for systems " + ", ".join(sorted(LLM_SYSTEMS & set(systems))))

        cache = Path(cache_dir) if cache_dir else None
        key = hashlib.sha1("\n".join([cache_tag, *qids]).encode()).hexdigest()[:12]  # tag = settings fingerprint
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        stage = (lambda name: cache / f"{name}.{key}.pkl") if cache else None

        def cached(name, fn):
            if stage is None:
                return fn()
            path = stage(name)
            if path.exists():
                print(f"[carag] reusing cached stage '{name}'", flush=True)
                return pickle.loads(path.read_bytes())
            val = fn()
            path.with_suffix(".tmp").write_bytes(pickle.dumps(val))
            path.with_suffix(".tmp").replace(path)
            return val

        retrieved = cached("retrieval", lambda: self.retriever.search_batch(questions, self.k))
        analyses = (cached("analysis", lambda: self.analyze(questions, retrieved)) if needs_analysis
                    else [None] * len(items))

        results: dict[str, list[RAGResult]] = {}
        for system in systems:
            outputs = cached(f"outputs-{system}", lambda: self._outputs(system, questions, retrieved, analyses))
            res = []
            for qid, q, rs, a, out in zip(qids, questions, retrieved, analyses, outputs):
                parsed = parse_output(out)
                res.append(RAGResult(
                    qid=qid, question=q, system=system, output=out, answer=parsed.answer,
                    flagged_conflict=parsed.flagged_conflict,
                    retrieved=[{"pid": r.passage.pid, "source": r.passage.source, "score": r.score} for r in rs],
                    arbitration=a.arbitration.to_dict() if a is not None and system in needs_analysis else None,
                    trace=a.trace() if a is not None and system in needs_analysis else None,
                ))
            results[system] = res
        return results

    def _outputs(self, system: str, questions: list[str], retrieved, analyses) -> list[str]:
        if system == "arbitration_only":
            return [render_arbitration(a.arbitration, a.claims) for a in analyses]
        if system == "vanilla":
            prompts = [vanilla_prompt(q, rs) for q, rs in zip(questions, retrieved)]
        elif system == "prompted":
            prompts = [prompted_prompt(q, rs) for q, rs in zip(questions, retrieved)]
        elif system == "conflict_aware":
            prompts = [arbitrated_prompt(q, a.retrieved, a.claims, a.arbitration, self.arbiter.reliability)
                       for q, a in zip(questions, analyses)]
        else:
            raise ValueError(f"unknown system {system!r}")
        return self.generator.generate(prompts)

    def ask(self, question: str, template: bool = False) -> tuple[str, Analysis]:
        """Single-question convenience API (LLM answer if a generator is configured, else templated)."""
        a = self.analyze([question])[0]
        if self.generator is None or template:
            return render_arbitration(a.arbitration, a.claims), a
        prompt = arbitrated_prompt(question, a.retrieved, a.claims, a.arbitration, self.arbiter.reliability)
        return self.generator.generate([prompt])[0], a
