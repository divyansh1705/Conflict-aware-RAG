"""Claim extraction: what does each retrieved passage say the answer is?

Two extractors:
  * LLMClaimExtractor - an instruction-tuned LLM reads one passage at a time and gives a short
    answer or NONE (default; much better at declining off-topic passages).
  * ExtractiveClaimExtractor - SQuAD2-style extractive QA (fast, no LLM needed).

Each claim's NLI hypothesis is the question-conditioned statement
    The answer to the question "<q>?" is <answer>.
which lets NLI check (a) that the claim's own passage really asserts it (self-verification) and
(b) whether other passages entail or contradict it.
"""
from __future__ import annotations

import math
import re

from .devices import pick_device, pick_dtype
from .schema import Claim, Passage
from .text import normalize_answer, sentence_containing


def claim_statement(question: str, answer: str) -> str:
    q = question.strip().rstrip("?").strip()
    return f'The answer to the question "{q}?" is {answer}.'


CLAIM_SYSTEM = "You read one document and extract the answer it gives to a question. You never use outside knowledge."
_CLAIM_HEAD = "Document:\n{text}\n\nQuestion: {q}\n\n"
CLAIM_PROMPTS = {
    # strict: prefers NONE when unsure (few false claims, but misses answers)
    "strict": _CLAIM_HEAD + "Using ONLY this document, answer the question in as few words as possible (a name, date, "
              "number or short phrase). If this document does not state the answer to this exact question, reply NONE.\nAnswer:",
    # soft: asks for the document's answer first, NONE only for off-topic documents
    "soft": _CLAIM_HEAD + "What answer does this document give to the question? Reply with a few words copied from the "
            "document (a name, date, number or short phrase), even if you believe the document is wrong. Reply NONE only "
            "if the document is about something else and gives no answer.\nAnswer:",
}


class LLMClaimExtractor:
    def __init__(self, generator, prompt: str = "strict"):
        self.generator, self.prompt = generator, CLAIM_PROMPTS[prompt]

    @staticmethod
    def _clean(text: str) -> str:
        s = text.strip().splitlines()[0] if text.strip() else ""
        s = re.sub(r"^(answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip().strip("\"'`*").rstrip(".").strip()
        return "" if normalize_answer(s) in {"", "none", "unknown", "n a", "not stated"} else s

    def extract(self, pairs: list[tuple[str, Passage]]) -> list[Claim]:
        outs = self.generator.generate([(CLAIM_SYSTEM, self.prompt.format(text=p.text, q=q.strip())) for q, p in pairs])
        claims = []
        for (q, p), o in zip(pairs, outs):
            a = self._clean(o)
            claims.append(Claim(p, a, 1.0 if a else 0.0, bool(a), claim_statement(q, a) if a else ""))
        return claims


class HybridClaimExtractor:
    """LLM extractor first; where it returns NONE, fall back to the extractive QA model's answer (if it has one).
    NLI self-verification downstream filters the extra candidates."""

    def __init__(self, llm: "LLMClaimExtractor", extractive: "ExtractiveClaimExtractor"):
        self.llm, self.extractive = llm, extractive

    def extract(self, pairs: list[tuple[str, Passage]]) -> list[Claim]:
        claims = self.llm.extract(pairs)
        missing = [i for i, c in enumerate(claims) if not c.has_answer]
        for i, c in zip(missing, self.extractive.extract([pairs[i] for i in missing])):
            if c.has_answer:
                claims[i] = c
        return claims


class ExtractiveClaimExtractor:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2", device: str | None = None,
                 batch_size: int = 32, max_length: int = 384, stride: int = 128,
                 max_answer_tokens: int = 30, null_margin: float = 0.0, statement: str = "qa"):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = pick_dtype(self.device, "encoder")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.batch_size, self.max_length, self.stride = batch_size, max_length, stride
        self.max_answer_tokens = max_answer_tokens
        # has_answer iff best_span_score - null_score > -null_margin. Larger margin => more permissive.
        self.null_margin = null_margin
        self.statement = statement  # "qa" (question-conditioned template) or "sentence" (answer-bearing sentence)

    def extract(self, pairs: list[tuple[str, Passage]]) -> list[Claim]:
        claims: list[Claim] = []
        for s in range(0, len(pairs), self.batch_size):
            claims.extend(self._extract_batch(pairs[s : s + self.batch_size]))
        return claims

    def _extract_batch(self, pairs: list[tuple[str, Passage]]) -> list[Claim]:
        torch = self.torch
        questions = [q.strip() for q, _ in pairs]
        contexts = [p.text for _, p in pairs]
        enc = self.tok(questions, contexts, truncation="only_second", max_length=self.max_length,
                       stride=self.stride, return_overflowing_tokens=True, return_offsets_mapping=True,
                       padding=True, return_tensors="pt")
        sample_map = enc.pop("overflow_to_sample_mapping").tolist()
        offsets = enc.pop("offset_mapping").tolist()
        model_inputs = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        with torch.inference_mode():
            out = self.model(**model_inputs)
        start = out.start_logits.float().cpu()
        end = out.end_logits.float().cpu()
        start_lp = start.log_softmax(-1)
        end_lp = end.log_softmax(-1)

        # best[sample] = (span_score, null_score, char_s, char_e, logprob)
        best: dict[int, tuple] = {}
        for w, si in enumerate(sample_map):
            seq_ids = enc.sequence_ids(w)
            ctx = [i for i, sid in enumerate(seq_ids) if sid == 1]
            if not ctx:
                continue
            null = float(start[w, 0] + end[w, 0])
            c0, c1 = ctx[0], ctx[-1]
            s_top = torch.topk(start[w, c0 : c1 + 1], min(20, c1 - c0 + 1)).indices + c0
            e_top = torch.topk(end[w, c0 : c1 + 1], min(20, c1 - c0 + 1)).indices + c0
            for a in s_top.tolist():
                for b in e_top.tolist():
                    if b < a or b - a + 1 > self.max_answer_tokens:
                        continue
                    score = float(start[w, a] + end[w, b])
                    if si not in best or score > best[si][0]:
                        lp = float(start_lp[w, a] + end_lp[w, b])
                        best[si] = (score, null, offsets[w][a][0], offsets[w][b][1], lp)
            if si in best:  # keep the *best* null score seen for the sample (most confident "no answer")
                sc, nl, cs, ce, lp = best[si]
                best[si] = (sc, max(nl, null), cs, ce, lp)

        claims = []
        for i, (q, p) in enumerate(pairs):
            if i not in best:
                claims.append(Claim(p, "", 0.0, False, ""))
                continue
            score, null, cs, ce, lp = best[i]
            answer = p.text[cs:ce].strip()
            has = bool(answer) and (score - null) > -self.null_margin
            if not answer:
                stmt = ""
            elif self.statement == "sentence":
                stmt = sentence_containing(p.text, cs, ce)
            else:
                stmt = claim_statement(q, answer)
            claims.append(Claim(p, answer, math.exp(lp), has, stmt))
        return claims
