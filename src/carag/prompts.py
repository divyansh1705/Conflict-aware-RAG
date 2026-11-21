"""Prompt builders for the three generation modes and a parser for their shared output format."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Arbitration, Claim, Retrieved
from .sources import SourceReliability

FORMAT_WITH_CONFLICT = """Reply in exactly this format:
ANSWER: <short answer, a few words; or "Sources disagree: <answer 1> vs <answer 2>" if the conflict is unresolved>
CONFLICT: <yes if the documents give contradictory answers, otherwise no>
RESPONSE: <one to three sentences for the user, citing documents like [1]>"""


def _docs(retrieved: list[Retrieved], with_sources: bool) -> str:
    lines = []
    for n, r in enumerate(retrieved, 1):
        tag = f"[{n}] (source: {r.passage.source}) " if with_sources else f"[{n}] "
        lines.append(tag + r.passage.text)
    return "\n\n".join(lines)


# --- standard RAG -----------------------------------------------------------------------------

VANILLA_SYSTEM = "You are a question answering assistant. Answer the question using the provided documents."


def vanilla_prompt(question: str, retrieved: list[Retrieved]) -> tuple[str, str]:
    user = f"""Documents:
{_docs(retrieved, with_sources=False)}

Question: {question}

Reply in exactly this format:
ANSWER: <short answer, a few words>
RESPONSE: <one or two sentences answering the question>"""
    return VANILLA_SYSTEM, user


# --- conflict-aware *prompting* baseline (no arbitration module) -----------------------------

PROMPTED_SYSTEM = (
    "You are a careful question answering assistant. Answer using the provided documents. The documents "
    "come from sources of differing reliability, and some may contain errors or contradict each other."
)


def prompted_prompt(question: str, retrieved: list[Retrieved]) -> tuple[str, str]:
    user = f"""Documents:
{_docs(retrieved, with_sources=True)}

Question: {question}

Instructions:
- If the documents agree, answer the question.
- If documents contradict each other, weigh the reliability of their sources (encyclopedias and major news outlets are more reliable than blogs, forums, social media and content farms) and how many sources support each claim.
- If one claim is clearly better supported, give it as the answer and mention the conflicting claim.
- If the conflict cannot be resolved, do not pick a side: say that the sources disagree and list each competing answer with its source.

{FORMAT_WITH_CONFLICT}"""
    return PROMPTED_SYSTEM, user


# --- conflict-aware RAG with evidence arbitration ---------------------------------------------

ARBITRATED_SYSTEM = (
    "You are a careful question answering assistant. You receive retrieved documents together with an evidence "
    "analysis produced by a claim-extraction and natural-language-inference module, including a verdict on how to "
    "answer. Follow the verdict."
)

_LETTERS = "ABCDEFGH"


def _evidence_brief(arb: Arbitration, claims: list[Claim], rel: SourceReliability) -> str:
    shown = [0] + ([arb.rival] if arb.conflict else [])  # the verdict concerns only these groups
    lines = []
    for letter, gi in zip(_LETTERS, shown):
        g = arb.groups[gi]
        who = ", ".join(f"[{m + 1}] {claims[m].passage.source} (reliability {rel(claims[m].passage.source):.2f})"
                        for m in g.members)
        lines.append(f'Claim {letter}: "{g.answer}" - stated by {who}; support score {g.support:.2f}')
    if arb.conflict:
        p = max(p for x, y, p in arb.contradictions if {x, y} == {0, arb.rival})
        lines.append(f"Claims A and B are incompatible answers to the question (conflict score {p:.2f}).")
    return "\n".join(lines)


def _verdict(arb: Arbitration, claims: list[Claim]) -> tuple[str, str]:
    """(verdict instruction, output format) - the format is fixed per verdict so the LLM cannot drift."""
    best = arb.groups[0]
    if not arb.conflict:
        return (f'Verdict: the documents do not contradict each other on this question. Answer with claim A '
                f'("{best.answer}"). Do not report a disagreement.',
                "Reply in exactly this format:\nANSWER: <short answer>\nCONFLICT: no\n"
                "RESPONSE: <one or two sentences for the user, citing documents like [1]>")
    rival = arb.groups[arb.rival]
    rival_docs = ", ".join(f"[{m + 1}] ({claims[m].passage.source})" for m in rival.members)
    if arb.decision == "answer":
        return (f'Verdict: the evidence conflicts, but claim A ("{best.answer}") is clearly better supported '
                f"(margin {arb.margin:.2f}). Give claim A as the answer. In RESPONSE, briefly note that "
                f'{rival_docs} claims "{rival.answer}" and that it is less well supported.',
                "Reply in exactly this format:\nANSWER: <claim A as a short answer>\nCONFLICT: yes\n"
                "RESPONSE: <two or three sentences for the user, citing documents like [1]>")
    return (f"Verdict: the evidence genuinely conflicts and neither claim is clearly better supported "
            f"(margin {arb.margin:.2f}). Do NOT pick a side; explain which sources say what.",
            f"Reply in exactly this format:\nANSWER: Sources disagree: {best.answer} vs {rival.answer}\nCONFLICT: yes\n"
            "RESPONSE: <two or three sentences for the user, citing documents like [1]>")


NO_EVIDENCE_NOTE = ("Evidence analysis: no document could be verified to state an answer to this question, so no "
                    "contradiction between documents was found.")


def arbitrated_prompt(question: str, retrieved: list[Retrieved], claims: list[Claim], arb: Arbitration,
                      rel: SourceReliability) -> tuple[str, str]:
    if arb.decision == "abstain":
        analysis = NO_EVIDENCE_NOTE
        fmt = ("Answer from the documents if they contain the answer; otherwise answer \"unknown\".\n"
               "Reply in exactly this format:\nANSWER: <short answer or unknown>\nCONFLICT: no\n"
               "RESPONSE: <one or two sentences for the user>")
    else:
        verdict, fmt = _verdict(arb, claims)
        analysis = f"Evidence analysis:\n{_evidence_brief(arb, claims, rel)}\n{verdict}"
    user = f"""Documents:
{_docs(retrieved, with_sources=True)}

Question: {question}

{analysis}

{fmt}"""
    return ARBITRATED_SYSTEM, user


def render_arbitration(arb: Arbitration, claims: list[Claim]) -> str:
    """Deterministic response straight from the arbitration result (no LLM) - the 'arbitration-only' system."""
    if arb.decision == "abstain":
        return ("ANSWER: unknown\nCONFLICT: no\n"
                "RESPONSE: None of the retrieved documents contains an answer to the question.")
    best = arb.groups[0]
    cite = ", ".join(f"[{m + 1}] {claims[m].passage.source}" for m in best.members)
    if not arb.conflict:
        return f"ANSWER: {best.answer}\nCONFLICT: no\nRESPONSE: {best.answer} (according to {cite})."
    rival = arb.groups[arb.rival]
    rcite = ", ".join(f"[{m + 1}] {claims[m].passage.source}" for m in rival.members)
    if arb.decision == "answer":
        return (f"ANSWER: {best.answer}\nCONFLICT: yes\nRESPONSE: The best-supported answer is {best.answer} ({cite}). "
                f"Note that {rcite} claims {rival.answer}, but that claim is less well supported.")
    return (f"ANSWER: Sources disagree: {best.answer} vs {rival.answer}\nCONFLICT: yes\n"
            f"RESPONSE: The retrieved sources conflict: {cite} says {best.answer}, while {rcite} says {rival.answer}.")


# --- output parsing ---------------------------------------------------------------------------

_FIELD = re.compile(r"^\s*\**(ANSWER|CONFLICT|RESPONSE)\**\s*:\s*(.*)$", re.IGNORECASE)
_DISAGREE = re.compile(
    r"\b(conflict(s|ing)?|disagree(s|ment)?|contradict(s|ory|ion|ing)?|inconsisten(t|cy)|discrepanc(y|ies)|"
    r"differing (accounts|answers|information|claims)|sources (differ|vary))\b",
    re.IGNORECASE,
)
_NEGATED = re.compile(r"\b(no|not|without|don't|do not|does not)\b[^.]{0,30}\b(conflict|disagree|contradict)", re.IGNORECASE)


@dataclass
class ParsedOutput:
    answer: str
    response: str
    conflict_field: bool | None
    flagged_conflict: bool


def parse_output(text: str) -> ParsedOutput:
    fields: dict[str, list[str]] = {}
    current = None
    for line in text.strip().splitlines():
        m = _FIELD.match(line)
        if m:
            current = m.group(1).upper()
            fields[current] = [m.group(2).strip()]
        elif current:
            fields[current].append(line.strip())
    get = lambda k: " ".join(x for x in fields.get(k, []) if x).strip()
    answer = get("ANSWER") or text.strip().splitlines()[0] if text.strip() else ""
    response = get("RESPONSE") or text.strip()
    cf_raw = get("CONFLICT").lower()
    conflict_field = True if cf_raw.startswith("yes") else False if cf_raw.startswith("no") else None
    body = f"{answer} {response}"
    keyword = bool(_DISAGREE.search(body)) and not _NEGATED.search(body)
    flagged = conflict_field if conflict_field is not None else keyword
    if re.search(r"sources disagree", answer, re.IGNORECASE):
        flagged = True
    return ParsedOutput(answer, response, conflict_field, flagged)
