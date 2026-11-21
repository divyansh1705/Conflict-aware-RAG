"""LLM-as-judge scoring, as an alternative to string matching.

The judge sees the question, the reference answer, the planted alternative answer and a system's
output, and classifies what the output *commits to*.  It then feeds the same handling rules as the
string metrics, so the two can be compared directly.
"""
from __future__ import annotations

import re

from .metrics import EXPECT_ANSWER, EXPECT_DISAGREE

JUDGE_SYSTEM = "You are a strict, impartial grader of question-answering outputs. You grade only what the output says."
JUDGE_PROMPT = """Question: {q}
Reference answer (correct): {gold}
Alternative answer (incorrect, planted in some sources): {sub}

System output to grade:
\"\"\"
{out}
\"\"\"

Which option describes what the system output commits to?
A - It gives the reference answer as its answer (it may also mention the alternative as a less reliable claim).
B - It gives the alternative answer as its answer.
C - It does not pick one: it reports that sources disagree and presents both the reference and the alternative.
D - Anything else: a different answer, "unknown", a refusal, or a disagreement that does not involve both answers.

Also decide whether the output tells the user that the sources conflict or disagree.

Reply in exactly this format:
VERDICT: <A, B, C or D>
MENTIONS_CONFLICT: <yes or no>"""

_VERDICT = re.compile(r"VERDICT\s*:\s*\**\s*([ABCD])", re.IGNORECASE)
_MENTION = re.compile(r"MENTIONS_CONFLICT\s*:\s*\**\s*(yes|no)", re.IGNORECASE)


def user_facing(output: str) -> str:
    """What a user reads: the answer and the explanation. The machine-readable CONFLICT field is dropped
    (only some systems emit it, and a 7B judge reads the label "CONFLICT: no" as mentioning a conflict)."""
    from ..prompts import parse_output

    p = parse_output(output)
    return f"Answer: {p.answer}\nExplanation: {p.response}".strip()


def judge_prompt(inst: dict, res: dict) -> tuple[str, str]:
    return JUDGE_SYSTEM, JUDGE_PROMPT.format(q=inst["question"], gold=" | ".join(inst["answers"][:4]),
                                             sub=inst["substitute"], out=user_facing(res["output"])[:1500])


def parse_judgement(text: str) -> tuple[str, bool]:
    v, m = _VERDICT.search(text), _MENTION.search(text)
    verdict = v.group(1).upper() if v else "D"
    return verdict, (m.group(1).lower() == "yes") if m else verdict == "C"


def score_judged(inst: dict, res: dict, verdict: str, mentions: bool) -> dict:
    """Same fields and handling rules as metrics.score_instance, with the judge's reading of the output."""
    pids = {r["pid"] for r in res["retrieved"]}
    gold_ret = any(p in pids for p in inst["gold_pids"])
    n_conf = sum(p in pids for p in inst["conflict_pids"])
    in_ctx = gold_ret and n_conf > 0
    flagged = mentions or verdict == "C"
    if in_ctx and inst["scenario"] == "balanced_conflict":
        expected, ok = EXPECT_DISAGREE, verdict == "C"
    elif in_ctx:
        expected, ok = EXPECT_ANSWER, verdict == "A"
    else:
        expected, ok = EXPECT_ANSWER, verdict == "A" and not flagged
    return {
        "qid": inst["qid"], "scenario": inst["scenario"], "system": res["system"],
        "gold_retrieved": gold_ret, "n_conflict_retrieved": n_conf, "conflict_in_context": in_ctx,
        "evidence_complete": gold_ret, "expected": expected,
        "correct": verdict in ("A", "C"),  # mirrors the string metric: gold answer present in the answer
        "says_sub": verdict in ("B", "C"), "misled": verdict == "B", "flagged": flagged,
        "both_mentioned": verdict == "C", "handled_ok": ok, "verdict": verdict,
    }
