"""Scoring: answer accuracy and conflict-handling quality.

Expected behaviour is defined by the evidence the system actually saw (after retrieval):

* gold passage retrieved, no edited passage retrieved -> should answer with the gold answer, no conflict flag
* balanced_conflict with both retrieved                -> should flag disagreement and mention both answers
* misinfo_* with both retrieved                        -> should answer gold, not the substituted answer
* gold passage not retrieved                           -> "evidence incomplete": reported, excluded from handling score

Per-instance fields
  correct        gold answer appears in the ANSWER line
  says_sub       substituted (wrong) answer appears in the ANSWER line
  misled         says_sub and not correct
  flagged        system reported a conflict
  both_mentioned both gold and substituted answer appear anywhere in the output
  handled_ok     the scenario-appropriate behaviour above
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..text import contains, contains_any, same_answer

EXPECT_ANSWER, EXPECT_DISAGREE = "answer", "disagree"


_DISAGREE_PREFIX = re.compile(r"^\s*sources?\s+disagree\s*:?\s*", re.IGNORECASE)
_ALTERNATIVES = re.compile(r"\s+(?:vs\.?|versus|or)\s+|\s+/\s+", re.IGNORECASE)


def answer_candidates(answer_line: str) -> list[str]:
    """The answers an ANSWER line commits to: 'Sources disagree: A vs B' -> ['A', 'B']."""
    body = _DISAGREE_PREFIX.sub("", answer_line).strip()
    return [c.strip(" .;:") for c in _ALTERNATIVES.split(body) if c.strip(" .;:")]


def matches(answer_line: str, references: list[str]) -> bool:
    """Exact containment of a reference in the line, or a candidate that is the same answer up to
    wording ('Copernicus' ~ 'Copernicus of Poland', 'Coal Township, Pennsylvania' ~ 'in Coal Township')."""
    if contains_any(answer_line, references):
        return True
    return any(same_answer(c, r) for c in answer_candidates(answer_line) for r in references)


def score_instance(inst: dict, res: dict, strict: bool = False) -> dict:
    """strict=True reproduces the original exact-containment matching (kept for comparison)."""
    pids = {r["pid"] for r in res["retrieved"]}
    gold_ret = any(p in pids for p in inst["gold_pids"])
    n_conf_ret = sum(p in pids for p in inst["conflict_pids"])
    conflict_in_context = gold_ret and n_conf_ret > 0
    complete = gold_ret

    ans, out = res["answer"], res["output"]
    if strict:
        correct = contains_any(ans, inst["answers"])
        says_sub = contains(ans, inst["substitute"])
        both = contains_any(out, inst["answers"]) and contains(out, inst["substitute"])
    else:
        correct = matches(ans, inst["answers"])
        says_sub = matches(ans, [inst["substitute"]])
        both = ((contains_any(out, inst["answers"]) or correct)
                and (contains(out, inst["substitute"]) or says_sub))
    # a planted answer that only matches the gold up to wording (e.g. "1901" vs "December 1901")
    # cannot be told apart, so it never counts as being misled
    if correct and says_sub and not answer_candidates(ans)[1:]:
        says_sub = False
    flagged = bool(res["flagged_conflict"])

    if conflict_in_context and inst["scenario"] == "balanced_conflict":
        expected = EXPECT_DISAGREE
        ok = flagged and both
    elif conflict_in_context:  # misinformation from less reliable sources
        expected = EXPECT_ANSWER
        ok = correct and not says_sub
    else:
        expected = EXPECT_ANSWER
        ok = correct and not says_sub and not flagged

    return {
        "qid": inst["qid"], "scenario": inst["scenario"], "system": res["system"],
        "gold_retrieved": gold_ret, "n_conflict_retrieved": n_conf_ret,
        "conflict_in_context": conflict_in_context, "evidence_complete": complete,
        "expected": expected, "correct": correct, "says_sub": says_sub, "misled": says_sub and not correct,
        "flagged": flagged, "both_mentioned": both, "handled_ok": ok,
    }


def _rate(rows, key):
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else float("nan")


def _prf(rows):
    tp = sum(r["flagged"] and r["conflict_in_context"] for r in rows)
    fp = sum(r["flagged"] and not r["conflict_in_context"] for r in rows)
    fn = sum((not r["flagged"]) and r["conflict_in_context"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": p, "recall": rc, "f1": f}


def aggregate(scored: list[dict]) -> dict:
    by_sys: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_sys[r["system"]].append(r)
    out = {}
    for sys_name, rows in by_sys.items():
        comp = [r for r in rows if r["evidence_complete"]]
        cells = {}
        for scen in sorted({r["scenario"] for r in rows}):
            sr = [r for r in comp if r["scenario"] == scen]
            # Scenario cells only use instances where the scenario's conflict was actually retrieved
            # (for conflict scenarios); otherwise they would mix in no-conflict behaviour.
            if scen != "no_conflict":
                sr = [r for r in sr if r["conflict_in_context"]]
            cells[scen] = {
                "n": len(sr), "accuracy": _rate(sr, "correct"), "handled_ok": _rate(sr, "handled_ok"),
                "misled": _rate(sr, "misled"), "flagged": _rate(sr, "flagged"),
                "both_mentioned": _rate(sr, "both_mentioned"),
            }
        answerable = [r for r in comp if r["expected"] == EXPECT_ANSWER]
        macro = [c["handled_ok"] for c in cells.values() if c["n"]]
        out[sys_name] = {
            "n_total": len(rows), "n_evidence_complete": len(comp),
            "accuracy_answerable": _rate(answerable, "correct"),
            "misled_rate_answerable": _rate(answerable, "misled"),
            "handling_micro": _rate(comp, "handled_ok"),
            "handling_macro": sum(macro) / len(macro) if macro else float("nan"),
            "conflict_detection": _prf(comp),
            "by_scenario": cells,
        }
    return out


def render_report(agg: dict, retrieval: dict | None = None, title: str = "Results") -> str:
    systems = list(agg)
    pct = lambda x: "-" if x != x else f"{100 * x:.1f}"
    lines = [f"# {title}", ""]
    if retrieval:
        lines += ["## Retrieval", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v if isinstance(v, int) else pct(v)} |" for k, v in retrieval.items()]
        lines.append("")
    lines += ["## Overall (evidence-complete instances)", "",
              "| system | accuracy (answerable) | misled | handling (micro) | handling (macro) | conflict P | conflict R | conflict F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        a = agg[s]
        cd = a["conflict_detection"]
        lines.append(f"| {s} | {pct(a['accuracy_answerable'])} | {pct(a['misled_rate_answerable'])} | "
                     f"{pct(a['handling_micro'])} | {pct(a['handling_macro'])} | {pct(cd['precision'])} | "
                     f"{pct(cd['recall'])} | {pct(cd['f1'])} |")
    scens = sorted({sc for s in systems for sc in agg[s]["by_scenario"]})
    for metric, label in [("handled_ok", "Conflict-handling success"), ("accuracy", "Accuracy (gold in ANSWER)"),
                          ("misled", "Misled (wrong substituted answer given)"), ("flagged", "Conflict flagged"),
                          ("both_mentioned", "Both answers mentioned")]:
        lines += ["", f"## {label} by scenario", "", "| system | " + " | ".join(scens) + " |",
                  "|---|" + "---|" * len(scens)]
        for s in systems:
            cells = agg[s]["by_scenario"]
            lines.append(f"| {s} | " + " | ".join(f"{pct(cells[sc][metric])} (n={cells[sc]['n']})" if sc in cells else "-"
                                                for sc in scens) + " |")
    return "\n".join(lines) + "\n"
