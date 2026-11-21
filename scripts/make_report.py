"""Generate the numbers and tables used by report/report.tex from the result files.

    .venv/bin/python scripts/make_report.py      # writes report/generated/*.tex
    cd report && pdflatex report && pdflatex report

Every number in the report comes from here, so re-running after new results keeps the text in sync.
Missing results are rendered as "--" so the report still compiles.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carag.eval.metrics import aggregate, score_instance  # noqa: E402
from carag.text import contains_any, same_answer  # noqa: E402

RUNS, DATA, OUT = ROOT / "runs", ROOT / "data", ROOT / "report" / "generated"
SYSTEMS = [("vanilla", "Standard RAG"), ("prompted", "Prompted RAG"),
           ("conflict_aware", r"\textbf{Conflict-aware (ours)}"), ("arbitration_only", "Arbitration only (no LLM writer)")]
SCENARIOS = [("no_conflict", "No conflict"), ("balanced_conflict", "Balanced"),
             ("misinfo_single", "Misinfo ×1"), ("misinfo_majority", "Misinfo ×2")]
macros: dict[str, str] = {}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []


def pct(x, digits=1) -> str:
    return "--" if x is None or x != x else f"{100 * x:.{digits}f}"


def m(key: str, value) -> None:
    macros[key] = str(value)


def esc(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text)


# ----------------------------------------------------------------------------------- data stats
def dataset_stats() -> None:
    for tag, d in (("test", "nq_conflict"), ("dev", "nq_conflict_dev")):
        meta = load(DATA / d / "meta.json") or {}
        m(f"{tag}-n", meta.get("n_instances", "--"))
        m(f"{tag}-corpus", f"{meta.get('n_corpus', 0):,}".replace(",", "{,}"))
        sk = meta.get("skipped", {})
        m(f"{tag}-leaks", f"{sk.get('background_passages_dropped_as_leaks', 0):,}".replace(",", "{,}"))
        sm = meta.get("substitute_methods", {})
        n = max(sum(sm.values()), 1)
        for k in ("llm", "rule", "pool"):
            m(f"{tag}-sub-{k}", pct(sm.get(k, 0) / n, 0))


# ----------------------------------------------------------------------------------- main results
def bootstrap(scored: list[dict], a: str, b: str, key: str, answerable: bool) -> tuple[float, float, float]:
    by: dict[str, dict] = {}
    for r in scored:
        if r["evidence_complete"]:
            by.setdefault(r["system"], {})[r["qid"]] = r
    if a not in by or b not in by:
        return (float("nan"),) * 3
    qids = sorted(q for q in by[a] if not answerable or by[a][q]["expected"] == "answer")
    d = [by[a][q][key] - by[b][q][key] for q in qids]
    rng = random.Random(0)
    bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
    return sum(d) / len(d), bs[50], bs[1949]


def main_tables() -> None:
    rows_main, rows_ci = [], []
    for prof, label in (("large", "Large (Qwen2.5-7B)"), ("laptop", "Laptop (Qwen2.5-3B, 4-bit)")):
        met = load(RUNS / f"test_{prof}" / "metrics.json")
        cfg = load(RUNS / f"test_{prof}" / "config.json") or {}
        m(f"{prof}-vram", cfg.get("peak_vram_gb", "--"))
        m(f"{prof}-minutes", round(cfg["elapsed_sec"] / 60) if cfg.get("elapsed_sec") else "--")
        if met:
            ret = met["retrieval"]
            m(f"{prof}-gold-recall", pct(ret["gold_recall@k"]))
            m(f"{prof}-both-recall", pct(ret["gold+conflict both retrieved (conflict scenarios)"]))
            m(f"{prof}-ncomplete", met["systems"]["conflict_aware"]["n_evidence_complete"])
        rows_main.append(rf"\midrule\multicolumn{{7}}{{l}}{{\emph{{{label}}}}}\\")
        for key, name in SYSTEMS:
            s = met["systems"].get(key) if met else None
            if not s:
                rows_main.append(f"{name} & -- & -- & -- & -- & -- & -- \\\\")
                continue
            cd = s["conflict_detection"]
            cells = [pct(s["accuracy_answerable"]), pct(s["misled_rate_answerable"]), pct(s["handling_macro"]),
                     pct(cd["precision"]) if key != "vanilla" else "--", pct(cd["recall"]), pct(cd["f1"])]
            rows_main.append(f"{name} & " + " & ".join(cells) + r" \\")
            short = {"vanilla": "van", "prompted": "pro", "conflict_aware": "ca", "arbitration_only": "arb"}[key]
            m(f"{prof}-{short}-acc", cells[0]); m(f"{prof}-{short}-misled", cells[1]); m(f"{prof}-{short}-hand", cells[2])
            m(f"{prof}-{short}-p", cells[3]); m(f"{prof}-{short}-r", cells[4]); m(f"{prof}-{short}-f1", cells[5])
            for sk, _ in SCENARIOS:
                c = s["by_scenario"].get(sk, {})
                m(f"{prof}-{short}-{sk}", pct(c.get("handled_ok")))
                m(f"{prof}-{short}-{sk}-misled", pct(c.get("misled")))
                m(f"{prof}-{short}-{sk}-flag", pct(c.get("flagged")))
        scored = jsonl(RUNS / f"test_{prof}" / "scored.jsonl")
        for base, bname in (("vanilla", "standard RAG"), ("prompted", "prompted RAG")):
            cells = []
            for key, ans in (("handled_ok", False), ("correct", True), ("misled", True)):
                mean, lo, hi = bootstrap(scored, "conflict_aware", base, key, ans)
                sig = "" if (lo != lo) or lo > 0 or hi < 0 else r"$^{\dagger}$"
                cells.append("--" if mean != mean else f"{100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]{sig}")
                if mean == mean:
                    m(f"{prof}-ci-{base}-{key}", f"{100 * mean:+.1f}")
            rows_ci.append(f"{label.split(' (')[0]} & vs.\\ {bname} & " + " & ".join(cells) + r" \\")
    write("table_main.tex", "\n".join(rows_main) + "\n")
    write("table_ci.tex", "\n".join(rows_ci) + "\n")

    rows = []
    for key, name in SYSTEMS[:3]:
        cells = []
        for sk, _ in SCENARIOS:
            cells.append(" / ".join(macros.get(f"{p}-{ {'vanilla': 'van', 'prompted': 'pro', 'conflict_aware': 'ca'}[key]}-{sk}", "--")
                                    for p in ("large", "laptop")))
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    write("table_scenarios.tex", "\n".join(rows) + "\n")


def strict_vs_lenient() -> None:
    inst = {x["qid"]: x for x in jsonl(DATA / "nq_conflict" / "instances.jsonl")}
    preds = jsonl(RUNS / "test_large" / "predictions.jsonl")
    if not preds:
        return
    for mode, strict in (("strict", True), ("lenient", False)):
        agg = aggregate([score_instance(inst[r["qid"]], r, strict=strict) for r in preds])
        for key, short in (("conflict_aware", "ca"), ("prompted", "pro"), ("vanilla", "van")):
            m(f"match-{mode}-{short}", pct(agg[key]["handling_macro"]))


# ----------------------------------------------------------------------------------- analyses
def judge() -> None:
    rows = []
    for prof in ("large", "laptop"):
        j = load(RUNS / f"test_{prof}" / "metrics_judge.json")
        if not j:
            continue
        for k, v in j["agreement"].items():
            m(f"judge-{prof}-agree-{k}", pct(v, 0))
        m(f"judge-{prof}-pro-acc", pct(j["systems"]["prompted"]["accuracy_answerable"]))
    for key, name in SYSTEMS[:3]:
        cells = []
        for prof in ("large", "laptop"):
            j = load(RUNS / f"test_{prof}" / "metrics_judge.json")
            s = j["systems"].get(key) if j else None
            cells += [pct(s["handling_macro"]) if s else "--", macros.get(f"{prof}-{ {'vanilla': 'van', 'prompted': 'pro', 'conflict_aware': 'ca'}[key]}-hand", "--")]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    write("table_judge.tex", "\n".join(rows) + "\n")


def reliability() -> None:
    names = [("prior table", "Hand-set prior (default)"), ("supervised", "Learned, supervised (dev labels)"),
             ("em", "Learned, EM (no labels)"), ("uniform (no reliability)", "None (all sources equal)")]
    rows = []
    for key, name in names:
        cells = []
        for prof in ("large", "laptop"):
            abl = load(RUNS / f"test_{prof}" / "reliability_ablation.json") or {}
            hit = next((v for k, v in abl.items() if k == key or k.endswith("_" + key)), None)
            cells += [pct(hit["handling_macro"]) if hit else "--", pct(hit["misled_rate_answerable"]) if hit else "--",
                      pct(hit["by_scenario"]["misinfo_majority"]["handled_ok"]) if hit else "--"]
            if hit:
                m(f"rel-{prof}-{key.split()[0]}", pct(hit["handling_macro"]))
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    write("table_reliability.tex", "\n".join(rows) + "\n")
    tab = load(ROOT / "reliability" / "large_supervised.json") or {}
    em = load(ROOT / "reliability" / "large_em.json") or {}
    from carag.sources import DEFAULT_RELIABILITY

    rows = [f"{esc(s)} & {DEFAULT_RELIABILITY.get(s, 0.5):.2f} & {tab.get(s, float('nan')):.2f} & {em.get(s, float('nan')):.2f} \\\\"
            for s in sorted(DEFAULT_RELIABILITY, key=lambda s: -DEFAULT_RELIABILITY[s])]
    write("table_reliability_values.tex", "\n".join(rows) + "\n")


def ablations() -> None:
    variants = [("dev_v2_base", "Without re-asking empty extractions"),
                ("dev_v2_reask", r"\textbf{Final configuration} (re-ask on)"),
                ("dev_v3_pair", "+ LLM same-subject check"),
                ("dev_v3_soft", "Softer extraction prompt everywhere"),
                ("dev_v3_hybrid", "Hybrid extractor (QA model fallback)"),
                ("dev_v3_k8", "8 passages instead of 5")]
    rows = []
    for run, name in variants:
        met = load(RUNS / run / "metrics.json")
        if not met:
            rows.append(f"{name} & -- & -- & -- & -- & -- \\\\")
            continue
        s = met["systems"]["arbitration_only"]
        cells = [pct(met["retrieval"]["gold_recall@k"]), pct(s["accuracy_answerable"]), pct(s["handling_macro"]),
                 pct(s["conflict_detection"]["recall"]), pct(s["by_scenario"]["balanced_conflict"]["handled_ok"])]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        m(f"abl-{run}-hand", cells[2]); m(f"abl-{run}-recall", cells[3]); m(f"abl-{run}-acc", cells[1])
    write("table_ablations.tex", "\n".join(rows) + "\n")


def error_analysis() -> None:
    """Where conflicts that were in the context are lost (conflict-aware, large profile)."""
    inst = {x["qid"]: x for x in jsonl(DATA / "nq_conflict" / "instances.jsonl")}
    scored = {(s["qid"], s["system"]): s for s in jsonl(RUNS / "test_large" / "scored.jsonl")}
    cause, reasons, n_conf = Counter(), Counter(), 0
    for r in jsonl(RUNS / "test_large" / "predictions.jsonl"):
        if r["system"] != "conflict_aware" or not r.get("trace"):
            continue
        s, i, t = scored[(r["qid"], r["system"])], inst[r["qid"]], r["trace"]
        for c in r["arbitration"].get("conflicts", []):
            if c["groups"][0] == 0 and c["counts"]:
                reasons[c["reason"]] += 1
                n_conf += 1
        if not (s["conflict_in_context"] and not s["flagged"]):
            continue
        cl, E = t["claims"], t["entail"]
        pos = {c["pid"]: k for k, c in enumerate(cl)}

        def status(pid, answers):
            k = pos[pid]
            if not cl[k]["has_answer"]:
                return "extractor returned no answer"
            if (E[k][k] or 0) < 0.3:
                return "claim failed NLI self-verification"
            a = cl[k]["answer"]
            if not (contains_any(a, answers) or any(same_answer(a, x) for x in answers)):
                return "extractor returned a different span"
            return "ok"

        g = status(i["gold_pids"][0], i["answers"])
        es = [status(p, [i["substitute"]]) for p in i["conflict_pids"] if p in pos]
        if g != "ok":
            cause[f"correct passage: {g}"] += 1
        elif all(e != "ok" for e in es):
            cause[f"planted passage: {es[0]}"] += 1
        else:
            cause["both claims found; relevance filter or grouping"] += 1
    total = sum(cause.values())
    m("err-total", total)
    write("table_errors.tex", "\n".join(f"{esc(k[0].upper() + k[1:])} & {v} & {pct(v / total, 0)} \\\\"
                                        for k, v in cause.most_common()) + "\n" if total else "-- & -- & -- \\\\\n")
    m("conf-total", n_conf)
    m("conf-nli", pct(reasons["NLI contradiction"] / n_conf, 0) if n_conf else "--")


def example_grid(question: str, key: str) -> None:
    """NLI grid for one benchmark question, as a LaTeX table."""
    inst = {x["qid"]: x for x in jsonl(DATA / "nq_conflict" / "instances.jsonl")}
    qid = next((q for q, x in inst.items() if x["question"] == question), None)
    row = next((r for r in jsonl(RUNS / "test_large" / "predictions.jsonl")
                if r["qid"] == qid and r["system"] == "conflict_aware"), None)
    if not row:
        write(f"grid_{key}.tex", "\\begin{tabular}{c}\\toprule results pending\\\\ \\bottomrule\\end{tabular}\n")
        return
    t, x = row["trace"], inst[qid]
    cols = [i for i, c in enumerate(t["claims"]) if c["has_answer"]]
    m(f"{key}-gold", esc(x["answers"][0])); m(f"{key}-planted", esc(x["substitute"]))
    m(f"{key}-output", esc(row["answer"])); m(f"{key}-decision", row["arbitration"]["decision"])
    head = " & ".join(rf"\makecell{{claim of [{i + 1}]\\\small {esc(t['claims'][i]['answer'][:28])}}}" for i in cols)
    lines = [rf"Passage (source) & {head} \\", r"\midrule"]
    for j, c in enumerate(t["claims"]):
        cells = []
        for i in cols:
            e, k = t["entail"][j][i], t["contra"][j][i]
            if e is None:
                cells.append("--")
                continue
            mark = rf"\checkmark\,{e:.2f}" if e >= k else rf"$\times$\,{k:.2f}"
            colour = "okc" if e >= k else "badc"
            shade = max(e, k)
            cell = rf"\cellcolor{{{colour}!{int(10 + 45 * shade) if shade >= 0.3 else 0}}}{mark}"
            if i == j:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(rf"{{[{j + 1}]}} {esc(c['source'])} & " + " & ".join(cells) + r" \\")
    spec = "l" + "c" * len(cols)
    write(f"grid_{key}.tex", f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n" + "\n".join(lines)
          + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_stats()
    main_tables()
    strict_vs_lenient()
    judge()
    reliability()
    ablations()
    error_analysis()
    example_grid("who is given credit for inventing the aqua lung", "aqua")
    lines = [r"% generated by scripts/make_report.py - do not edit",
             r"\newcommand{\R}[1]{\ifcsname r@#1\endcsname\csname r@#1\endcsname\else\textbf{??}\fi}"]
    lines += [rf"\expandafter\def\csname r@{k}\endcsname{{{v}}}" for k, v in sorted(macros.items())]
    write("numbers.tex", "\n".join(lines) + "\n")
    missing = [k for k, v in macros.items() if v == "--"]
    print(f"wrote {OUT} ({len(macros)} numbers; {len(missing)} missing)")


if __name__ == "__main__":
    main()
