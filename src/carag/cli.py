"""Command line interface.

  carag build-data --out data/nq_conflict                       # benchmark + 60k-passage corpus
  carag serve --data data/nq_conflict                           # web UI at http://localhost:7860
  carag chat  --data data/nq_conflict                           # interactive questions in the terminal
  carag ask   --data data/nq_conflict "who wrote take me home country roads"
  carag eval  --data data/nq_conflict --out runs/my_run         # all four systems, scored
  carag sweep --run runs/dev_run                                # tune arbitration thresholds offline
  carag report --run runs/my_run                                # re-score an existing run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

DEFAULT_QA = "deepset/roberta-base-squad2"
# small: fits a 4 GB laptop GPU (helpers on CPU) or an Apple-silicon laptop.  large: ~18 GB GPU.
PROFILES = {
    "tiny": {"generator": "hf:Qwen/Qwen2.5-1.5B-Instruct", "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"},
    "laptop": {"generator": "hf:Qwen/Qwen2.5-3B-Instruct", "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"},
    "large": {"generator": "hf:Qwen/Qwen2.5-7B-Instruct",
              "nli_model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"},
}
DEFAULT_DENSE = "BAAI/bge-small-en-v1.5"


def _auto_profile(device: str) -> str:
    """large (Qwen2.5-7B) when at least 17 GB of GPU memory is *free*, otherwise the laptop bundle.

    Free rather than total memory, so a second process on a busy GPU falls back instead of crashing."""
    from .devices import free_gpu_bytes

    free = free_gpu_bytes(device)
    if free is not None and free >= 17 * 1024**3:
        return "large"
    if free is not None:
        import torch

        if torch.cuda.get_device_properties(device).total_memory >= 18 * 1024**3:
            print(f"[carag] only {free / 1024**3:.1f} GB of GPU memory is free (another process is using the GPU); "
                  "using the laptop profile instead of large", flush=True)
    return "laptop"


def _apply_profile(args) -> None:
    if args.profile == "auto":
        from .devices import pick_device

        args.profile = _auto_profile(pick_device(args.device))
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.profile == "laptop" and getattr(args, "quantize", "auto") == "auto":
        from .devices import free_gpu_bytes, pick_device

        free = free_gpu_bytes(pick_device(args.device))
        if free is not None and free < 10 * 1024**3:
            args.quantize = "4bit"  # judge by free memory: the GPU may be shared


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", required=True, help="directory produced by build-data")
    p.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--profile", default="auto", choices=["auto", *PROFILES],
                   help="model bundle. auto = large on a GPU with >= 18 GB, else laptop. "
                        "laptop: 4 GB GPU with 4-bit, or 16 GB Mac; tiny: 8 GB Mac; large: 20 GB GPU")
    p.add_argument("--quantize", default="auto", choices=["auto", "4bit", "none"],
                   help="4-bit LLM weights (CUDA only); auto = on when the GPU has < 10 GB")
    p.add_argument("--generator", default=None, help="override: hf:<model> or claude:<model>")
    p.add_argument("--dense-model", default=DEFAULT_DENSE)
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                   help="cross-encoder that re-scores the top 30 candidates; 'none' to disable")
    p.add_argument("--relevance-gap", type=float, default=2.0,
                   help="ignore claims from passages scoring this far below the best passage (needs a reranker)")
    p.add_argument("--k", type=int, default=5, help="passages per question")
    p.add_argument("--claim-extractor", default="llm", choices=["llm", "extractive", "hybrid"],
                   help="llm: the generator reads each passage separately; extractive: SQuAD2 QA model; "
                        "hybrid: llm, with the QA model as fallback where the llm found nothing")
    p.add_argument("--claim-prompt", default="strict", choices=["strict", "soft"])
    p.add_argument("--qa-model", default=DEFAULT_QA, help="model for --claim-extractor extractive")
    p.add_argument("--nli-model", default=None, help="override the profile's NLI model")
    p.add_argument("--contra-threshold", type=float, default=0.5)
    p.add_argument("--entail-threshold", type=float, default=0.5)
    p.add_argument("--verify-threshold", type=float, default=0.3)
    p.add_argument("--margin-threshold", type=float, default=0.1)
    p.add_argument("--no-distinct-conflict", action="store_true",
                   help="only NLI contradiction (not distinct unreconciled answers) counts as conflict")
    p.add_argument("--merge-rule", default="mean", choices=["max", "mean"])
    p.add_argument("--pair-check", action=argparse.BooleanOptionalAction, default=False,
                   help="on a detected conflict, ask the LLM whether the two clashing passages are about the same subject")
    p.add_argument("--reask-none", action=argparse.BooleanOptionalAction, default=True,
                   help="re-ask the extractor (softer prompt) for highly relevant passages where it found no answer")
    p.add_argument("--conflict-relevance-gap", type=float, default=None,
                   help="a rival only counts as a conflict if its passage is within this many reranker logits of the top answer")
    p.add_argument("--reliability", help="JSON file {source: reliability} overriding the default table")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--aux-device", default=None,
                   help="device for NLI/embedding/QA models (default: same as --device, or cpu on a <6 GB GPU)")


def _build_system(args, use_llm: bool = True):
    from .arbitration import ArbitrationConfig, EvidenceArbiter
    from .claims import ExtractiveClaimExtractor, HybridClaimExtractor, LLMClaimExtractor
    from .devices import pick_aux_device, pick_device
    from .data import load_corpus
    from .generator import make_generator
    from .nli import NLIModel
    from .pipeline import ConflictAwareRAG
    from .retriever import build_retriever
    from .sources import SourceReliability

    _apply_profile(args)
    device = pick_device(args.device)
    aux = pick_aux_device(device, args.aux_device)
    corpus = load_corpus(args.data)
    t = time.time()
    retriever = build_retriever(args.retriever, corpus, args.dense_model, aux, cache_dir=Path(args.data) / "cache",
                                reranker=None if args.reranker == "none" else args.reranker)
    print(f"[carag] profile={args.profile} device={device} aux={aux}; indexed {len(corpus)} passages "
          f"({args.retriever}) in {time.time() - t:.1f}s", flush=True)
    rel = SourceReliability.from_json(args.reliability) if args.reliability else SourceReliability()
    arbiter = EvidenceArbiter(rel, ArbitrationConfig(
        entail_threshold=args.entail_threshold, contra_threshold=args.contra_threshold,
        margin_threshold=args.margin_threshold, verify_threshold=args.verify_threshold,
        distinct_is_conflict=not args.no_distinct_conflict, merge_rule=args.merge_rule,
        relevance_gap=args.relevance_gap, conflict_relevance_gap=args.conflict_relevance_gap))
    gen = make_generator(args.generator, device, args.quantize) if use_llm else None
    if args.claim_extractor == "extractive":
        extractor = ExtractiveClaimExtractor(args.qa_model, aux)
    elif gen is None:
        raise SystemExit("--claim-extractor llm/hybrid needs a generator")
    elif args.claim_extractor == "hybrid":
        extractor = HybridClaimExtractor(LLMClaimExtractor(gen, args.claim_prompt), ExtractiveClaimExtractor(args.qa_model, aux))
    else:
        extractor = LLMClaimExtractor(gen, args.claim_prompt)
    return ConflictAwareRAG(retriever, extractor, NLIModel(args.nli_model, aux), arbiter, gen, k=args.k,
                            pair_check=args.pair_check, reask_none=args.reask_none)


def cmd_build_data(args) -> None:
    from .data import build_dataset

    meta = build_dataset(args.out, n=args.n, n_distractors=args.distractors, seed=args.seed, split=args.split,
                         substituter=args.substituter, sub_model=args.sub_model)
    print(json.dumps(meta, indent=2))


def _retrieval_stats(instances, preds) -> dict:
    first = next(iter(preds.values()))
    by_q = {r["qid"]: r for r in first}
    gold = conf = both = n_conf = 0
    for inst in instances:
        pids = {x["pid"] for x in by_q[inst["qid"]]["retrieved"]}
        g = any(p in pids for p in inst["gold_pids"])
        c = any(p in pids for p in inst["conflict_pids"])
        gold += g
        if inst["conflict_pids"]:
            n_conf += 1
            conf += c
            both += g and c
    n = len(instances)
    return {"instances": n, "gold_recall@k": gold / n, "conflict_recall@k (conflict scenarios)": conf / max(n_conf, 1),
            "gold+conflict both retrieved (conflict scenarios)": both / max(n_conf, 1)}


def _write_report(run: Path, instances: list[dict], preds: dict[str, list[dict]], title: str) -> str:
    from .eval import aggregate, render_report, score_instance

    by_qid = {x["qid"]: x for x in instances}
    scored = [score_instance(by_qid[r["qid"]], r) for rs in preds.values() for r in rs]
    agg = aggregate(scored)
    retrieval = _retrieval_stats(instances, preds)
    (run / "scored.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scored))
    (run / "metrics.json").write_text(json.dumps({"retrieval": retrieval, "systems": agg}, indent=2))
    report = render_report(agg, retrieval, title)
    (run / "report.md").write_text(report)
    return report


def cmd_eval(args) -> None:
    from .data import load_instances

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    instances = load_instances(args.data)
    if args.limit:
        rng = random.Random(args.seed)
        instances = rng.sample(instances, min(args.limit, len(instances)))
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    needs_llm = bool(set(systems) - {"arbitration_only"}) or args.claim_extractor != "extractive"
    rag = _build_system(args, use_llm=needs_llm)

    t = time.time()
    results = rag.run([(x["qid"], x["question"]) for x in instances], systems, cache_dir=run / "cache",
                      cache_tag=json.dumps({k: v for k, v in sorted(vars(args).items())
                                            if k not in ("func", "out", "systems")}, default=str))
    elapsed = time.time() - t
    preds = {s: [r.to_dict() for r in rs] for s, rs in results.items()}
    with open(run / "predictions.jsonl", "w") as f:
        for s, rs in preds.items():
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config = {k: v for k, v in vars(args).items() if k != "func"}
    import torch

    if torch.cuda.is_available():
        config["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    config.update(elapsed_sec=round(elapsed, 1), n=len(instances),
                  generator_name=rag.generator.name if rag.generator else None)
    (run / "config.json").write_text(json.dumps(config, indent=2))
    print(_write_report(run, instances, preds, f"Results: {run.name}"))
    print(f"[carag] wrote {run}/report.md, metrics.json, predictions.jsonl  ({elapsed:.0f}s)")


def cmd_report(args) -> None:
    from .data import load_instances

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    by_qid = {x["qid"]: x for x in load_instances(config["data"])}
    preds: dict[str, list[dict]] = {}
    with open(run / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            preds.setdefault(r["system"], []).append(r)
    qids = {r["qid"] for r in next(iter(preds.values()))}
    print(_write_report(run, [by_qid[q] for q in sorted(qids)], preds, f"Results: {run.name}"))


def _arb_config(run: Path):
    """ArbitrationConfig that an eval run used (from its config.json)."""
    from .arbitration import ArbitrationConfig

    c = json.loads((run / "config.json").read_text())
    return ArbitrationConfig(
        entail_threshold=c["entail_threshold"], contra_threshold=c["contra_threshold"],
        margin_threshold=c["margin_threshold"], verify_threshold=c["verify_threshold"],
        distinct_is_conflict=not c["no_distinct_conflict"], merge_rule=c["merge_rule"],
        relevance_gap=c.get("relevance_gap"), conflict_relevance_gap=c.get("conflict_relevance_gap"))


def cmd_sweep(args) -> None:
    """Re-run arbitration offline over cached claims/NLI (from an eval run) for a grid of settings.

    Scores the templated (arbitration_only) output, so no model is loaded. Use on a dev split.
    """
    import itertools

    from .arbitration import ArbitrationConfig, EvidenceArbiter
    from .eval.offline import load_traces, rearbitrate

    by_qid, cached = load_traces(args.run)
    grid = {"contra_threshold": [0.5, 0.7, 0.9], "entail_threshold": [0.5, 0.7, 0.9],
            "verify_threshold": [0.3, 0.5, 0.7, 0.9], "margin_threshold": [0.1, 0.2, 0.3],
            "distinct_is_conflict": [True, False], "merge_rule": ["max", "mean"],
            "relevance_gap": [None, 2.0, 3.0], "conflict_relevance_gap": [None, 0.5, 1.0, 2.0, 3.0]}
    keys = list(grid)
    table = []
    for values in itertools.product(*grid.values()):
        a = rearbitrate(cached, by_qid, EvidenceArbiter(config=ArbitrationConfig(**dict(zip(keys, values)))))
        table.append((a["handling_macro"], a["handling_micro"], a["accuracy_answerable"],
                      a["conflict_detection"]["f1"], dict(zip(keys, values))))
    table.sort(key=lambda t: -t[0])
    print(f"{len(cached)} questions, {len(table)} configs. Top {args.top} by macro handling:")
    print("macro  micro  acc    confF1  config")
    for t in table[: args.top]:
        print(f"{t[0]:.3f}  {t[1]:.3f}  {t[2]:.3f}  {t[3]:.3f}  {t[4]}")


def cmd_conflicts(args) -> None:
    """List every conflict the arbiter detected in a run: why it arose, whether it counted, and per scenario."""
    from collections import Counter, defaultdict

    from .data import load_instances

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    by_qid = {x["qid"]: x for x in load_instances(config["data"])}
    seen, rows = set(), []
    for line in open(run / "predictions.jsonl"):
        r = json.loads(line)
        if r.get("arbitration") and r["qid"] not in seen:
            seen.add(r["qid"])
            rows.append(r)
    stats = defaultdict(Counter)
    examples = defaultdict(list)
    for r in rows:
        inst, arb, t = by_qid[r["qid"]], r["arbitration"], r["trace"]
        scen = inst["scenario"]
        stats[scen]["questions"] += 1
        top = [c for c in arb.get("conflicts", []) if c["groups"][0] == 0]
        if not top:
            stats[scen]["no conflict"] += 1
        for c in top:
            stats[scen][("counted: " if c["counts"] else "ignored (low relevance): ") + c["reason"]] += 1
        if t and t.get("pair_verdict"):
            stats[scen]["same-question check: " + t["pair_verdict"]] += 1
        if top and len(examples[scen]) < args.examples:
            c = top[0]
            ca, cb = c["claims"]
            src = lambda k: t["claims"][k]["source"]
            examples[scen].append(
                f"- **{inst['question']}** (correct: {inst['answers'][0]}; planted: {inst['substitute']})  \n"
                f"  {arb['groups'][0]['answer']} [{src(ca)}] vs {arb['groups'][c['groups'][1]]['answer']} [{src(cb)}] - "
                f"{c['reason']}, NLI contradiction {c['nli_contradiction']:.2f}; "
                f"{'counted' if c['counts'] else 'ignored'}; verdict: {arb['decision']}")
    lines = [f"# Conflicts in {run.name}", "", f"{len(rows)} questions. Conflicts between the top claim and a rival, by scenario.", ""]
    for scen in sorted(stats):
        st = stats[scen]
        lines += [f"## {scen} ({st['questions']} questions)", ""]
        lines += [f"| {k} | {v} |" for k, v in st.most_common() if k != "questions"]
        lines.insert(len(lines) - sum(1 for k in st if k != "questions"), "| | count |\n|---|---|")
        lines += ["", "Examples:", ""] + examples[scen] + [""]
    text = "\n".join(lines) + "\n"
    (run / "conflicts.md").write_text(text)
    print(text)


def cmd_learn_reliability(args) -> None:
    """Estimate source reliability from a calibration run's claims (supervised or unsupervised EM)."""
    from .eval.offline import load_traces
    from .reliability import learn_em, learn_supervised
    from .sources import DEFAULT_RELIABILITY

    run = Path(args.run)
    by_qid, cached = load_traces(run)
    cfg = _arb_config(run)
    table, counts = (learn_supervised(cached, by_qid, cfg) if args.method == "supervised" else learn_em(cached, cfg))
    Path(args.out).write_text(json.dumps(table, indent=2))
    print(f"{args.method} reliability from {len(cached)} questions -> {args.out}")
    print(f"{'source':<14} learned  prior-table  claims")
    for s, v in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"{s:<14} {v:.3f}    {DEFAULT_RELIABILITY.get(s, float('nan')):.2f}         {counts[s]}")


def cmd_rearbitrate(args) -> None:
    """Score a run's cached evidence under different reliability tables (no models loaded)."""
    from .arbitration import EvidenceArbiter
    from .eval.offline import load_traces, rearbitrate
    from .sources import SourceReliability

    run = Path(args.run)
    by_qid, cached = load_traces(run)
    cfg = _arb_config(run)
    tables = {"prior table": SourceReliability(), "uniform (no reliability)": SourceReliability({})}
    for path in args.reliability or []:
        tables[Path(path).stem] = SourceReliability.from_json(path)
    print(f"{len(cached)} questions from {run} (templated arbitration output)")
    print(f"{'reliability':<26} acc    misled  handling  " + "  ".join(f"{sc[:12]:<12}" for sc in
          ("balanced", "misinfo_maj", "misinfo_sgl", "no_conflict")))
    out = {}
    for name, rel in tables.items():
        a = rearbitrate(cached, by_qid, EvidenceArbiter(rel, cfg))
        out[name] = a
        sc = a["by_scenario"]
        cells = "  ".join(f"{100 * sc[k]['handled_ok']:<12.1f}" for k in
                          ("balanced_conflict", "misinfo_majority", "misinfo_single", "no_conflict"))
        print(f"{name:<26} {100 * a['accuracy_answerable']:.1f}   {100 * a['misled_rate_answerable']:.1f}     "
              f"{100 * a['handling_macro']:.1f}      {cells}")
    (run / "reliability_ablation.json").write_text(json.dumps(out, indent=2))


def cmd_judge(args) -> None:
    """Grade every output of a run with an LLM judge and compare with the string-matching metrics."""
    from .data import load_instances
    from .eval import aggregate, render_report
    from .eval.judge import judge_prompt, parse_judgement, score_judged
    from .generator import make_generator

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    by_qid = {x["qid"]: x for x in load_instances(config["data"])}
    preds = [json.loads(l) for l in open(run / "predictions.jsonl")]
    string = {(s["qid"], s["system"]): s for s in map(json.loads, open(run / "scored.jsonl"))}
    judge = make_generator(args.judge, args.device)
    judge.max_new_tokens = 16
    outs = judge.generate([judge_prompt(by_qid[r["qid"]], r) for r in preds])
    judged = [score_judged(by_qid[r["qid"]], r, *parse_judgement(o)) for r, o in zip(preds, outs)]
    (run / "judged.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    agg = aggregate(judged)
    agree = {}
    for key in ("correct", "misled", "flagged", "handled_ok"):
        same = [j[key] == string[(j["qid"], j["system"])][key] for j in judged if j["evidence_complete"]]
        agree[key] = sum(same) / len(same)
    (run / "metrics_judge.json").write_text(json.dumps({"judge": judge.name, "systems": agg, "agreement": agree}, indent=2))
    report = render_report(agg, None, f"LLM-judge results: {run.name} (judge: {judge.name})")
    report += "\n## Agreement with string-matching metrics (evidence-complete instances)\n\n| signal | agreement |\n|---|---|\n"
    report += "".join(f"| {k} | {100 * v:.1f} |\n" for k, v in agree.items())
    (run / "report_judge.md").write_text(report)
    print(report)


def _print_answer(out: str, a, verbose: bool = True) -> None:
    if verbose:
        print("\n=== retrieved ===")
        for n, (r, c) in enumerate(zip(a.retrieved, a.claims), 1):
            claim = f'claim="{c.answer}"' if c.has_answer else "no answer"
            print(f"[{n}] {r.passage.source:<13} {claim}\n    {r.passage.text[:160]}...")
        arb = a.arbitration
        print(f"\n=== arbitration === decision={arb.decision} conflict={arb.conflict} margin={arb.margin:.2f}")
        for g in arb.groups:
            print(f"  '{g.answer}': docs {[m + 1 for m in g.members]} sources={g.sources} support={g.support:.2f}")
        for c in arb.conflicts:
            x, y = c["groups"]
            ca, cb = c["claims"]
            status = "COUNTS" if c["counts"] else "ignored"
            print(f"  conflict [{status}]: '{arb.groups[x].answer}' (doc {ca + 1}) vs '{arb.groups[y].answer}' (doc {cb + 1})"
                  f"\n      reason: {c['reason']}; NLI contradiction {c['nli_contradiction']:.2f}, "
                  f"compatibility {c['compatibility']:.2f}" + (f"; {c['note']}" if c["note"] else ""))
        if a.off_subject:
            print(f"  same-question check: docs {[m + 1 for m in a.off_subject]} are about a different subject; dropped")
        print("\n=== answer ===")
    print(out)


def cmd_ask(args) -> None:
    rag = _build_system(args)
    out, a = rag.ask(args.question, template=args.template)
    _print_answer(out, a, verbose=not args.quiet)


def cmd_chat(args) -> None:
    """Interactive loop: models are loaded once, then each question takes a few seconds."""
    rag = _build_system(args)
    print("Ask a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        out, a = rag.ask(q, template=args.template)
        _print_answer(out, a, verbose=not args.quiet)


def cmd_serve(args) -> None:
    from .data import load_instances
    from .serve import serve

    rag = _build_system(args)
    try:
        instances = load_instances(args.data)
    except FileNotFoundError:  # a plain corpus without benchmark questions
        instances = []
    serve(rag, instances, args.profile, args.host, args.port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="carag", description="Conflict-aware RAG with evidence arbitration")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("build-data", help="build the synthetic NQ knowledge-conflict benchmark")
    p.add_argument("--out", default="data/nq_conflict")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--distractors", type=int, default=-1, help="background passages; -1 = all ~64k NQ passages")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--substituter", default="llm", choices=["llm", "pool"],
                   help="how to make counterfactual answers for names/phrases (years/numbers always use rules)")
    p.add_argument("--sub-model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model for --substituter llm")
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("eval", help="run systems on the benchmark and score them")
    _add_model_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--systems", default="vanilla,prompted,conflict_aware,arbitration_only")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="re-score an existing run")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sweep", help="tune arbitration thresholds offline on a run's cached claims/NLI")
    p.add_argument("--run", required=True)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("conflicts", help="report every conflict detected in a run, with reasons")
    p.add_argument("--run", required=True)
    p.add_argument("--examples", type=int, default=5, help="examples per scenario")
    p.set_defaults(func=cmd_conflicts)

    p = sub.add_parser("learn-reliability", help="estimate source reliability from a calibration run")
    p.add_argument("--run", required=True, help="eval run on a calibration (dev) split")
    p.add_argument("--method", default="supervised", choices=["supervised", "em"])
    p.add_argument("--out", required=True, help="output JSON, usable with --reliability")
    p.set_defaults(func=cmd_learn_reliability)

    p = sub.add_parser("rearbitrate", help="re-score a run's cached evidence under other reliability tables")
    p.add_argument("--run", required=True)
    p.add_argument("--reliability", nargs="*", help="JSON tables to compare with the prior and uniform tables")
    p.set_defaults(func=cmd_rearbitrate)

    p = sub.add_parser("judge", help="grade a run's outputs with an LLM judge")
    p.add_argument("--run", required=True)
    p.add_argument("--judge", default="hf:Qwen/Qwen2.5-7B-Instruct", help="hf:<model> or claude:<model>")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("ask", help="answer one question with the full arbitration trace")
    _add_model_args(p)
    p.add_argument("question")
    p.add_argument("--template", action="store_true", help="templated answer from the arbitration (no LLM writing)")
    p.add_argument("--quiet", action="store_true", help="print only the answer, not the evidence trace")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="local web UI with evidence view and live trust sliders")
    _add_model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("chat", help="interactive question loop (models stay loaded)")
    _add_model_args(p)
    p.add_argument("--template", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_chat)

    args = ap.parse_args(argv)
    try:
        args.func(args)
    except RuntimeError as e:
        from .devices import OOM_HINT, is_oom

        if is_oom(e):
            raise SystemExit(f"[carag] {OOM_HINT}") from e
        raise


if __name__ == "__main__":
    main()
