"""Synthetic knowledge-conflict benchmark built from Natural Questions + gold Wikipedia passages.

Source data: `florin-hf/nq_open_gold` (NQ-open questions paired with the gold Wikipedia passage that
contains the answer). For each question we create a *conflicting* passage by entity substitution
(Longpre et al., 2021): every mention of the gold answer in the gold passage is replaced with a
plausible same-type alternative (a shifted year/number, or another NQ answer to a question of the
same wh-type). Each instance is assigned one scenario:

  no_conflict        gold passage only (high-reliability source)            -> expected: answer gold
  balanced_conflict  gold + 1 edited passage, both from the same tier       -> expected: report disagreement
  misinfo_single     gold (high tier) + 1 edited passage (low tier)         -> expected: answer gold
  misinfo_majority   gold (high tier) + 2 edited passages (low tier)        -> expected: answer gold

All passages go into one shared corpus together with the gold passages of every other NQ question
(~64k Wikipedia passages) as background, so the retriever - not the dataset - decides what the
generator sees, and the same corpus can answer general NQ-style questions.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..schema import Passage
from ..sources import HIGH_TIER, LOW_TIER
from ..text import contains_any, normalize_answer, same_answer

SCENARIOS = ("no_conflict", "balanced_conflict", "misinfo_single", "misinfo_majority")
DATASET = "florin-hf/nq_open_gold"

_MONTHS = "january february march april may june july august september october november december".split()
_WH = ("how many", "how much", "what year", "when", "who", "where", "which", "what")


def wh_type(question: str) -> str:
    q = question.lower()
    for w in _WH:
        if re.search(rf"\b{w}\b", q):
            return w
    return "other"


def answer_type(ans: str) -> str:
    a = ans.strip()
    if re.fullmatch(r"(1[0-9]{3}|20[0-2][0-9])", a):
        return "year"
    if re.fullmatch(r"\d{1,3}(,\d{3})+|\d+(\.\d+)?", a):
        return "number"
    if re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", a) or (
            any(m in a.lower().split() for m in _MONTHS) and re.search(r"\d", a)):
        return "date"  # any answer containing a year or a day-of-month
    if re.fullmatch(r"(about |over |around )?\d[\d,.]*( [a-z]+){0,2}", a):
        return "quantity"  # "20 episodes", "about 300 km"
    toks = [t for t in a.split() if t.isalpha()]
    if toks and all(t[0].isupper() for t in toks if t.lower() not in {"of", "the", "and", "de", "von", "van"}):
        return "proper"
    return "other"


def _mention_re(ans: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(ans) + r"(?!\w)", re.IGNORECASE)


def _shift_year(y: int, rng: random.Random) -> int:
    for _ in range(20):
        d = rng.choice([-1, 1]) * rng.randint(3, 40)
        if 1000 <= y + d <= 2023:
            return y + d
    return y - 7


def _perturb_number(s: str, rng: random.Random) -> str | None:
    raw = s.replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    for _ in range(20):
        f = rng.choice([rng.uniform(0.3, 0.8), rng.uniform(1.3, 3.0)])
        new = val * f
        if "." in raw:
            out = f"{new:.{len(raw.split('.')[1])}f}"
        else:
            n = int(round(new))
            if n == int(val) or n <= 0:
                n = int(val) + rng.randint(2, 9)
            out = f"{n:,}" if "," in s else str(n)
        if out != s:
            return out
    return None


def make_substitute(ans: str, question: str, golds: list[str], passage: str,
                    pool: dict[tuple[str, str], list[str]], rng: random.Random) -> str | None:
    atype = answer_type(ans)
    if atype == "year":
        return str(_shift_year(int(ans), rng))
    if atype == "number":
        return _perturb_number(ans, rng)
    if atype == "quantity":
        m = re.search(r"\d[\d,.]*", ans)
        new = _perturb_number(m.group().rstrip(".,"), rng)
        return None if new is None else ans[: m.start()] + new + ans[m.start() + len(m.group().rstrip(".,")):]
    if atype == "date":
        m = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", ans)
        if m:
            return ans[: m.start()] + str(_shift_year(int(m.group()), rng)) + ans[m.end():]
        m = re.search(r"\b([1-9]|[12][0-9]|3[01])\b", ans)
        if m:
            day = int(m.group())
            new = rng.choice([d for d in range(1, 29) if abs(d - day) > 2])
            return ans[: m.start()] + str(new) + ans[m.end():]
        return None
    # Entity swap: another NQ answer to a question with the same wh-word and answer type.
    cands = pool.get((wh_type(question), atype)) or (pool.get(("*", atype)) if atype == "proper" else None) or []
    n_tok = len(ans.split())
    for _ in range(100):
        if not cands:
            return None
        c = rng.choice(cands)
        if abs(len(c.split()) - n_tok) > 1:
            continue
        if any(same_answer(c, g) for g in golds) or contains_any(passage, [c]):
            continue
        return c
    return None


def _answer_pool(rows) -> dict[tuple[str, str], list[str]]:
    pool: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        for a in r["answers"][:1]:
            if 1 <= len(a.split()) <= 5:
                t = answer_type(a)
                pool[(wh_type(r["question"]), t)].add(a)
                pool[("*", t)].add(a)
    return {k: sorted(v) for k, v in pool.items()}


RULE_TYPES = {"year", "number", "date", "quantity"}

SUB_SYSTEM = "You create counterfactual test data for evaluating question answering systems."
SUB_PROMPT = """Question: {q}
Correct answer: {a}

Give ONE different answer that is plausible for this question but factually incorrect. It must be the same kind of thing as the correct answer (a person -> another real person, a place -> another place, a band -> another band, a phrase -> a comparable phrase), with similar length and format. It must not be a synonym, alias, abbreviation or part of the correct answer.
Output only the alternative answer, nothing else."""


def _clean_llm_sub(text: str) -> str:
    s = text.strip().splitlines()[0] if text.strip() else ""
    s = re.sub(r"^(alternative answer|answer)\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip().strip("\"'`*").rstrip(".").strip()


def _valid_sub(sub: str | None, golds: list[str], passage: str) -> bool:
    return bool(sub) and len(sub.split()) <= 8 and not any(same_answer(sub, g) for g in golds) \
        and not contains_any(passage, [sub])


def build_dataset(out_dir: str | Path, n: int = 1000, n_distractors: int = -1, seed: int = 13,
                  split: str = "test", max_answer_tokens: int = 5, max_passage_words: int = 250,
                  substituter: str = "llm", sub_model: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """substituter: 'llm' (LLM proposes same-type wrong answers for names/phrases; years/numbers are
    shifted by rule) or 'pool' (entity swap with other NQ answers of the same wh/answer type)."""
    from datasets import load_dataset

    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_rows = load_dataset(DATASET, split=split)
    train_rows = load_dataset(DATASET, split="train")
    pool = _answer_pool(train_rows)

    # 1. candidate questions whose gold passage mentions the answer
    order = list(range(len(eval_rows)))
    rng.shuffle(order)
    cands, seen_text, skipped = [], set(), Counter()
    rank = {"year": 0, "number": 0, "date": 0, "quantity": 0, "proper": 1, "other": 2}
    for k in order:
        if len(cands) >= int(n * 1.25) + 20:  # head-room for later filtering
            break
        row = eval_rows[k]
        q, text, golds = row["question"].strip(), row["text"].strip(), [a for a in row["answers"] if a.strip()]
        if not golds or len(text.split()) > max_passage_words or text in seen_text:
            skipped["filtered"] += 1
            continue
        present = [a for a in golds if _mention_re(a).search(text) and len(a.split()) <= max_answer_tokens]
        if not present:
            skipped["answer_not_in_passage"] += 1
            continue
        seen_text.add(text)
        # Prefer typed aliases (names, years, numbers) over free-form phrases; then the shortest.
        anchor = min(present, key=lambda a: (rank[answer_type(a)], len(a)))
        cands.append({"question": q, "answers": golds, "present": present, "anchor_answer": anchor,
                      "gold_text": text, "nq_example_id": str(row["example_id"])})

    # 2. substitutes
    subs: list[tuple[str | None, str]] = [(None, "")] * len(cands)
    llm_idx = []
    for i, c in enumerate(cands):
        if substituter == "pool" or answer_type(c["anchor_answer"]) in RULE_TYPES:
            subs[i] = (make_substitute(c["anchor_answer"], c["question"], c["answers"], c["gold_text"], pool, rng),
                       "rule" if answer_type(c["anchor_answer"]) in RULE_TYPES else "pool")
        else:
            llm_idx.append(i)
    if llm_idx:
        from ..generator import HFGenerator

        gen = HFGenerator(sub_model, batch_size=32, max_new_tokens=24)
        outs = gen.generate([(SUB_SYSTEM, SUB_PROMPT.format(q=cands[i]["question"], a=cands[i]["anchor_answer"]))
                             for i in llm_idx])
        del gen
        for i, o in zip(llm_idx, outs):
            c, s = cands[i], _clean_llm_sub(o)
            if _valid_sub(s, c["answers"], c["gold_text"]):
                subs[i] = (s, "llm")
            else:
                skipped["llm_sub_invalid->pool"] += 1
                subs[i] = (make_substitute(c["anchor_answer"], c["question"], c["answers"], c["gold_text"], pool, rng), "pool")

    # 3. edit passages
    instances = []
    for c, (sub, method) in zip(cands, subs):
        if len(instances) >= n:
            break
        if not _valid_sub(sub, c["answers"], c["gold_text"]):
            skipped["no_substitute"] += 1
            continue
        edited = c["gold_text"]
        for a in sorted(c.pop("present"), key=len, reverse=True):  # replace every gold alias mention
            edited = _mention_re(a).sub(lambda _m: sub, edited)
        if contains_any(edited, c["answers"]) or edited == c["gold_text"]:
            skipped["gold_survives_edit"] += 1
            continue
        instances.append({**c, "substitute": sub, "substitute_method": method,
                          "answer_type": answer_type(c["anchor_answer"]), "edited_text": edited})

    # Assign scenarios evenly and sources by scenario.
    corpus: list[Passage] = []
    for i, inst in enumerate(instances):
        qid = f"q{i:05d}"
        scen = SCENARIOS[i % len(SCENARIOS)]
        if scen == "no_conflict":
            srcs = [rng.choice(HIGH_TIER)]
        elif scen == "balanced_conflict":
            srcs = rng.sample(rng.choice([HIGH_TIER, LOW_TIER]), 2)
        elif scen == "misinfo_single":
            srcs = [rng.choice(HIGH_TIER), rng.choice(LOW_TIER)]
        else:
            srcs = [rng.choice(HIGH_TIER), *rng.sample(LOW_TIER, 2)]
        gold = Passage(f"{qid}-gold", inst["gold_text"], srcs[0], meta={"qid": qid, "role": "gold"})
        conflicts = [Passage(f"{qid}-conflict-{c}", inst["edited_text"], s, meta={"qid": qid, "role": "conflict"})
                     for c, s in enumerate(srcs[1:])]
        corpus.extend([gold, *conflicts])
        inst.update(qid=qid, scenario=scen, gold_pids=[gold.pid], conflict_pids=[p.pid for p in conflicts],
                    sources={p.pid: p.source for p in [gold, *conflicts]})

    # Background corpus: gold passages of all other NQ questions (every split), deduplicated.
    eval_texts = {normalize_answer(x["gold_text"]) for x in instances}
    pool, seen = [], set()
    for rows in (train_rows, load_dataset(DATASET, split="validation"), load_dataset(DATASET, split="test")):
        for t in rows["text"]:
            t = t.strip()
            nt = normalize_answer(t)
            if t and nt not in eval_texts and nt not in seen and len(t.split()) <= max_passage_words:
                seen.add(nt)
                pool.append(t)
    rng.shuffle(pool)
    if n_distractors >= 0:
        pool = pool[:n_distractors]

    # Keep scenarios controlled: drop background passages that a benchmark question would retrieve
    # and that also state its gold or substituted answer (they would be unlabeled extra evidence).
    from ..retriever import BM25Retriever

    bm25 = BM25Retriever([Passage(str(i), t) for i, t in enumerate(pool)])
    leaked: set[int] = set()
    for inst in instances:
        for r in bm25.search(inst["question"], 30):
            if contains_any(r.passage.text, inst["answers"] + [inst["substitute"]]):
                leaked.add(int(r.passage.pid))
    added = 0
    for i, t in enumerate(pool):
        if i in leaked:
            continue
        # background passages are unedited Wikipedia text, so they are labeled as such; only the planted
        # benchmark passages (gold + edited) get scenario-specific sources
        corpus.append(Passage(f"d{added:05d}", t, "wikipedia", meta={"role": "distractor"}))
        added += 1
    skipped["background_passages_dropped_as_leaks"] = len(leaked)

    rng.shuffle(corpus)
    with open(out / "instances.jsonl", "w") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    with open(out / "corpus.jsonl", "w") as f:
        for p in corpus:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    meta = {"dataset": DATASET, "split": split, "seed": seed, "n_instances": len(instances),
            "n_corpus": len(corpus), "n_distractors": added,
            "scenarios": dict(Counter(x["scenario"] for x in instances)),
            "answer_types": dict(Counter(x["answer_type"] for x in instances)),
            "substitute_methods": dict(Counter(x["substitute_method"] for x in instances)),
            "skipped": dict(skipped)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_instances(data_dir: str | Path) -> list[dict]:
    with open(Path(data_dir) / "instances.jsonl") as f:
        return [json.loads(l) for l in f]


def load_corpus(data_dir: str | Path) -> list[Passage]:
    with open(Path(data_dir) / "corpus.jsonl") as f:
        return [Passage.from_dict(json.loads(l)) for l in f]
