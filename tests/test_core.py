"""Model-free unit tests (stubbed extractor / NLI / generator)."""
import random

import numpy as np

from carag.arbitration import ArbitrationConfig, EvidenceArbiter, nli_requests
from carag.data.build import _perturb_number, answer_type, make_substitute, wh_type
from carag.eval.metrics import aggregate, score_instance
from carag.pipeline import ConflictAwareRAG
from carag.prompts import parse_output, render_arbitration
from carag.retriever import BM25Retriever
from carag.schema import Claim, Passage
from carag.text import contains, normalize_answer, same_answer, sentence_containing


# ---------------------------------------------------------------- text
def test_normalize_and_match():
    assert normalize_answer("The Wilhelm Conrad Röntgen!") == "wilhelm conrad rontgen"
    assert contains("It was Röntgen in 1901.", "rontgen")
    assert not contains("in 19011", "1901")
    assert same_answer("Röntgen", "Wilhelm Conrad Röntgen")
    assert not same_answer("1901", "1905")


def test_sentence_containing():
    t = "First sentence here. The prize went to Röntgen in 1901. Last one."
    s = t.index("Röntgen")
    assert sentence_containing(t, s, s + 7) == "The prize went to Röntgen in 1901."


# ---------------------------------------------------------------- data
def test_answer_types_and_substitutes():
    assert answer_type("1901") == "year"
    assert answer_type("150,782") == "number"
    assert answer_type("December 10, 1901") == "date"
    assert answer_type("Wilhelm Conrad Röntgen") == "proper"
    assert wh_type("how many players are on a team") == "how many"
    rng = random.Random(0)
    sub = make_substitute("1901", "when was it", ["1901"], "in 1901", {}, rng)
    assert sub != "1901" and sub.isdigit()
    assert _perturb_number("150,782", rng) not in (None, "150,782")
    pool = {("who", "proper"): ["Max Planck", "Marie Curie"]}
    sub = make_substitute("Albert Einstein", "who won", ["Albert Einstein"], "Albert Einstein won", pool, rng)
    assert sub in {"Max Planck", "Marie Curie"}


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_relevant_first():
    ps = [Passage("a", "The Eiffel Tower is in Paris."), Passage("b", "Bananas are yellow fruit."),
          Passage("c", "Paris is the capital of France.")]
    top = BM25Retriever(ps).search("where is the eiffel tower", 2)
    assert top[0].passage.pid == "a"


# ---------------------------------------------------------------- arbitration
def _claim(pid, ans, src, has=True):
    return Claim(Passage(pid, f"text {ans}", src), ans, 0.9, has, f"It is {ans}.")


def _mats(n, contra_pairs=(), entail_pairs=()):
    E = np.eye(n)  # every claim verified by its own passage
    C = np.zeros((n, n))
    for j, i in contra_pairs:
        C[j, i] = C[i, j] = 0.97
    for j, i in entail_pairs:
        E[j, i] = E[i, j] = 0.95
    return E, C


def test_no_conflict_answers():
    claims = [_claim("g", "1901", "wikipedia"), _claim("d", "", "bbc", has=False)]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2))
    assert arb.decision == "answer" and not arb.conflict and arb.best.answer == "1901"


def test_balanced_conflict_disagrees():
    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "britannica")]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2, contra_pairs=[(0, 1)]))
    assert arb.conflict and arb.decision == "disagree"
    assert {g.answer for g in arb.groups} == {"1901", "1925"}


def test_reliability_resolves_misinformation():
    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "forum"), _claim("y", "1925", "social-media")]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(3, contra_pairs=[(0, 1), (0, 2)]))
    assert arb.conflict and arb.decision == "answer" and arb.best.answer == "1901"
    assert arb.groups[arb.rival].members == [1, 2]


def test_compatible_answers_merge():
    claims = [_claim("a", "1901", "wikipedia"), _claim("b", "December 1901", "bbc"), _claim("c", "Paris", "forum")]
    E, C = _mats(3, entail_pairs=[(0, 1)])
    arb = EvidenceArbiter(config=ArbitrationConfig(distinct_is_conflict=False)).arbitrate(claims, E, C)
    assert arb.groups[0].members == [0, 1]
    assert not arb.conflict and arb.decision == "answer"
    # with the default rule, a distinct verified answer that NLI cannot reconcile counts as a conflict,
    # but the weak forum claim loses on support
    arb = EvidenceArbiter().arbitrate(claims, E, C)
    assert arb.conflict and arb.decision == "answer" and arb.groups[0].members == [0, 1]


def test_unverified_claims_are_dropped():
    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "britannica")]
    E, C = _mats(2, contra_pairs=[(0, 1)])
    E[1, 1] = 0.1  # passage x does not actually support its extracted answer
    arb = EvidenceArbiter().arbitrate(claims, E, C)
    assert not arb.conflict and arb.best.answer == "1901" and len(arb.groups) == 1


def test_abstain_without_answers():
    claims = [_claim("a", "", "wikipedia", has=False)]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(1))
    assert arb.decision == "abstain"
    assert "unknown" in render_arbitration(arb, claims)


def test_nli_requests_cover_all_premises():
    claims = [_claim("a", "x", "wikipedia"), _claim("b", "", "bbc", has=False), _claim("c", "y", "forum")]
    reqs = nli_requests(claims)
    assert {(j, i) for j, i, _, _ in reqs} == {(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2)}


# ---------------------------------------------------------------- parsing & metrics
def test_parse_output():
    p = parse_output("ANSWER: Sources disagree: 1901 vs 1925\nCONFLICT: yes\nRESPONSE: [1] says 1901 but [2] says 1925.")
    assert p.flagged_conflict and p.conflict_field is True
    p = parse_output("ANSWER: 1901\nRESPONSE: The documents do not conflict; it was 1901.")
    assert not p.flagged_conflict
    p = parse_output("ANSWER: 1901\nRESPONSE: Document 2 contradicts this, claiming 1925.")
    assert p.flagged_conflict


def _inst(scen):
    return {"qid": "q1", "scenario": scen, "answers": ["1901"], "substitute": "1925",
            "gold_pids": ["g"], "conflict_pids": [] if scen == "no_conflict" else ["x"]}


def _res(answer, out, flagged, pids, system="s"):
    return {"qid": "q1", "system": system, "answer": answer, "output": out, "flagged_conflict": flagged,
            "retrieved": [{"pid": p} for p in pids]}


def test_scoring_rules():
    ok = score_instance(_inst("balanced_conflict"), _res("Sources disagree: 1901 vs 1925", "1901 vs 1925", True, ["g", "x"]))
    assert ok["handled_ok"] and ok["expected"] == "disagree"
    bad = score_instance(_inst("balanced_conflict"), _res("1925", "1925", False, ["g", "x"]))
    assert not bad["handled_ok"]
    mis = score_instance(_inst("misinfo_single"), _res("1925", "1925", False, ["g", "x"]))
    assert mis["misled"] and not mis["handled_ok"]
    good = score_instance(_inst("misinfo_single"), _res("1901", "1901; [2] claims 1925", True, ["g", "x"]))
    assert good["handled_ok"]
    # conflict passage not retrieved -> behaves like no-conflict
    nc = score_instance(_inst("misinfo_single"), _res("1901", "1901", False, ["g"]))
    assert nc["handled_ok"] and not nc["conflict_in_context"]
    agg = aggregate([ok, mis])
    assert agg["s"]["conflict_detection"]["recall"] == 0.5


# ---------------------------------------------------------------- pipeline with stubs
class StubExtractor:
    def extract(self, pairs):
        out = []
        for _, p in pairs:
            ans = p.meta.get("ans", "")
            out.append(Claim(p, ans, 0.9, bool(ans), f"The answer is {ans}."))
        return out


class StubNLI:
    def predict(self, pairs):
        res = []
        for prem, hyp in pairs:
            a = hyp.removeprefix("The answer is ").rstrip(".")
            if a in prem:
                res.append([0.95, 0.04, 0.01])
            elif any(y in prem for y in ("1901", "1925")):
                res.append([0.01, 0.02, 0.97])
            else:
                res.append([0.02, 0.96, 0.02])
        return np.array(res)


class StubGen:
    name = "stub"

    def generate(self, prompts):
        return ["ANSWER: 1901\nCONFLICT: no\nRESPONSE: ok" for _ in prompts]


def test_pipeline_end_to_end_with_stubs():
    corpus = [
        Passage("g", "The first Nobel Prize in Physics was awarded in 1901.", "wikipedia", meta={"ans": "1901"}),
        Passage("x", "The first Nobel Prize in Physics was awarded in 1925.", "britannica", meta={"ans": "1925"}),
        Passage("d", "Bananas are a yellow fruit grown in the tropics.", "bbc"),
    ]
    rag = ConflictAwareRAG(BM25Retriever(corpus), StubExtractor(), StubNLI(), EvidenceArbiter(), StubGen(), k=3)
    res = rag.run([("q1", "when was the first nobel prize in physics awarded")])
    ao = res["arbitration_only"][0]
    assert ao.flagged_conflict and "Sources disagree" in ao.answer
    assert res["conflict_aware"][0].arbitration["decision"] == "disagree"
    assert len(res["vanilla"]) == 1


def test_margin_threshold_configurable():
    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "personal-blog")]
    mats = _mats(2, contra_pairs=[(0, 1)])
    assert EvidenceArbiter().arbitrate(claims, *mats).decision == "answer"
    strict = EvidenceArbiter(config=ArbitrationConfig(margin_threshold=0.9))
    assert strict.arbitrate(claims, *mats).decision == "disagree"


def test_arbitrated_prompt_fixes_format_per_verdict():
    from carag.prompts import arbitrated_prompt
    from carag.schema import Retrieved
    from carag.sources import SourceReliability

    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "britannica")]
    rs = [Retrieved(c.passage, 1.0, i) for i, c in enumerate(claims)]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2, contra_pairs=[(0, 1)]))
    _, user = arbitrated_prompt("when?", rs, claims, arb, SourceReliability())
    assert "ANSWER: Sources disagree: 1901 vs 1925" in user and "Do NOT pick a side" in user
    arb = EvidenceArbiter().arbitrate(claims[:1], *_mats(1))
    _, user = arbitrated_prompt("when?", rs[:1], claims[:1], arb, SourceReliability())
    assert "CONFLICT: no" in user and "Sources disagree" not in user


def test_run_cache_resumes_without_regenerating(tmp_path):
    corpus = [Passage("g", "The first Nobel Prize in Physics was awarded in 1901.", "wikipedia", meta={"ans": "1901"}),
              Passage("d", "Bananas are a yellow fruit grown in the tropics.", "bbc")]

    class CountingGen(StubGen):
        calls = 0

        def generate(self, prompts):
            CountingGen.calls += 1
            return super().generate(prompts)

    rag = ConflictAwareRAG(BM25Retriever(corpus), StubExtractor(), StubNLI(), EvidenceArbiter(), CountingGen(), k=2)
    items = [("q1", "when was the first nobel prize in physics awarded")]
    first = rag.run(items, ("vanilla", "conflict_aware"), cache_dir=tmp_path)
    n = CountingGen.calls
    second = rag.run(items, ("vanilla", "conflict_aware"), cache_dir=tmp_path)
    assert CountingGen.calls == n
    assert first["conflict_aware"][0].output == second["conflict_aware"][0].output


def test_relevance_gate_drops_near_topic_claims():
    claims = [_claim("g", "James Arthur", "wikipedia"), _claim("d", "James Morrison", "bbc")]
    mats = _mats(2, contra_pairs=[(0, 1)])
    assert EvidenceArbiter().arbitrate(claims, *mats).conflict  # no relevance scores: both claims count
    arb = EvidenceArbiter().arbitrate(claims, *mats, relevance=[8.0, 3.0])  # passage 2 is far less relevant
    assert not arb.conflict and arb.best.answer == "James Arthur" and len(arb.groups) == 1
    arb = EvidenceArbiter().arbitrate(claims, *mats, relevance=[8.0, 7.5])
    assert arb.conflict


def test_bm25_cache_roundtrip(tmp_path):
    ps = [Passage("a", "The Eiffel Tower is in Paris."), Passage("b", "Bananas are yellow fruit.")]
    first = BM25Retriever(ps, cache_dir=tmp_path).search("eiffel tower", 1)
    assert list(tmp_path.glob("bm25-*.pkl"))
    second = BM25Retriever(ps, cache_dir=tmp_path).search("eiffel tower", 1)
    assert first[0].passage.pid == second[0].passage.pid == "a"
    assert BM25Retriever(ps).search("zzzz", 3) == []


def test_hybrid_extractor_falls_back_only_when_llm_has_no_answer():
    from carag.claims import HybridClaimExtractor

    class Fixed:
        def __init__(self, answers):
            self.answers = answers

        def extract(self, pairs):
            return [Claim(p, a, 1.0, bool(a), a) for (_, p), a in zip(pairs, self.answers)]

    pairs = [("q", Passage("a", "x")), ("q", Passage("b", "y"))]
    out = HybridClaimExtractor(Fixed(["llm", ""]), Fixed(["qa"])).extract(pairs)
    assert [c.answer for c in out] == ["llm", "qa"]


def _cached_question(qid, items, contra=True):
    """items = [(answer, source)] -> one cached trace row where all claims are verified and mutually contradict."""
    claims = [_claim(f"{qid}-{k}", a, s) for k, (a, s) in enumerate(items)]
    pairs = [(i, j) for i in range(len(items)) for j in range(i) if items[i][0] != items[j][0]] if contra else []
    E, C = _mats(len(items), contra_pairs=pairs)
    return ({"qid": qid}, claims, E, C, None)


def test_learned_reliability_supervised_and_em():
    from carag.reliability import learn_em, learn_supervised

    cached, by_qid = [], {}
    for n in range(30):  # good source is right; in a third of questions a bad source contradicts it
        qid = f"q{n}"
        by_qid[qid] = {"answers": ["right"]}
        items = [("right", "good"), ("right", "ok")] + ([("wrong", "bad")] if n % 3 == 0 else [])
        cached.append(_cached_question(qid, items))
    sup, counts = learn_supervised(cached, by_qid)
    assert sup["good"] > 0.9 and sup["bad"] < 0.2 and counts["bad"] == 10
    em, _ = learn_em(cached)
    assert em["good"] > em["bad"] and em["bad"] < 0.3  # outvoted source is discovered without labels


def test_judge_parsing_and_scoring():
    from carag.eval.judge import parse_judgement, score_judged

    assert parse_judgement("VERDICT: C\\nMENTIONS_CONFLICT: yes") == ("C", True)
    assert parse_judgement("verdict: **A**\\nmentions_conflict: no") == ("A", False)
    assert parse_judgement("garbage") == ("D", False)
    res = _res("x", "x", False, ["g", "x"])
    assert score_judged(_inst("balanced_conflict"), res, "C", True)["handled_ok"]
    assert not score_judged(_inst("balanced_conflict"), res, "A", True)["handled_ok"]
    j = score_judged(_inst("misinfo_single"), res, "B", False)
    assert j["misled"] and not j["handled_ok"]
    assert score_judged(_inst("no_conflict"), _res("x", "x", False, ["g"]), "A", False)["handled_ok"]
    assert not score_judged(_inst("no_conflict"), _res("x", "x", False, ["g"]), "A", True)["handled_ok"]


def test_web_api_ask_and_live_rearbitration():
    """The UI's server: ask a question, then flip the verdict by changing source trust (no model call)."""
    import json as _json
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from carag.serve import App, make_handler

    corpus = [
        Passage("g", "The first Nobel Prize in Physics was awarded in 1901.", "wikipedia", meta={"ans": "1901"}),
        Passage("x", "The first Nobel Prize in Physics was awarded in 1925.", "forum", meta={"ans": "1925"}),
        Passage("d", "Bananas are a yellow fruit grown in the tropics.", "bbc"),
    ]
    rag = ConflictAwareRAG(BM25Retriever(corpus), StubExtractor(), StubNLI(), EvidenceArbiter(), None, k=3)
    inst = {"question": "when was the first nobel prize in physics awarded", "scenario": "misinfo_single",
            "answers": ["1901"], "substitute": "1925", "conflict_pids": ["x"]}
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(App(rag, [inst], "test")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=_json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json"})
        return _json.loads(urllib.request.urlopen(req).read())

    try:
        assert b"Conflict-Aware RAG" in urllib.request.urlopen(base + "/").read()
        assert call("/api/info")["passages"] == 3
        assert call("/api/examples")[0]["scenario"] == "misinfo_single"
        assert call("/api/search?q=nobel%20prize")[0]["text"].startswith("The first Nobel")
        d = call("/api/ask", {"question": inst["question"]})
        assert d["arbitration"]["decision"] == "answer" and d["arbitration"]["conflict"] and "1901" in d["answer"]
        assert d["benchmark"]["planted"] == "1925" and d["passages"][0]["claim"]
        flipped = call("/api/rearbitrate", {"id": d["id"], "reliability": {"forum": 0.95, "wikipedia": 0.1}})
        assert "1925" in flipped["answer"] and "1901" not in flipped["answer"]
        tie = call("/api/rearbitrate", {"id": d["id"], "reliability": {"forum": 0.9, "wikipedia": 0.9}})
        assert tie["arbitration"]["decision"] == "disagree"
    finally:
        server.shutdown()


def test_judge_sees_only_user_facing_text():
    from carag.eval.judge import judge_prompt

    inst = {"question": "when?", "answers": ["1901"], "substitute": "1925"}
    _, prompt = judge_prompt(inst, {"output": "ANSWER: 1901\nCONFLICT: no\nRESPONSE: It was 1901 [1]."})
    assert "Answer: 1901" in prompt and "Explanation: It was 1901 [1]." in prompt and "CONFLICT: no" not in prompt


def test_relevance_gate_measured_from_best_answering_passage():
    # passage 0 is the most relevant but states no answer; the only answer must survive the gate
    claims = [_claim("top", "", "wikipedia", has=False), _claim("a", "LIFO", "wikipedia")]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2), relevance=[5.7, -6.3])
    assert arb.decision == "answer" and arb.best.answer == "LIFO"


def test_answer_matching_tolerates_wording_variants():
    from carag.eval.metrics import answer_candidates, matches

    assert answer_candidates("Sources disagree: Copernicus vs Aristarchus of Samos") == ["Copernicus", "Aristarchus of Samos"]
    assert matches("Sources disagree: Copernicus vs Aristarchus of Samos", ["Copernicus of Poland"])
    assert matches("Sources disagree: Coal Township, Pennsylvania vs Rural Valley", ["in Coal Township"])
    assert matches("seven", ["seven cards"]) and not matches("five", ["seven cards"])
    assert not matches("Bill Clinton", ["Bill Nighy"]) and not matches("1925", ["1901"])
    inst = {"qid": "q1", "scenario": "balanced_conflict", "answers": ["Aristarchus of Samos"],
            "substitute": "Copernicus of Poland", "gold_pids": ["g"], "conflict_pids": ["x"]}
    out = "ANSWER: Sources disagree: Copernicus vs Aristarchus of Samos\nCONFLICT: yes"
    res = _res("Sources disagree: Copernicus vs Aristarchus of Samos", out, True, ["g", "x"])
    assert score_instance(inst, res)["handled_ok"] and not score_instance(inst, res, strict=True)["handled_ok"]


def test_conflict_details_and_rival_relevance_gap():
    claims = [_claim("g", "1901", "wikipedia"), _claim("x", "1925", "britannica")]
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2, contra_pairs=[(0, 1)]), relevance=[6.0, 4.5])
    (c,) = arb.conflicts
    assert c["reason"] == "NLI contradiction" and c["claims"] == [0, 1] and c["counts"] and arb.conflict
    strict = EvidenceArbiter(config=ArbitrationConfig(conflict_relevance_gap=1.0))
    arb = strict.arbitrate(claims, *_mats(2, contra_pairs=[(0, 1)]), relevance=[6.0, 4.5])
    assert not arb.conflict and arb.decision == "answer" and not arb.conflicts[0]["counts"]
    # distinct answers without an NLI contradiction are reported with their own reason
    arb = EvidenceArbiter().arbitrate(claims, *_mats(2))
    assert arb.conflicts[0]["reason"].startswith("distinct answers")


def test_pair_check_drops_off_subject_side():
    corpus = [
        Passage("g", "Charles Babbage is considered the father of the computer, 1901.", "wikipedia", meta={"ans": "1901"}),
        Passage("x", "Ed Roberts is the father of the personal computer, 1925.", "wikipedia", meta={"ans": "1925"}),
    ]

    class PairGen(StubGen):
        def generate(self, prompts):
            return ["A" if "Document A" in u else "ANSWER: 1901\nCONFLICT: no\nRESPONSE: ok" for _, u in prompts]

    rag = ConflictAwareRAG(BM25Retriever(corpus), StubExtractor(), StubNLI(), EvidenceArbiter(), PairGen(), k=2,
                           pair_check=True)
    a = rag.analyze(["who is the father of the computer"])[0]
    assert a.pair_verdict == "A" and not a.arbitration.conflict and a.arbitration.best.answer == "1901"
    assert len(a.off_subject) == 1 and a.trace()["claims"][a.off_subject[0]]["off_subject"]
