"""Local web UI: `carag serve --data data/nq_conflict` then open http://localhost:7860

Standard-library HTTP server; the page is a single static file (web/index.html).

  GET  /api/info                -> profile, corpus size, reliability table
  GET  /api/examples            -> random benchmark questions, by scenario
  GET  /api/search?q=...        -> passages only (no LLM; instant)
  POST /api/ask {question}      -> answer + full evidence trace
  POST /api/rearbitrate {id, reliability, margin}
       -> re-run arbitration on the cached claims/NLI of a previous question with different trust
          settings (no model call, so it is instant)
"""
from __future__ import annotations

import json
import random
import threading
from collections import OrderedDict
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .arbitration import EvidenceArbiter
from .prompts import parse_output, render_arbitration
from .sources import SourceReliability
from .text import normalize_answer

WEB = Path(__file__).parent / "web" / "index.html"


class App:
    def __init__(self, rag, instances: list[dict], profile: str):
        self.rag, self.profile = rag, profile
        self.by_question = {normalize_answer(i["question"]): i for i in instances}
        self.instances = instances
        self.traces: OrderedDict[str, object] = OrderedDict()
        self.lock = threading.Lock()  # one model call at a time
        self.counter = 0

    # -- serialisation ---------------------------------------------------------------------------
    def _arbitration(self, arb, claims) -> dict:
        return {
            "decision": arb.decision, "conflict": arb.conflict, "margin": round(arb.margin, 3), "rival": arb.rival,
            "groups": [{"answer": g.answer, "members": g.members, "support": round(g.support, 3),
                        "sources": g.sources} for g in arb.groups],
            "conflicts": [{**c, "score": round(c["score"], 3), "nli_contradiction": round(c["nli_contradiction"], 3),
                           "compatibility": round(c["compatibility"], 3)} for c in arb.conflicts],
        }

    @staticmethod
    def _matrix(a) -> dict:
        """Passage x claim NLI grid: P(passage j entails / contradicts claim i) for answer-bearing claims."""
        cols = [i for i, c in enumerate(a.claims) if c.has_answer or i in a.off_subject]
        cell = lambda m, j, i: None if m[j][i] != m[j][i] else round(float(m[j][i]), 2)
        return {"claims": cols, "labels": [a.claims[i].answer for i in cols],
                "entail": [[cell(a.entail, j, i) for i in cols] for j in range(len(a.claims))],
                "contra": [[cell(a.contra, j, i) for i in cols] for j in range(len(a.claims))]}

    def _payload(self, tid: str, question: str, a, output: str, arbiter: EvidenceArbiter) -> dict:
        parsed = parse_output(output)
        passages = []
        for n, (r, c) in enumerate(zip(a.retrieved, a.claims)):
            v = a.entail[n][n]
            passages.append({
                "source": r.passage.source, "reliability": arbiter.reliability(r.passage.source),
                "text": r.passage.text, "claim": c.answer if (c.has_answer or n in a.off_subject) else None,
                "verified": None if v != v else round(float(v), 3), "relevance": round(float(r.score), 2),
                "off_subject": n in a.off_subject,
            })
        inst = self.by_question.get(normalize_answer(question))
        bench = inst and {"scenario": inst["scenario"], "gold": inst["answers"][:3], "planted": inst["substitute"]
                          if inst["conflict_pids"] else None}
        return {"id": tid, "question": question, "output": output, "answer": parsed.answer,
                "response": parsed.response, "flagged": parsed.flagged_conflict, "passages": passages,
                "arbitration": self._arbitration(a.arbitration, a.claims), "benchmark": bench,
                "matrix": self._matrix(a), "pair_verdict": a.pair_verdict}

    # -- endpoints -------------------------------------------------------------------------------
    def info(self) -> dict:
        rel = self.rag.arbiter.reliability
        return {"profile": self.profile, "passages": len(self.rag.retriever.passages),
                "generator": self.rag.generator.name if self.rag.generator else None,
                "reliability": rel.table, "default_reliability": rel.default,
                "margin": self.rag.arbiter.cfg.margin_threshold, "benchmark_questions": len(self.instances)}

    def examples(self, per_scenario: int = 2) -> list[dict]:
        by: dict[str, list[dict]] = {}
        for i in self.instances:
            by.setdefault(i["scenario"], []).append(i)
        return [{"question": i["question"], "scenario": s}
                for s, items in by.items() for i in random.sample(items, min(per_scenario, len(items)))]

    def search(self, q: str, k: int = 8) -> list[dict]:
        with self.lock:
            hits = self.rag.retriever.search_batch([q], k)[0]
        return [{"source": h.passage.source, "text": h.passage.text, "relevance": round(float(h.score), 2)} for h in hits]

    def ask(self, question: str) -> dict:
        with self.lock:
            output, a = self.rag.ask(question)
            self.counter += 1
            tid = str(self.counter)
            self.traces[tid] = (question, a)
            while len(self.traces) > 50:
                self.traces.popitem(last=False)
        return self._payload(tid, question, a, output, self.rag.arbiter)

    def rearbitrate(self, tid: str, reliability: dict[str, float], margin: float | None) -> dict:
        if tid not in self.traces:
            raise KeyError("unknown question id (ask again)")
        question, a = self.traces[tid]
        base = self.rag.arbiter
        cfg = replace(base.cfg, margin_threshold=base.cfg.margin_threshold if margin is None else float(margin))
        table = {**base.reliability.table, **{k: min(max(float(v), 0.0), 1.0) for k, v in reliability.items()}}
        arbiter = EvidenceArbiter(SourceReliability(table, base.reliability.default), cfg)
        relevance = [r.score for r in a.retrieved] if a.retrieved and self.rag.uses_reranker else None
        arb = arbiter.arbitrate(a.claims, a.entail, a.contra, relevance)
        b = replace(a, arbitration=arb)
        return self._payload(tid, question, b, render_arbitration(arb, a.claims), arbiter)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, WEB.read_bytes(), "text/html")
                elif url.path == "/api/info":
                    self._json(app.info())
                elif url.path == "/api/examples":
                    self._json(app.examples())
                elif url.path == "/api/search":
                    q = parse_qs(url.query).get("q", [""])[0].strip()
                    self._json(app.search(q) if q else [])
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # surface errors to the page instead of dropping the connection
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/ask":
                    q = str(body.get("question", "")).strip()
                    self._json(app.ask(q) if q else {"error": "empty question"}, 200 if q else 400)
                elif self.path == "/api/rearbitrate":
                    self._json(app.rearbitrate(str(body["id"]), body.get("reliability", {}), body.get("margin")))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                from .devices import OOM_HINT, is_oom

                if is_oom(e):
                    import torch

                    torch.cuda.empty_cache()
                    self._json({"error": OOM_HINT}, 503)
                else:
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def serve(rag, instances: list[dict], profile: str, host: str = "127.0.0.1", port: int = 7860) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(App(rag, instances, profile)))
    print(f"[carag] web UI ready: http://{host}:{port}   (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
