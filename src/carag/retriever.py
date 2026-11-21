"""Retrievers: BM25 (pure python), dense (BGE via transformers) and a hybrid (reciprocal rank fusion)."""
from __future__ import annotations

import hashlib
import heapq
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm.auto import tqdm

from .devices import pick_device, pick_dtype
from .schema import Passage, Retrieved
from .text import normalize_answer

_STOP = set(
    "a an the of in on at to for is are was were be been by with and or as from that this which who whom "
    "whose what when where how why did does do it its into than then there their they he she his her".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize_answer(text)) if t not in _STOP]


class Retriever(Protocol):
    passages: list[Passage]

    def search_batch(self, queries: list[str], k: int) -> list[list[Retrieved]]: ...


class BM25Retriever:
    """Okapi BM25 with numpy postings. With `cache_dir` the index is pickled and reloaded on the next start."""

    def __init__(self, passages: list[Passage], k1: float = 0.9, b: float = 0.4, cache_dir: str | Path | None = None):
        self.passages, self.k1, self.b = passages, k1, b
        cache = None
        if cache_dir:
            sig = hashlib.sha1("\n".join(p.pid for p in passages).encode()).hexdigest()[:12]
            cache = Path(cache_dir) / f"bm25-{sig}.pkl"
        if cache and cache.exists():
            self.postings, self.idf, self.norm = pickle.loads(cache.read_bytes())
            return
        raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len = np.zeros(len(passages), dtype=np.float32)
        for i, p in enumerate(passages):
            toks = tokenize(f"{p.title} {p.text}")
            doc_len[i] = len(toks)
            for t, c in Counter(toks).items():
                raw[t].append((i, c))
        n = len(passages)
        avgdl = float(doc_len.mean()) if n else 1.0
        self.norm = (k1 * (1 - b + b * doc_len / avgdl)).astype(np.float32)  # per-document length normalizer
        self.postings = {t: (np.fromiter((i for i, _ in ps), np.int32, len(ps)),
                             np.fromiter((c for _, c in ps), np.float32, len(ps))) for t, ps in raw.items()}
        self.idf = {t: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5)) for t, ps in raw.items()}
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((self.postings, self.idf, self.norm), protocol=pickle.HIGHEST_PROTOCOL))

    def search(self, query: str, k: int) -> list[Retrieved]:
        scores = np.zeros(len(self.passages), dtype=np.float32)
        for t in set(tokenize(query)):
            if t in self.postings:
                ids, tf = self.postings[t]
                scores[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.norm[ids])
        k = min(k, int((scores > 0).sum()))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Retrieved(self.passages[int(i)], float(scores[i]), r) for r, i in enumerate(top)]

    def search_batch(self, queries: list[str], k: int) -> list[list[Retrieved]]:
        return [self.search(q, k) for q in queries]


class DenseRetriever:
    """Bi-encoder retrieval with a BGE-style model (CLS pooling, query instruction).

    Passage embeddings are cached under `cache_dir` (keyed by model and corpus), so a large corpus
    is only embedded once."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, passages: list[Passage], model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None, batch_size: int = 128, max_length: int = 512,
                 cache_dir: str | Path | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.passages = passages
        self.device = pick_device(device)
        self.batch_size, self.max_length = batch_size, max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=pick_dtype(self.device, "encoder")).to(self.device).eval()

        cache = None
        if cache_dir:
            sig = hashlib.sha1("\n".join(p.pid for p in passages).encode()).hexdigest()[:12]
            cache = Path(cache_dir) / f"emb-{model_name.replace('/', '--')}-{sig}.npy"
        if cache and cache.exists():
            self.emb = torch.from_numpy(np.load(cache)).float().to(self.device)
        else:
            self.emb = self._encode([f"{p.title} {p.text}".strip() for p in passages], progress=len(passages) > 5000)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.emb.cpu().numpy().astype(np.float16))

    def _encode(self, texts: list[str], progress: bool = False):
        torch = self.torch
        out = []
        # Sort by length so batches pad less.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for s in tqdm(range(0, len(texts), self.batch_size), desc="embedding passages", disable=not progress):
                idx = order[s : s + self.batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append((idx, torch.nn.functional.normalize(cls.float(), dim=-1)))
        emb = torch.empty(len(texts), out[0][1].shape[1], device=self.device)
        for idx, e in out:
            emb[torch.tensor(idx, device=self.device)] = e
        return emb

    def search_batch(self, queries: list[str], k: int) -> list[list[Retrieved]]:
        q = self._encode([self.QUERY_PREFIX + x for x in queries])
        sims = q @ self.emb.T
        vals, idxs = sims.topk(min(k, len(self.passages)), dim=-1)
        return [
            [Retrieved(self.passages[i], float(v), r) for r, (v, i) in enumerate(zip(vrow.tolist(), irow.tolist()))]
            for vrow, irow in zip(vals, idxs)
        ]


class HybridRetriever:
    """Reciprocal rank fusion of BM25 and dense rankings."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, depth: int = 50, rrf_k: int = 60):
        self.bm25, self.dense, self.depth, self.rrf_k = bm25, dense, depth, rrf_k
        self.passages = bm25.passages

    def search_batch(self, queries: list[str], k: int) -> list[list[Retrieved]]:
        lex = self.bm25.search_batch(queries, self.depth)
        den = self.dense.search_batch(queries, self.depth)
        results = []
        for a, b in zip(lex, den):
            fused: dict[str, float] = defaultdict(float)
            by_pid: dict[str, Passage] = {}
            for ranking in (a, b):
                for r in ranking:
                    fused[r.passage.pid] += 1.0 / (self.rrf_k + r.rank + 1)
                    by_pid[r.passage.pid] = r.passage
            top = heapq.nlargest(k, fused.items(), key=lambda kv: kv[1])
            results.append([Retrieved(by_pid[pid], s, i) for i, (pid, s) in enumerate(top)])
        return results


class RerankRetriever:
    """Second stage: a small cross-encoder re-scores the base retriever's top `depth` candidates.

    Returned scores are the cross-encoder logits; the arbiter also uses them as a relevance signal."""

    def __init__(self, base: Retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None, depth: int = 30, max_length: int = 384):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.base, self.depth, self.max_length = torch, base, depth, max_length
        self.passages = base.passages
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()

    def search_batch(self, queries: list[str], k: int) -> list[list[Retrieved]]:
        torch = self.torch
        results = []
        for q, cands in zip(queries, self.base.search_batch(queries, max(self.depth, k))):
            if not cands:
                results.append([])
                continue
            enc = self.tok([q] * len(cands), [f"{c.passage.title} {c.passage.text}".strip() for c in cands],
                           padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                scores = self.model(**enc).logits.view(-1).float().cpu().tolist()
            order = sorted(range(len(cands)), key=lambda i: -scores[i])[:k]
            results.append([Retrieved(cands[i].passage, scores[i], r) for r, i in enumerate(order)])
        return results


def build_retriever(kind: str, passages: list[Passage], dense_model: str = "BAAI/bge-small-en-v1.5",
                    device: str | None = None, cache_dir: str | Path | None = None,
                    reranker: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Retriever:
    if kind == "bm25":
        base = BM25Retriever(passages, cache_dir=cache_dir)
    elif kind == "dense":
        base = DenseRetriever(passages, dense_model, device, cache_dir=cache_dir)
    elif kind == "hybrid":
        base = HybridRetriever(BM25Retriever(passages, cache_dir=cache_dir),
                               DenseRetriever(passages, dense_model, device, cache_dir=cache_dir))
    else:
        raise ValueError(f"unknown retriever {kind!r} (bm25|dense|hybrid)")
    return RerankRetriever(base, reranker, device) if reranker else base
