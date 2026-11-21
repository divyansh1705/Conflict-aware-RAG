"""Answer normalization, matching and sentence utilities."""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | {"–", "—", "‘", "’", "“", "”"}


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization, plus accent folding (Röntgen == Rontgen)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = "".join(" " if ch in _PUNCT else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def contains(text: str, needle: str) -> bool:
    """Word-boundary containment after normalization."""
    n = normalize_answer(needle)
    if not n:
        return False
    return f" {n} " in f" {normalize_answer(text)} "


def contains_any(text: str, answers: list[str]) -> bool:
    return any(contains(text, a) for a in answers)


def same_answer(a: str, b: str, f1_threshold: float = 0.6) -> bool:
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    if na == nb or contains(a, b) or contains(b, a):
        return True
    return token_f1(a, b) >= f1_threshold


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_BOUNDARY.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def sentence_containing(text: str, char_start: int, char_end: int, max_words: int = 60) -> str:
    """Sentence that contains [char_start, char_end), clipped to a word window around the span."""
    for s, e in sentence_spans(text):
        if s <= char_start < e or (s < char_end <= e):
            sent_s, sent_e = s, max(e, char_end)
            break
    else:
        sent_s, sent_e = 0, len(text)
    sent = text[sent_s:sent_e]
    words = sent.split()
    if len(words) <= max_words:
        return sent.strip()
    # Clip to a window centred on the answer span.
    rel = len(text[sent_s:char_start].split())
    lo = max(0, rel - max_words // 2)
    return " ".join(words[lo : lo + max_words])
