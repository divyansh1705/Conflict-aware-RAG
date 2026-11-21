"""NLI cross-encoder wrapper returning [entailment, neutral, contradiction] probabilities."""
from __future__ import annotations

import numpy as np

from .devices import pick_device, pick_dtype


class NLIModel:
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 device: str | None = None, batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = pick_dtype(self.device, "deberta")  # DeBERTa-v3 is unstable in fp16; bf16 is fine
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.batch_size, self.max_length = batch_size, max_length
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}

        def find(key: str) -> int:
            hits = [i for i, l in labels.items() if l.startswith(key)]
            if len(hits) != 1:
                raise ValueError(f"cannot map NLI label {key!r} in {labels}")
            return hits[0]

        self.order = [find("entail"), find("neutral"), find("contra")]

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(premise, hypothesis)] -> array (n, 3): P(entail), P(neutral), P(contradiction)."""
        torch = self.torch
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        if not pairs:
            return out
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                               truncation="only_first", max_length=self.max_length, return_tensors="pt").to(self.device)
                probs = self.model(**enc).logits.float().softmax(-1)[:, self.order].cpu().numpy()
                out[idx] = probs
        return out
