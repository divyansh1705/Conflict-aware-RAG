"""LLM generators. All take a list of (system, user) prompts and return completions in order."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from tqdm.auto import tqdm

from .devices import is_oom, is_small_gpu, pick_device, pick_dtype


class Generator(Protocol):
    name: str

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]: ...


class HFGenerator:
    """Local instruction-tuned model via transformers (greedy decoding, left-padded batches)."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str | None = None,
                 batch_size: int = 32, max_new_tokens: int = 160, token_budget: int | None = None,
                 quantize: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"hf:{model_name}"
        self.device = pick_device(device)
        if token_budget is None:  # padded tokens per batch; small on laptops, OOM halving is the safety net
            token_budget = 12000 if self.device.startswith("cuda") and not is_small_gpu(self.device) else 2500
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = pick_dtype(self.device, "llm")
        if quantize == "4bit":  # bitsandbytes NF4; CUDA only. ~0.8 GB per billion parameters.
            from transformers import BitsAndBytesConfig

            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=qc,
                                                              device_map={"": self.device}).eval()
            self.name += "[4bit]"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()
        self.batch_size, self.max_new_tokens, self.token_budget = batch_size, max_new_tokens, token_budget

    def _batches(self, lengths: list[int]) -> list[list[int]]:
        """Length-sorted batches capped by batch_size and by padded tokens (batch * longest)."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches, cur = [], []
        for i in order:
            longest = lengths[i] + self.max_new_tokens  # sorted ascending: i is the longest so far
            if cur and (len(cur) + 1 > self.batch_size or (len(cur) + 1) * longest > self.token_budget):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        return batches

    def _run(self, texts: list[str]) -> list[str]:
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        try:
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id)
        except RuntimeError as e:  # torch.OutOfMemoryError (CUDA) and MPS OOM are RuntimeErrors
            if not is_oom(e) or len(texts) == 1:
                raise
            del enc
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            h = len(texts) // 2
            return self._run(texts[:h]) + self._run(texts[h:])
        new = gen[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True)]

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        texts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for s, u in prompts
        ]
        lengths = [len(x) for x in self.tok(texts)["input_ids"]]
        out: list[str] = [""] * len(texts)
        for idx in tqdm(self._batches(lengths), desc=f"generate[{self.name}]", leave=False):
            for i, t in zip(idx, self._run([texts[i] for i in idx])):
                out[i] = t
        return out


class ClaudeGenerator:
    """Claude via the Anthropic Python SDK. Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4096,
                 concurrency: int = 8):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.concurrency = model, effort, max_tokens, concurrency
        self.name = f"claude:{model}"

    def _one(self, prompt: tuple[str, str]) -> str:
        system, user = prompt
        try:
            # Server-side fallback re-runs a safety-declined request on a fallback model in the same call.
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (self.anthropic.BadRequestError, self.anthropic.AuthenticationError,
                self.anthropic.PermissionDeniedError, self.anthropic.NotFoundError):
            raise  # non-retryable: configuration problem, surface it
        except self.anthropic.APIError as e:  # rate limits / 5xx / connection after SDK retries
            return f"ANSWER: unknown\nRESPONSE: [generation failed: {type(e).__name__}]"
        if resp.stop_reason == "refusal":
            return "ANSWER: unknown\nRESPONSE: [declined]"
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(tqdm(ex.map(self._one, prompts), total=len(prompts), desc=f"generate[{self.name}]", leave=False))


def make_generator(spec: str, device: str | None = None, quantize: str = "auto") -> Generator:
    """'hf:<model id>' or 'claude:<model id>'. quantize: auto | 4bit | none (HF models on CUDA only).

    auto = 4-bit when the GPU has less than 10 GB, so a 3B model fits a 4 GB laptop card."""
    kind, _, model = spec.partition(":")
    if kind == "hf":
        device = pick_device(device)
        if quantize == "auto":
            import torch

            small = device.startswith("cuda") and torch.cuda.get_device_properties(device).total_memory < 10 * 1024**3
            quantize = "4bit" if small else "none"
        return HFGenerator(model or "Qwen/Qwen2.5-3B-Instruct", device=device,
                           quantize="4bit" if quantize == "4bit" else None)
    if kind == "claude":
        return ClaudeGenerator(model or "claude-opus-5-5")
    raise ValueError(f"unknown generator spec {spec!r}; use hf:<model> or claude:<model>")
