"""Device / dtype selection so the same code runs on a CUDA workstation, a 4 GB laptop GPU, Apple Silicon or CPU."""
from __future__ import annotations

SMALL_GPU_BYTES = 6 * 1024**3


def pick_device(pref: str | None = None) -> str:
    import torch

    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_small_gpu(device: str) -> bool:
    import torch

    return device.startswith("cuda") and torch.cuda.get_device_properties(device).total_memory < SMALL_GPU_BYTES


def pick_aux_device(main: str, pref: str | None = None) -> str:
    """Where the small helper models (NLI, embedder, QA) go. On a <6 GB GPU the LLM needs all the VRAM."""
    if pref:
        return pref
    return "cpu" if is_small_gpu(main) else main


def pick_dtype(device: str, kind: str = "llm"):
    """kind: 'llm' (causal LM), 'encoder' (BERT-like), 'deberta' (unstable in fp16)."""
    import torch

    if device.startswith("cuda"):
        if kind == "encoder":
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float32 if kind == "deberta" else torch.float16)
    if device == "mps":
        return torch.float16 if kind == "llm" else torch.float32
    return torch.float32


def is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def free_gpu_bytes(device: str) -> int | None:
    import torch

    if not device.startswith("cuda"):
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return free


OOM_HINT = ("The GPU ran out of memory. Another program (often a second `carag` process) is probably using it: "
            "check with `nvidia-smi`, stop the other process, or use a smaller model with "
            "`--profile laptop --quantize 4bit`.")
