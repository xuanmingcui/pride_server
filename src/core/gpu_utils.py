"""GPU assignment policy for multi-GPU setups.

Policy
------
Main MLLM (vLLM or Transformers backend)
    Always occupies GPUs 0 .. tensor_parallel_size-1.
    vLLM does this automatically; TransformersVLBackend is explicitly pinned via device_map.

Whisper (audio transcription)
    First GPU not used by the MLLM (i.e., GPU tensor_parallel_size).
    Falls back to GPU 0 (shared with MLLM) when all GPUs are occupied.

Embedding model (sentence-transformers / ChromaDB)
    Second GPU not used by the MLLM (GPU tensor_parallel_size + 1).
    Falls back to CPU when no spare GPU is available — the model is small and
    CPU inference is acceptable for embedding workloads.

All device strings follow the PyTorch / faster-whisper convention:
    "cuda:0", "cuda:1", "cpu"
"""
from __future__ import annotations

import logging
from typing import Dict

log = logging.getLogger("pride.gpu")


def detect_gpu_count() -> int:
    """Return the number of CUDA GPUs visible to the current process."""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def assign_gpus(model_config: Dict) -> Dict[str, str]:
    """Compute device-string assignments for Whisper and the embedding model.

    The main MLLM device is *not* returned here — it is handled by the
    backend constructors (vLLM naturally starts at GPU 0; TransformersVLBackend
    is pinned explicitly).

    Args:
        model_config: The ``model`` section of the application config dict.

    Returns:
        {
            "mllm":    "cuda:0",           # always GPU 0 for Transformers backend
            "whisper": "cuda:1" | "cuda:0" | "cpu",
            "embed":   "cuda:2" | "cuda:1" | "cpu",
        }
    """
    n_gpus = detect_gpu_count()
    tp     = model_config.get("tensor_parallel_size", 1)

    if n_gpus == 0:
        log.info("GPU assignment: no CUDA GPUs detected — all components run on CPU.")
        return {"mllm": "cpu", "whisper": "cpu", "embed": "cpu"}

    # GPUs 0..min(tp,n_gpus)-1 are occupied by the main MLLM.
    mllm_end   = min(tp, n_gpus)
    mllm_gpus  = set(range(mllm_end))
    free_gpus  = [i for i in range(n_gpus) if i not in mllm_gpus]

    log.info(
        "GPU assignment: %d GPU(s) detected. Main MLLM → GPU(s) %s.",
        n_gpus,
        list(mllm_gpus) if len(mllm_gpus) > 1 else 0,
    )

    # Whisper
    if free_gpus:
        whisper_device = f"cuda:{free_gpus[0]}"
        log.info("GPU assignment: Whisper → %s.", whisper_device)
    else:
        whisper_device = "cuda:0"
        log.warning(
            "GPU assignment: no free GPU for Whisper — it will share GPU 0 with the MLLM "
            "(tensor_parallel_size=%d, total GPUs=%d). "
            "Consider reducing tensor_parallel_size or using more GPUs.",
            tp, n_gpus,
        )

    # Embedding — share Whisper's GPU when no second free GPU exists
    if len(free_gpus) >= 2:
        embed_device = f"cuda:{free_gpus[1]}"
        log.info("GPU assignment: embedding model → %s.", embed_device)
    else:
        embed_device = whisper_device   # share with Whisper (both are small vs. the MLLM)
        log.info("GPU assignment: embedding model → %s (shared with Whisper).", embed_device)

    return {"mllm": "cuda:0", "whisper": whisper_device, "embed": embed_device}


def whisper_compute_type(device: str, preferred: str = "float16") -> str:
    """Return an appropriate faster-whisper compute_type for the given device.

    GPU supports float16/int8_float16; CPU only supports int8/int8_float32/float32.
    """
    if device.startswith("cuda"):
        return preferred          # honour config (float16 by default on GPU)
    return "int8"                 # safest CPU option for faster-whisper
