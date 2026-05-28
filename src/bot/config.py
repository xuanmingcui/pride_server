"""Config loading and merging for the Discord bot.

Priority (highest → lowest):
  1. Environment variables  (PRIDE_MODEL_NAME, PRIDE_BACKEND, …)
  2. config.yaml            (committed defaults)
  3. Hard-coded fallbacks   (defined in this file)
"""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml


_DEFAULTS: Dict[str, Any] = {
    "model": {
        "name":                "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "backend":             "vllm",   # vllm | transformers
        "device":              "cuda",
        "dtype":               "auto",
        "max_new_tokens":      2048,
        "temperature":         0.8,
        "tensor_parallel_size": 1,
    },
    "whisper": {
        "model_size":    "large-v3",
        "device":        "cuda",
        "compute_type":  "float16",
        "language":      "en",
        "beam_size":     5,
        "vad_filter":    True,
    },
    "scenegraph": {
        "num_frames":             16,
        "normalize_pass":         True,
        "default_output":         "json",
        "temporal_target_fps":    0.25,
        "tokens_per_frame":       256,
        "prompt_overhead_tokens": 2048,
    },
    "validation": {
        "top_k":             5,
        "default_db":        "default",
        "num_frames":        8,
        "embed_model":       "Qwen/Qwen3-Embedding",
        "embed_backend":     "transformers",
        "embed_max_length":  4096,
        "embed_batch_size":  32,
    },
    "paths": {
        "db_dir":     "./data/db",
        "tmp_dir":    "./tmp",
        "output_dir": "./output",
    },
    "discord": {
        "max_upload_mb": 25,
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)

    if os.path.isfile(path):
        with open(path, "r") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, file_cfg)

    # Selective env-var overrides (add more as needed)
    env_map = {
        "PRIDE_MODEL_NAME":         ("model", "name"),
        "PRIDE_BACKEND":            ("model", "backend"),
        "PRIDE_DEVICE":             ("model", "device"),
        "PRIDE_MAX_NEW_TOKENS":     ("model", "max_new_tokens"),
        "PRIDE_TEMPERATURE":        ("model", "temperature"),
        "PRIDE_TENSOR_PARALLEL_SIZE": ("model", "tensor_parallel_size"),
        "PRIDE_WHISPER_MODEL":        ("whisper", "model_size"),
        "PRIDE_WHISPER_LANGUAGE":   ("whisper", "language"),
        "PRIDE_NUM_FRAMES":         ("scenegraph", "num_frames"),
        "PRIDE_DB_DIR":             ("paths", "db_dir"),
        "PRIDE_TMP_DIR":            ("paths", "tmp_dir"),
        "PRIDE_OUTPUT_DIR":         ("paths", "output_dir"),
    }
    for env_key, (section, field) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            # Coerce to int/float when the default is numeric
            default_val = _DEFAULTS.get(section, {}).get(field)
            if isinstance(default_val, int):
                val = int(val)
            elif isinstance(default_val, float):
                val = float(val)
            cfg[section][field] = val

    # Ensure output directories exist
    for key in ("db_dir", "tmp_dir", "output_dir"):
        p = cfg["paths"][key]
        os.makedirs(p, exist_ok=True)

    return cfg
