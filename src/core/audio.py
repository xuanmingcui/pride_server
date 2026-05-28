"""Audio extraction and Whisper transcription utilities.

Two transcription backends are supported and tried in order:

1. faster-whisper  (CTranslate2-based)
   Fast, memory-efficient, supports VAD filtering and beam search.
   Limitation: CTranslate2 has hardcoded CUDA library soname dependencies
   (e.g. libcublas.so.12 for CUDA 12) that break on CUDA 13+.

2. HuggingFace transformers Whisper pipeline  (PyTorch-based)
   Used automatically when faster-whisper fails to initialise on GPU.
   PyTorch resolves CUDA symbols at runtime, so it works regardless of the
   installed CUDA minor version — the same path used by the MLLM backend.

Fallback chain on GPU failure:
    faster-whisper (GPU) → transformers Whisper (GPU) → faster-whisper (CPU, int8)

Both backends produce the same output format:
    {"segments": [{"start", "end", "text"}, ...], "language": str, "language_probability": float}
"""
from __future__ import annotations

import logging
import os
import subprocess
import shutil
from typing import Any, Dict, List, Optional

log = logging.getLogger("pride.audio")

# ── Singleton caches ──────────────────────────────────────────────────────────
_fw_model = None                        # faster-whisper WhisperModel
_fw_model_key: Optional[tuple] = None

_hf_pipe = None                         # transformers ASR pipeline
_hf_pipe_key: Optional[tuple] = None


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install with: brew install ffmpeg  or  sudo apt-get install ffmpeg"
        )


def extract_wav(mp4_path: str, wav_path: str, sr: int = 16_000) -> None:
    check_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le",
        wav_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{p.stderr}")


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------

def _parse_device(device: str):
    """Split "cuda:N" into ("cuda", N). Pass "cuda"/"cpu" through unchanged."""
    if ":" in device:
        dev, idx = device.split(":", 1)
        return dev, int(idx) if idx.isdigit() else 0
    return device, 0


def _load_faster_whisper(model_size: str, device: str, compute_type: str):
    """Load a faster-whisper WhisperModel. Raises on any failure."""
    from faster_whisper import WhisperModel
    dev, device_index = _parse_device(device)
    return WhisperModel(model_size, device=dev, device_index=device_index,
                        compute_type=compute_type)


def _load_hf_whisper(model_size: str, device: str):
    """Load a HuggingFace transformers ASR pipeline for Whisper.

    Uses PyTorch under the hood, so it is compatible with any CUDA version
    that PyTorch supports — unlike CTranslate2 which requires exact sonames.
    """
    import torch
    from transformers import pipeline as hf_pipeline
    model_name = f"openai/whisper-{model_size}"
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    log.info("Loading transformers Whisper (%s) on %s …", model_name, device)
    return hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,      # segment long audio into 30 s windows
        stride_length_s=5,      # overlap prevents boundary artefacts
        return_timestamps=True,
    )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def _segments_from_hf(result: Dict[str, Any], language: str) -> Dict[str, Any]:
    """Convert a transformers pipeline result into the faster-whisper segment format."""
    segments: List[Dict[str, Any]] = []
    for chunk in result.get("chunks", []):
        ts = chunk.get("timestamp") or (None, None)
        t0 = float(ts[0]) if ts[0] is not None else 0.0
        t1 = float(ts[1]) if ts[1] is not None else t0 + 0.1
        text = chunk.get("text", "").strip()
        if text:
            segments.append({"start": t0, "end": t1, "text": text})
    return {"segments": segments, "language": language or "en", "language_probability": 1.0}


def transcribe_wav(wav_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe a WAV file. Returns dict with 'segments' and 'language'.

    Tries backends in order: faster-whisper → transformers (GPU fallback) → CPU.
    """
    global _fw_model, _fw_model_key, _hf_pipe, _hf_pipe_key

    model_size   = config.get("model_size",   "large-v3")
    device       = config.get("device",       "cuda")
    compute_type = config.get("compute_type", "float16")
    language     = config.get("language", None)
    if language in (None, "null", ""):
        language = None
    beam_size    = config.get("beam_size",    5)
    vad_filter   = config.get("vad_filter",   True)

    is_gpu = device.startswith("cuda")

    # ── 1. faster-whisper ────────────────────────────────────────────────────
    fw_key = (model_size, device, compute_type)
    if _fw_model_key != fw_key:
        _fw_model = None        # invalidate stale cache
    if _fw_model is None:
        try:
            _fw_model = _load_faster_whisper(model_size, device, compute_type)
            _fw_model_key = fw_key
            log.info("faster-whisper loaded on %s.", device)
        except Exception as exc:
            log.warning(
                "faster-whisper could not load on %s: %s",
                device, exc,
            )

    if _fw_model is not None:
        try:
            segments_iter, info = _fw_model.transcribe(
                wav_path, language=language, task="transcribe",
                beam_size=beam_size, vad_filter=vad_filter,
            )
            segments = [
                {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
                for s in segments_iter
            ]
            return {
                "segments":             segments,
                "language":             info.language,
                "language_probability": info.language_probability,
            }
        except Exception as exc:
            log.warning("faster-whisper transcription failed: %s", exc)
            _fw_model = None    # don't reuse a broken model

    # ── 2. transformers Whisper on GPU (CUDA-version-agnostic via PyTorch) ───
    if is_gpu:
        hf_key = (model_size, device)
        if _hf_pipe_key != hf_key:
            _hf_pipe = None
        if _hf_pipe is None:
            try:
                _hf_pipe = _load_hf_whisper(model_size, device)
                _hf_pipe_key = hf_key
                log.info("transformers Whisper loaded on %s (CUDA fallback).", device)
            except Exception as exc:
                log.warning(
                    "transformers Whisper also failed on %s: %s. "
                    "Will fall back to CPU.",
                    device, exc,
                )

        if _hf_pipe is not None:
            try:
                generate_kwargs = {"language": language} if language else {}
                result = _hf_pipe(wav_path, generate_kwargs=generate_kwargs,
                                  return_timestamps=True)
                return _segments_from_hf(result, language)
            except Exception as exc:
                log.warning("transformers Whisper transcription failed: %s", exc)
                _hf_pipe = None

    # ── 3. CPU fallback via faster-whisper int8 ──────────────────────────────
    log.warning(
        "All GPU Whisper backends failed — transcribing on CPU (int8). "
        "This will be slower."
    )
    cpu_key = (model_size, "cpu", "int8")
    if _fw_model_key != cpu_key:
        _fw_model = _load_faster_whisper(model_size, "cpu", "int8")
        _fw_model_key = cpu_key
    segments_iter, info = _fw_model.transcribe(
        wav_path, language=language, task="transcribe",
        beam_size=beam_size, vad_filter=vad_filter,
    )
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments_iter
    ]
    return {
        "segments":             segments,
        "language":             info.language,
        "language_probability": info.language_probability,
    }


def transcribe_video(mp4_path: str, config: Dict[str, Any], tmp_dir: str = "/tmp") -> Dict[str, Any]:
    """Extract audio from video and transcribe. Cleans up WAV file afterward."""
    stem     = os.path.splitext(os.path.basename(mp4_path))[0]
    wav_path = os.path.join(tmp_dir, f"{stem}_{os.getpid()}.wav")
    log.info("Extracting audio from %s …", os.path.basename(mp4_path))
    extract_wav(mp4_path, wav_path)
    log.info("Audio extracted → %s (%.1f MB). Starting Whisper transcription …",
             os.path.basename(wav_path), os.path.getsize(wav_path) / 1_048_576)
    try:
        result = transcribe_wav(wav_path, config)
        log.info("Transcription done: %d segment(s), language=%s (p=%.2f)",
                 len(result.get("segments", [])),
                 result.get("language", "?"),
                 result.get("language_probability", 0.0))
        return result
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def get_full_text(segments: List[Dict[str, Any]]) -> str:
    return " ".join(s["text"] for s in segments).strip()
