"""Multimodal RAG-based validation pipeline (Function 2).

Flow
----
1. Audio extraction + transcript (if video input).
2. Multimodal embedding of the full input (text + transcript + visual frames)
   via MultimodalEmbedder → the query vector encodes the frames directly (shared
   image/text space), so image- and video-only inputs retrieve relevant facts
   without needing a transcript. Retrieve the most relevant facts from the DB.
3. Feed the original input (frames + transcript + text) together with the
   retrieved facts to the MLLM in a single call.
4. The MLLM writes a natural-language fact-check report — no structured JSON,
   no per-claim parsing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from ..core.mllm import BaseMLLM
from ..core.audio import transcribe_video, get_full_text
from ..core.scenegraph import sample_frames, _video_duration, _IMAGE_EXTS
from .database import FactDatabase

log = logging.getLogger("pride.validation")

_DEFAULT_FPS        = 1.0
_DEFAULT_MIN_FRAMES = 4
_DEFAULT_TOP_K      = 5
_DEFAULT_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    database:        str
    report:          str            # natural-language MLLM report
    retrieved_facts: List[str]      # facts used as context
    transcript:      str  = ""
    num_facts_found: int  = 0

    def format_discord(self, max_chars: int = 1900) -> str:
        header = f"**Fact-Check Report** — database: `{self.database}`"
        if self.num_facts_found:
            header += f" · {self.num_facts_found} fact(s) retrieved"
        text = f"{header}\n\n{self.report}"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…*(truncated)*"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database":        self.database,
            "num_facts_found": self.num_facts_found,
            "retrieved_facts": self.retrieved_facts,
            "transcript":      self.transcript,
            "report":          self.report,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_validation_prompt(
    facts: List[str],
    database: str,
    transcript: str,
    text: str,
    has_visual: bool,
    template_override: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the {system, user} pair for the validation MLLM call."""
    from ..core.triplets import _render_pair

    facts_block = (
        "\n".join(f"  {i+1}. {f}" for i, f in enumerate(facts))
        if facts else "  (no relevant facts found in the database)"
    )

    input_parts = []
    if has_visual:
        input_parts.append("the video frames / image shown above")
    if transcript:
        input_parts.append("the audio transcript")
    if text:
        input_parts.append("the submitted text")
    input_desc = " and ".join(input_parts) if input_parts else "the submitted content"

    ctx_parts = []
    if transcript:
        ctx_parts.append(f"TRANSCRIPT:\n{transcript}")
    if text:
        ctx_parts.append(f"SUBMITTED TEXT:\n{text}")
    context_block = ("\n\n" + "\n\n".join(ctx_parts)) if ctx_parts else ""

    return _render_pair(
        "validation",
        template_override,
        {
            "database":      database,
            "facts_block":   facts_block,
            "input_desc":    input_desc,
            "context_block": context_block,
        },
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """End-to-end multimodal fact-checking pipeline.

    Args:
        backend:    MLLM backend (same instance used for scene-graph generation).
        db:         FactDatabase backed by MultimodalEmbedder.
        whisper_config: Whisper transcription config (forwarded from app config).
        top_k:      Number of facts to retrieve.
        fps:        Target frame sampling rate (frames per second) for video input.
                    Per-call frame count = ceil(window_seconds * fps), clamped to
                    [min_frames, context cap].
        min_frames: Lower bound on frames sampled from a video (default 4).
        tmp_dir:    Temp directory for intermediate audio files.
    """

    def __init__(
        self,
        backend: BaseMLLM,
        db: FactDatabase,
        whisper_config: Dict[str, Any],
        top_k: int = _DEFAULT_TOP_K,
        fps: float = _DEFAULT_FPS,
        min_frames: int = _DEFAULT_MIN_FRAMES,
        tmp_dir: str = "/tmp",
    ):
        self.backend   = backend
        self.db        = db
        self.whisper   = whisper_config
        self.top_k     = top_k
        self.fps       = float(fps)
        self.min_frames = int(min_frames)
        self.tmp_dir   = tmp_dir

    def validate(
        self,
        database: str,
        media_path: Optional[str] = None,
        text: str = "",
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        prompt_override: Optional[Dict[str, str]] = None,
    ) -> ValidationReport:
        k = top_k or self.top_k

        # ── 1. Extract transcript and sample frames ───────────────────────────
        transcript = ""
        frames: List[Image.Image] = []
        is_image = media_path is not None and _is_image(media_path)

        if media_path:
            if is_image:
                try:
                    frames = [Image.open(media_path).convert("RGB")]
                except Exception as e:
                    log.warning("Could not open image %s: %s", media_path, e)
            else:
                # Video — sample frames at the configured fps
                duration = _video_duration(media_path)
                sampled, _eff_fps = sample_frames(
                    media_path, self.fps, min_frames=self.min_frames,
                )
                frames = sampled or []
                # Attempt audio transcription
                try:
                    asr = transcribe_video(media_path, self.whisper, self.tmp_dir)
                    transcript = get_full_text(asr.get("segments", []))
                    log.info("Transcript: %d chars for validation.", len(transcript))
                except Exception as e:
                    log.warning("Transcription skipped during validation: %s", e)

        # ── 2. Build query text and retrieve facts ────────────────────────────
        query_text = " ".join(filter(None, [transcript, text])).strip()
        log.info("Retrieving top-%d facts from '%s' …", k, database)
        hits = self.db.search_by_embedding(
            database,
            self.db._embedder.embed_query(query_text, frames if frames else None),
            top_k=k,
        )
        retrieved = [h["fact"] for h in hits]
        log.info("Retrieved %d fact(s).", len(retrieved))

        # ── 3. Build prompt and call MLLM ─────────────────────────────────────
        has_visual = bool(frames)
        pair = _build_validation_prompt(
            facts             = retrieved,
            database          = database,
            transcript        = transcript,
            text              = text,
            has_visual        = has_visual,
            template_override = prompt_override or None,
        )

        log.info("Running MLLM validation (frames=%d, max_tokens=%d) …",
                 len(frames), _DEFAULT_MAX_TOKENS)
        report_text = self.backend.generate_raw(
            user_prompt   = pair["user"],
            system_prompt = pair["system"],
            frames        = frames if frames else None,
            fps           = 1.0 if frames else None,
            max_tokens    = _DEFAULT_MAX_TOKENS,
        )
        log.info("Validation report generated (%d chars).", len(report_text))

        return ValidationReport(
            database        = database,
            report          = report_text.strip(),
            retrieved_facts = retrieved,
            transcript      = transcript,
            num_facts_found = len(retrieved),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS
