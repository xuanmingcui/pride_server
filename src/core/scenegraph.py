"""Unified scene-graph generation pipeline (Function 1).

Handles all input modality combinations:
  - text only
  - image (+ optional text context)
  - video (+ optional text context, + optional audio transcription)

The key efficiency improvement over generate_news_scenegraph.py is batching:
all video-segment requests are collected first and sent in ONE llm.generate()
call, letting vLLM fill GPU batches optimally.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional

log = logging.getLogger("pride.scenegraph")

import numpy as np
from PIL import Image

from .audio import transcribe_video, get_full_text
from .mllm import BaseMLLM
from .overlay import annotate_image_with_triplets_panel, annotate_video_with_triplets_panel
from .triplets import (
    Quintuple,
    Triplet,
    build_normalize_prompt,
    build_normalize_quintuples_prompt,
    build_scenegraph_prompt,
    build_srt_from_segments,
    build_text_only_scenegraph_prompt,
    build_video_segment_prompt,
    extract_quintuples_from_text,
    extract_triplets_from_text,
    shift_quintuples,
    validate_quintuples,
    validate_triplets,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


def _looks_complete_list(text: str) -> bool:
    """Heuristic check that a refinement-pass output is a complete Python list.

    Generation that hits `max_tokens` mid-row produces output whose last
    non-whitespace character is not `]`. We use that as the only signal —
    a legitimate refinement that drops every input row is still a closed
    `[]`, which passes.
    """
    if not text:
        return False
    stripped = text.rstrip()
    return stripped.endswith("]")


# ---------------------------------------------------------------------------
# Frame sampling helpers
# ---------------------------------------------------------------------------

def _frames_for_window(window_sec: float, fps: float,
                       min_frames: int, max_frames: int) -> int:
    """Frame budget for a window of `window_sec` seconds at target `fps`.

    Clamped to [min_frames, max_frames] so very short windows still produce
    enough visual context and very long windows don't blow past the model's
    context budget.
    """
    target = int(math.ceil(max(window_sec, 0.0) * max(fps, 1e-6)))
    return max(min_frames, min(max_frames, target if target > 0 else min_frames))


def sample_frames(video_path: str, fps: float = 1.0,
                  start_sec: float = 0.0, end_sec: float = -1.0,
                  min_frames: int = 4, max_frames: int = 256):
    """Sample frames uniformly from [start_sec, end_sec) at the requested fps.

    The actual frame count is `ceil(window_seconds * fps)`, floored to
    `min_frames` and capped at `max_frames` (context-budget guard). When the
    window is shorter than `min_frames / fps`, the last frame is duplicated
    to reach `min_frames`.

    Returns (frames: List[PIL.Image], sample_fps: float). `sample_fps` is the
    EFFECTIVE rate the model will see (n_frames / window_span); it differs
    from the requested `fps` when clamping kicks in.

    Returns (None, None) if the video cannot be opened or is empty.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = max(0, int(start_sec * orig_fps))
    f_end   = int(end_sec * orig_fps) if end_sec > 0 else max(total_frames - 1, 0)
    f_end   = min(f_end, total_frames - 1)
    if f_end < f_start:
        f_end = f_start

    window_sec = max(0.0, (f_end - f_start + 1) / orig_fps)
    n_frames = _frames_for_window(window_sec, fps, min_frames, max_frames)

    idxs = set(np.linspace(f_start, f_end, num=n_frames).astype(int).tolist())

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    frames: List[Image.Image] = []
    cur = f_start
    while cur <= f_end:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idxs:
            frames.append(Image.fromarray(frame[:, :, ::-1]))  # BGR → RGB
        cur += 1
        if len(frames) >= len(idxs):
            break
    cap.release()

    if not frames:
        return None, None
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    span = (f_end - f_start + 1) / orig_fps
    sample_fps = n_frames / span if span > 0 else float(fps)
    return frames, sample_fps


def _video_duration(video_path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SceneGraphPipeline:
    """Orchestrates audio extraction, frame sampling, batched MLLM inference,
    and optional overlay rendering.
    """

    def __init__(
        self,
        backend: BaseMLLM,
        whisper_config: Dict[str, Any],
        scenegraph_config: Dict[str, Any],
        tmp_dir: str = "/tmp",
    ):
        self.backend   = backend
        self.whisper   = whisper_config
        self.cfg       = scenegraph_config
        self.tmp_dir   = tmp_dir
        # Frame sampling is now FPS-driven. Per-call frame count =
        # ceil(window_seconds * fps), clamped to [min_frames, context cap].
        self.fps = float(scenegraph_config.get("fps", 1.0))
        self.min_frames = int(scenegraph_config.get("min_frames", 4))
        # Config default for whether the first-round generation auto-runs the
        # normalize pass. Most callers should leave this False — normalization
        # is exposed as a separate, user-triggered step (see normalize_segments).
        self.do_normalize = scenegraph_config.get("normalize_pass", False)
        # Per-call override set at the start of process(). Helpers consult this
        # rather than the static config flag.
        self._normalize_now = self.do_normalize

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        media_path: Optional[str] = None,
        text: str = "",
        output_type: str = "json",
        output_path: Optional[str] = None,
        temperature: Optional[float] = None,
        fps: Optional[float] = None,
        mode: str = "high",
        prompt_override: Optional[Dict[str, str]] = None,
        raw_output: bool = False,
        normalize: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate scene graph from any combination of media + text.

        Args:
            media_path:   Path to video or image; None for text-only.
            text:         Plain-text input or additional context.
            output_type:  "json" | "overlay".
            output_path:  Destination path for overlay output.
            temperature:  Per-request temperature override.
            fps:          Target frame sampling rate (frames per second). Per-call frame
                          count = ceil(window_seconds * fps), clamped by min_frames /
                          context-budget cap. Defaults to config `scenegraph.fps`.
            mode:         "high" (semantic/news) | "low" (visual/everyday).

        Returns dict:
            {
                "triplets":     List[Triplet],          # aggregated
                "segments":     List[{start, end, triplets}],
                "transcript":   str,
                "overlay_path": str | None,
            }
        """
        eff_fps = float(fps) if fps is not None else self.fps
        media_name = os.path.basename(media_path) if media_path else "<text-only>"
        # Per-call normalize override; helpers consult self._normalize_now.
        # Normalization is a separate user-triggered step by default, so only
        # auto-run when the caller explicitly asks for it (or, for legacy
        # callers that pass None, fall back to the config flag).
        self._normalize_now = self.do_normalize if normalize is None else bool(normalize)
        log.info("=== SceneGraph request | media=%s text_len=%d output=%s fps=%.3f mode=%s normalize=%s ===",
                 media_name, len(text), output_type, eff_fps, mode, self._normalize_now)

        if media_path is None:
            result = self._text_only(text, temperature, mode, prompt_override, raw_output)
        elif self._is_image(media_path):
            result = self._image(media_path, text, temperature, mode, prompt_override, raw_output)
        else:
            result = self._video(media_path, text, temperature, eff_fps, mode, prompt_override, raw_output)

        if output_type == "overlay" and output_path and media_path:
            log.info("Rendering overlay → %s …", os.path.basename(output_path))
            try:
                self._create_overlay(media_path, result, output_path)
                result["overlay_path"] = output_path
                log.info("Overlay written.")
            except Exception as e:
                log.warning("Overlay failed: %s", e)
                result["overlay_error"] = str(e)
        else:
            result["overlay_path"] = None

        segs = result.get("segments", [])
        total = sum(len(s.get("triplets", [])) for s in segs)
        log.info("=== Done | %d segment(s), %d total triplet(s) ===", len(segs), total)
        return result

    # ------------------------------------------------------------------
    # Modality-specific processors
    # ------------------------------------------------------------------

    def _text_only(self, text: str, temperature: Optional[float], mode: str,
                   prompt_override: Optional[Dict[str, str]] = None,
                   raw_output: bool = False) -> Dict[str, Any]:
        log.info("Mode: text-only. Running MLLM …")
        pair = build_text_only_scenegraph_prompt(text, mode=mode, template_override=prompt_override)
        req = [{"prompt": pair["user"], "system_prompt": pair["system"],
                "frames": None, "fps": None}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Text-only raw done (%d chars).", len(raw_text))
            return {"triplets": [], "segments": [], "transcript": text, "raw_text": raw_text}
        trips = self._run_batch(req, temperature, mode)[0]
        log.info("Text-only done: %d triplet(s).", len(trips))
        return {"triplets": trips, "segments": [], "transcript": text}

    def _image(
        self, image_path: str, text: str, temperature: Optional[float], mode: str,
        prompt_override: Optional[Dict[str, str]] = None, raw_output: bool = False,
    ) -> Dict[str, Any]:
        log.info("Mode: image. Running MLLM …")
        img = Image.open(image_path).convert("RGB")
        pair = build_scenegraph_prompt("", mode=mode, user_text=text, template_override=prompt_override)
        req = [{"prompt": pair["user"], "system_prompt": pair["system"],
                "frames": [img], "fps": None}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Image raw done (%d chars).", len(raw_text))
            return {"triplets": [], "segments": [], "transcript": text, "raw_text": raw_text}
        trips = self._run_batch(req, temperature, mode)[0]
        log.info("Image done: %d triplet(s).", len(trips))
        return {"triplets": trips, "segments": [], "transcript": text}

    def _video(
        self, video_path: str, text: str, temperature: Optional[float], fps: float, mode: str,
        prompt_override: Optional[Dict[str, str]] = None, raw_output: bool = False,
    ) -> Dict[str, Any]:
        duration = _video_duration(video_path)
        log.info("Video: %.1fs duration, user text: %d chars.", duration, len(text))
        user_text = text          # preserved verbatim for every prompt call
        transcript_text = ""
        asr_segments: List[Dict] = []

        try:
            result = transcribe_video(video_path, self.whisper, self.tmp_dir)
            asr_segments = result.get("segments", [])
            full_text = get_full_text(asr_segments)
            log.info("Transcript: %d ASR segment(s), %d chars.", len(asr_segments), len(full_text))
            transcript_text = full_text
        except Exception as e:
            log.warning("Audio transcription skipped: %s", e)

        if raw_output:
            # Skip segmentation entirely — one call over the whole video
            log.info("Raw output mode: bypassing segmentation, processing as single clip.")
            return self._video_whole(video_path, transcript_text, user_text, temperature, fps,
                                     mode, prompt_override, raw_output)
        if asr_segments:
            return self._video_by_segments(
                video_path, asr_segments, transcript_text, user_text, temperature, fps, mode,
                prompt_override,
            )
        log.info("No ASR segments → processing video as a single whole clip.")
        return self._video_whole(video_path, transcript_text, user_text, temperature, fps, mode,
                                 prompt_override)

    # ------------------------------------------------------------------
    # Adaptive temporal segmentation (no-ASR path)
    # ------------------------------------------------------------------

    def _max_frames_per_call(self) -> int:
        """Upper bound on how many frames fit in one MLLM call (context-budget cap)."""
        tok_per_frm = self.cfg.get("tokens_per_frame", 256)
        overhead    = self.cfg.get("prompt_overhead_tokens", 2048)
        max_len     = self.cfg.get("max_model_len", 32768)
        return max(8, (max_len - overhead) // tok_per_frm)

    def _max_segment_duration(self, fps: float) -> float:
        """Longest window (seconds) that can be sampled at `fps` without
        exceeding the model's per-call frame budget.

            max_frames_ctx = (max_model_len - prompt_overhead) // tokens_per_frame
            max_duration_ctx = max_frames_ctx / fps
        """
        return self._max_frames_per_call() / max(fps, 1e-6)

    def _video_whole(
        self, video_path: str, transcript_text: str, user_text: str,
        temperature: Optional[float], fps: float, mode: str,
        prompt_override: Optional[Dict[str, str]] = None, raw_output: bool = False,
    ) -> Dict[str, Any]:
        duration = _video_duration(video_path)
        max_seg_dur = self._max_segment_duration(fps)

        if duration > max_seg_dur and not raw_output:
            log.info(
                "Video duration %.1fs exceeds adaptive threshold %.1fs (fps=%.2f) — switching to temporal segmentation.",
                duration, max_seg_dur, fps,
            )
            return self._video_temporal(
                video_path, duration, transcript_text, user_text, temperature, fps, max_seg_dur, mode,
                prompt_override,
            )

        log.info("Mode: video (whole). Sampling at fps=%.2f …", fps)
        frames, eff_fps = sample_frames(
            video_path, fps,
            min_frames=self.min_frames, max_frames=self._max_frames_per_call(),
        )
        if not frames:
            log.warning("No frames sampled from %s.", video_path)
            return {"triplets": [], "segments": [], "transcript": transcript_text}
        log.info("Frames sampled (%d frames, effective_fps=%.2f, duration=%.1fs). Running MLLM …",
                 len(frames), eff_fps, duration)
        pair = build_video_segment_prompt(
            segment_duration_sec=duration,
            transcript_text=transcript_text, mode=mode, user_text=user_text,
            template_override=prompt_override,
        )
        req = [{"prompt": pair["user"], "system_prompt": pair["system"],
                "frames": frames, "fps": eff_fps}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Video (whole) raw done (%d chars).", len(raw_text))
            # Return flat (no segments) so the UI shows one plain text block
            return {"triplets": [], "segments": [], "transcript": transcript_text, "raw_text": raw_text}
        quints = self._run_batch_quintuples(req, temperature, mode, [duration])[0]
        # Single-segment whole-video path: quintuple times are already in absolute video time.
        log.info("Video (whole) done: %d quintuple(s).", len(quints))
        return {
            "triplets": quints,
            "segments": [{"start": 0.0, "end": duration, "triplets": quints}],
            "transcript": transcript_text,
        }

    def _video_temporal(
        self,
        video_path: str,
        duration: float,
        transcript_text: str,
        user_text: str,
        temperature: Optional[float],
        fps: float,
        seg_duration: float,
        mode: str,
        prompt_override: Optional[Dict[str, str]] = None,
        raw_output: bool = False,
    ) -> Dict[str, Any]:
        """Process a long video without ASR by splitting into equal temporal segments."""
        boundaries: List[tuple] = []
        t = 0.0
        while t < duration:
            boundaries.append((t, min(t + seg_duration, duration)))
            t += seg_duration

        log.info(
            "Temporal segmentation: %d segment(s) of up to %.1fs each (fps=%.2f).",
            len(boundaries), seg_duration, fps,
        )

        max_per_call = self._max_frames_per_call()
        requests: List[Dict] = []
        seg_meta: List[Dict] = []
        seg_durations: List[float] = []
        for start, end in boundaries:
            frames, eff_fps = sample_frames(
                video_path, fps, start, end,
                min_frames=self.min_frames, max_frames=max_per_call,
            )
            if not frames:
                continue
            seg_dur = max(0.0, end - start)
            pair = build_video_segment_prompt(
                segment_duration_sec=seg_dur,
                transcript_text=transcript_text, mode=mode, user_text=user_text,
                template_override=prompt_override,
            )
            requests.append({"prompt": pair["user"], "system_prompt": pair["system"],
                             "frames": frames, "fps": eff_fps})
            seg_meta.append({"start": start, "end": end})
            seg_durations.append(seg_dur)

        if not requests:
            log.warning("No usable temporal segments after frame sampling.")
            return {"triplets": [], "segments": [], "transcript": transcript_text}

        log.info("Running MLLM batch: %d temporal segment(s) …", len(requests))
        if raw_output:
            raw_texts = self._run_batch_raw(requests, temperature)
            out_segments = []
            for meta, raw_text in zip(seg_meta, raw_texts):
                meta["triplets"] = []
                meta["raw_text"] = raw_text
                out_segments.append(meta)
            log.info("Temporal segmentation raw done: %d segment(s).", len(out_segments))
            return {"triplets": [], "segments": out_segments, "transcript": transcript_text}

        batch_results = self._run_batch_quintuples(requests, temperature, mode, seg_durations)
        out_segments = []
        all_quints: List[Quintuple] = []
        for meta, seg_dur, quints in zip(seg_meta, seg_durations, batch_results):
            shifted = shift_quintuples(quints, offset_sec=meta["start"],
                                       segment_duration_sec=seg_dur)
            meta["triplets"] = shifted
            out_segments.append(meta)
            all_quints.extend(shifted)

        validated = validate_quintuples(all_quints)
        log.info(
            "Temporal segmentation done: %d segment(s), %d quintuple(s).",
            len(out_segments), len(validated),
        )
        return {"triplets": validated, "segments": out_segments, "transcript": transcript_text}

    def _video_by_segments(
        self,
        video_path: str,
        asr_segments: List[Dict],
        transcript_text: str,
        user_text: str,
        temperature: Optional[float],
        fps: float,
        mode: str,
        prompt_override: Optional[Dict[str, str]] = None,
        raw_output: bool = False,
    ) -> Dict[str, Any]:
        """Build one request per ASR segment, submit ALL in a single batch call."""
        requests: List[Dict] = []
        seg_meta: List[Dict] = []
        seg_durations: List[float] = []
        max_per_call = self._max_frames_per_call()

        log.info("Mode: video by segments (%d ASR segment(s), fps=%.2f). Sampling frames …",
                 len(asr_segments), fps)
        for seg in asr_segments:
            frames, eff_fps = sample_frames(
                video_path, fps, seg["start"], seg["end"],
                min_frames=self.min_frames, max_frames=max_per_call,
            )
            if not frames:
                continue
            seg_text = seg.get("text", "")
            seg_dur = max(0.0, seg["end"] - seg["start"])
            pair = build_video_segment_prompt(
                segment_duration_sec=seg_dur,
                transcript_text=seg_text, mode=mode, user_text=user_text,
                template_override=prompt_override,
            )
            requests.append({"prompt": pair["user"], "system_prompt": pair["system"],
                             "frames": frames, "fps": eff_fps})
            seg_meta.append({"start": seg["start"], "end": seg["end"]})
            seg_durations.append(seg_dur)

        if not requests:
            log.warning("No usable segments after frame sampling.")
            return {"triplets": [], "segments": [], "transcript": transcript_text}

        log.info("Running MLLM batch: %d request(s) …", len(requests))
        if raw_output:
            raw_texts = self._run_batch_raw(requests, temperature)
            out_segments = []
            for meta, raw_text in zip(seg_meta, raw_texts):
                meta["triplets"] = []
                meta["raw_text"] = raw_text
                out_segments.append(meta)
            log.info("Video by segments raw done: %d segment(s).", len(out_segments))
            return {"triplets": [], "segments": out_segments, "transcript": transcript_text}

        batch_results = self._run_batch_quintuples(requests, temperature, mode, seg_durations)
        out_segments = []
        all_quints: List[Quintuple] = []
        for meta, seg_dur, quints in zip(seg_meta, seg_durations, batch_results):
            shifted = shift_quintuples(quints, offset_sec=meta["start"],
                                       segment_duration_sec=seg_dur)
            meta["triplets"] = shifted
            out_segments.append(meta)
            all_quints.extend(shifted)

        validated = validate_quintuples(all_quints)
        log.info("Video by segments done: %d segment(s), %d total quintuple(s).",
                 len(out_segments), len(validated))
        return {"triplets": validated, "segments": out_segments, "transcript": transcript_text}

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_batch(
        self, requests: List[Dict], temperature: Optional[float], mode: str = "high"
    ) -> List[List[Triplet]]:
        """Run batched inference, optionally overriding temperature."""
        if temperature is not None:
            old_temp = getattr(self.backend, "temperature", None)
            if old_temp is not None:
                self.backend.temperature = temperature
            results = self.backend.generate_batch(requests)
            if old_temp is not None:
                self.backend.temperature = old_temp
        else:
            results = self.backend.generate_batch(requests)

        if self._normalize_now and mode == "high":
            log.info("Running normalization pass on %d result(s) …", len(results))
            results = self._normalize_batch(results)
            log.info("Normalization done.")
        return results

    def _run_batch_quintuples(
        self, requests: List[Dict], temperature: Optional[float],
        mode: str, seg_durations: List[float],
    ) -> List[List[Quintuple]]:
        """Run batched inference for video segments and parse 5-tuple outputs.

        Each segment's quintuples are clamped to [0, seg_duration] and stay in
        SEGMENT-RELATIVE time; the caller is responsible for shifting them into
        absolute video time.
        """
        raw_texts = self._run_batch_raw(requests, temperature)
        results: List[List[Quintuple]] = []
        for raw_text, seg_dur in zip(raw_texts, seg_durations):
            quints = extract_quintuples_from_text(raw_text)
            # Fallback: if the model dropped to 3-tuple (no times) output,
            # recover those rows and stamp the whole clip's [0, seg_dur] window.
            if not quints:
                trips = extract_triplets_from_text(raw_text)
                quints = [(s, r, o, 0.0, seg_dur) for s, r, o in trips]
            # Clamp segment-relative times to [0, seg_dur].
            clamped: List[Quintuple] = []
            for s, r, o, t0, t1 in quints:
                t0 = max(0.0, min(t0, seg_dur))
                t1 = max(t0, min(t1, seg_dur))
                clamped.append((s, r, o, t0, t1))
            results.append(validate_quintuples(clamped))

        if self._normalize_now and mode == "high":
            log.info("Running quintuple normalization pass on %d result(s) …", len(results))
            results = self._normalize_quintuples_batch(results)
            log.info("Quintuple normalization done.")
        return results

    def _run_batch_raw(
        self, requests: List[Dict], temperature: Optional[float]
    ) -> List[str]:
        """Run batched inference and return raw model text (no parsing)."""
        if temperature is not None:
            old_temp = getattr(self.backend, "temperature", None)
            if old_temp is not None:
                self.backend.temperature = temperature
            results = self.backend.generate_batch_raw(requests)
            if old_temp is not None:
                self.backend.temperature = old_temp
        else:
            results = self.backend.generate_batch_raw(requests)
        return results

    # ------------------------------------------------------------------
    # Standalone normalize entry point (user-triggered, post-generation)
    # ------------------------------------------------------------------

    def normalize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize entity names across already-generated segments.

        Accepts segments in the API-JSON shape — each segment is
        ``{"start", "end", "triplets": [item, ...]}`` where item is either a
        triplet dict ``{subject, relation, object}`` or a quintuple dict
        ``{subject, relation, object, start_sec, end_sec}``. The returned
        segments preserve the same shape and the original timestamps; only
        the subject / relation / object strings are touched.
        """
        if not segments:
            return []

        # Detect whether any segment carries quintuples; quintuple shape wins
        # if any item has start_sec, otherwise fall back to triplet shape.
        is_quintuple = any(
            (trip := (seg.get("triplets") or [None])[0])
            and isinstance(trip, dict) and "start_sec" in trip
            for seg in segments
        )

        if is_quintuple:
            all_quints: List[List[Quintuple]] = []
            for seg in segments:
                quints: List[Quintuple] = []
                for t in seg.get("triplets", []):
                    try:
                        quints.append((
                            str(t["subject"]), str(t["relation"]), str(t["object"]),
                            float(t.get("start_sec", 0.0)), float(t.get("end_sec", 0.0)),
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
                all_quints.append(quints)
            log.info("Standalone normalize: %d segment(s), %d quintuple(s) total.",
                     len(all_quints), sum(len(q) for q in all_quints))
            normalized = self._normalize_quintuples_batch(all_quints)
            return [
                {
                    "start": seg.get("start"),
                    "end":   seg.get("end"),
                    "triplets": [
                        {"subject": s, "relation": r, "object": o,
                         "start_sec": float(t0), "end_sec": float(t1)}
                        for s, r, o, t0, t1 in quints
                    ],
                }
                for seg, quints in zip(segments, normalized)
            ]

        # Triplet path
        all_trips: List[List[Triplet]] = []
        for seg in segments:
            trips: List[Triplet] = []
            for t in seg.get("triplets", []):
                try:
                    trips.append((str(t["subject"]), str(t["relation"]), str(t["object"])))
                except (KeyError, TypeError):
                    continue
            all_trips.append(trips)
        log.info("Standalone normalize: %d segment(s), %d triplet(s) total.",
                 len(all_trips), sum(len(t) for t in all_trips))
        normalized = self._normalize_batch(all_trips)
        return [
            {
                "start": seg.get("start"),
                "end":   seg.get("end"),
                "triplets": [
                    {"subject": s, "relation": r, "object": o}
                    for s, r, o in trips
                ],
            }
            for seg, trips in zip(segments, normalized)
        ]

    def _normalize_quintuples_batch(
        self, all_quints: List[List[Quintuple]]
    ) -> List[List[Quintuple]]:
        """Refine quintuple lists (entity normalization + dedup + quality filter).

        Time fields on surviving rows are preserved by the prompt; the only
        local safety check is structural — if the model's output is not a
        properly-terminated list, we assume the generation got cut off and
        keep the pre-refinement result for that segment rather than silently
        dropping data.
        """
        indices = [i for i, q in enumerate(all_quints) if q]
        if not indices:
            return all_quints
        pairs = [build_normalize_quintuples_prompt(all_quints[i]) for i in indices]
        try:
            texts = self.backend.generate_text_batch(
                [p["user"] for p in pairs],
                [p["system"] for p in pairs],
            )
        except Exception:
            return all_quints
        out = list(all_quints)
        for i, text in zip(indices, texts):
            try:
                if not _looks_complete_list(text):
                    log.warning(
                        "Refinement output for segment %d is not a closed list — "
                        "assuming truncation, keeping pre-refinement result.", i,
                    )
                    continue
                normalized = validate_quintuples(extract_quintuples_from_text(text))
                if not normalized:
                    continue  # keep original
                out[i] = normalized
            except Exception:
                pass
        return out

    def _normalize_batch(self, all_trips: List[List[Triplet]]) -> List[List[Triplet]]:
        """Refine triplet lists (entity normalization + dedup + quality filter).

        Same truncation guard as ``_normalize_quintuples_batch``: outputs that
        are not a properly-closed Python list are rejected in favor of the
        pre-refinement input.
        """
        indices = [i for i, t in enumerate(all_trips) if t]
        if not indices:
            return all_trips
        pairs = [build_normalize_prompt(all_trips[i]) for i in indices]
        try:
            texts = self.backend.generate_text_batch(
                [p["user"] for p in pairs],
                [p["system"] for p in pairs],
            )
        except Exception:
            return all_trips
        out = list(all_trips)
        for i, text in zip(indices, texts):
            try:
                if not _looks_complete_list(text):
                    log.warning(
                        "Refinement output for segment %d is not a closed list — "
                        "assuming truncation, keeping pre-refinement result.", i,
                    )
                    continue
                normalized = validate_triplets(extract_triplets_from_text(text))
                out[i] = normalized if normalized else all_trips[i]
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _is_image(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in _IMAGE_EXTS

    def _create_overlay(
        self, media_path: str, result: Dict[str, Any], output_path: str
    ) -> None:
        if self._is_image(media_path):
            self._overlay_image(media_path, result["triplets"], output_path)
        else:
            self._overlay_video(media_path, result, output_path)

    def _overlay_image(
        self, image_path: str, trips: List[Triplet], output_path: str
    ) -> None:
        annotate_image_with_triplets_panel(image_path, trips, output_path)

    def _overlay_video(
        self, video_path: str, result: Dict[str, Any], output_path: str
    ) -> None:
        segments = result.get("segments", [])
        if not segments or (len(segments) == 1 and segments[0].get("end") == -1.0):
            segments = [{"start": 0.0, "end": 359999.0, "triplets": result.get("triplets", [])}]

        srt_content = build_srt_from_segments(segments)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".srt", delete=False, dir=self.tmp_dir
        ) as fh:
            fh.write(srt_content)
            srt_tmp = fh.name

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            panel_position = "bottom" if vid_h > vid_w else "side"

            annotate_video_with_triplets_panel(
                video_path=video_path,
                merged_srt_path=srt_tmp,
                out_path=output_path,
                panel_position=panel_position,
            )
        finally:
            try:
                os.remove(srt_tmp)
            except OSError:
                pass
