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
    Triplet,
    build_scenegraph_prompt,
    build_text_only_scenegraph_prompt,
    build_temporal_scenegraph_prompt,
    build_normalize_prompt,
    build_srt_from_segments,
    extract_triplets_from_text,
    validate_triplets,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Frame sampling helpers
# ---------------------------------------------------------------------------

def sample_frames(video_path: str, num_frames: int = 16,
                  start_sec: float = 0.0, end_sec: float = -1.0):
    """Sample `num_frames` uniformly from [start_sec, end_sec) of video_path.

    Returns (frames: List[PIL.Image], sample_fps: float).
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

    idxs = set(np.linspace(f_start, f_end, num=num_frames).astype(int).tolist())

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
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    span = (f_end - f_start + 1) / orig_fps
    sample_fps = num_frames / span if span > 0 else 1.0
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
        self.num_frames  = scenegraph_config.get("num_frames", 16)
        self.do_normalize = scenegraph_config.get("normalize_pass", True)

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
        num_frames: Optional[int] = None,
        mode: str = "high",
        prompt_override: Optional[str] = None,
        raw_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate scene graph from any combination of media + text.

        Args:
            media_path:   Path to video or image; None for text-only.
            text:         Plain-text input or additional context.
            output_type:  "json" | "overlay".
            output_path:  Destination path for overlay output.
            temperature:  Per-request temperature override.
            num_frames:   Frames per segment override.
            mode:         "high" (semantic/news) | "low" (visual/everyday).

        Returns dict:
            {
                "triplets":     List[Triplet],          # aggregated
                "segments":     List[{start, end, triplets}],
                "transcript":   str,
                "overlay_path": str | None,
            }
        """
        n_frames = num_frames or self.num_frames
        media_name = os.path.basename(media_path) if media_path else "<text-only>"
        log.info("=== SceneGraph request | media=%s text_len=%d output=%s frames=%d mode=%s ===",
                 media_name, len(text), output_type, n_frames, mode)

        if media_path is None:
            result = self._text_only(text, temperature, mode, prompt_override, raw_output)
        elif self._is_image(media_path):
            result = self._image(media_path, text, temperature, mode, prompt_override, raw_output)
        else:
            result = self._video(media_path, text, temperature, n_frames, mode, prompt_override, raw_output)

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
                   prompt_override: Optional[str] = None, raw_output: bool = False) -> Dict[str, Any]:
        log.info("Mode: text-only. Running MLLM …")
        prompt = build_text_only_scenegraph_prompt(text, mode=mode, template_override=prompt_override)
        req = [{"prompt": prompt, "frames": None, "fps": None}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Text-only raw done (%d chars).", len(raw_text))
            return {"triplets": [], "segments": [], "transcript": text, "raw_text": raw_text}
        trips = self._run_batch(req, temperature, mode)[0]
        log.info("Text-only done: %d triplet(s).", len(trips))
        return {"triplets": trips, "segments": [], "transcript": text}

    def _image(
        self, image_path: str, text: str, temperature: Optional[float], mode: str,
        prompt_override: Optional[str] = None, raw_output: bool = False,
    ) -> Dict[str, Any]:
        log.info("Mode: image. Running MLLM …")
        img = Image.open(image_path).convert("RGB")
        prompt = build_scenegraph_prompt("", mode=mode, user_text=text, template_override=prompt_override)
        req = [{"prompt": prompt, "frames": [img], "fps": None}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Image raw done (%d chars).", len(raw_text))
            return {"triplets": [], "segments": [], "transcript": text, "raw_text": raw_text}
        trips = self._run_batch(req, temperature, mode)[0]
        log.info("Image done: %d triplet(s).", len(trips))
        return {"triplets": trips, "segments": [], "transcript": text}

    def _video(
        self, video_path: str, text: str, temperature: Optional[float], num_frames: int, mode: str,
        prompt_override: Optional[str] = None, raw_output: bool = False,
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
            return self._video_whole(video_path, transcript_text, user_text, temperature, num_frames,
                                     mode, prompt_override, raw_output)
        if asr_segments:
            return self._video_by_segments(
                video_path, asr_segments, transcript_text, user_text, temperature, num_frames, mode,
                prompt_override,
            )
        log.info("No ASR segments → processing video as a single whole clip.")
        return self._video_whole(video_path, transcript_text, user_text, temperature, num_frames, mode,
                                 prompt_override)

    # ------------------------------------------------------------------
    # Adaptive temporal segmentation (no-ASR path)
    # ------------------------------------------------------------------

    def _max_segment_duration(self, num_frames: int) -> float:
        """Compute maximum video duration (seconds) that can be adequately covered
        by `num_frames` sampled frames, subject to model context-window capacity.

        Two constraints are applied — the tighter one wins:

        1. Temporal coverage: at `temporal_target_fps` we need
           `duration * temporal_target_fps <= num_frames` frames.
           → max_duration_temporal = num_frames / temporal_target_fps

        2. Context budget: each frame costs ~tokens_per_frame tokens; the
           model has max_model_len tokens total (minus prompt overhead).
           → max_duration_ctx = max_frames_ctx / temporal_target_fps
        """
        target_fps  = self.cfg.get("temporal_target_fps", 0.25)
        tok_per_frm = self.cfg.get("tokens_per_frame", 256)
        overhead    = self.cfg.get("prompt_overhead_tokens", 2048)
        max_len     = self.cfg.get("max_model_len", 32768)

        max_dur_temporal = num_frames / target_fps
        max_frames_ctx   = max(8, (max_len - overhead) // tok_per_frm)
        max_dur_ctx      = max_frames_ctx / target_fps
        return min(max_dur_temporal, max_dur_ctx)

    def _video_whole(
        self, video_path: str, transcript_text: str, user_text: str,
        temperature: Optional[float], num_frames: int, mode: str,
        prompt_override: Optional[str] = None, raw_output: bool = False,
    ) -> Dict[str, Any]:
        duration = _video_duration(video_path)
        max_seg_dur = self._max_segment_duration(num_frames)

        if duration > max_seg_dur and not raw_output:
            log.info(
                "Video duration %.1fs exceeds adaptive threshold %.1fs — switching to temporal segmentation.",
                duration, max_seg_dur,
            )
            return self._video_temporal(
                video_path, duration, transcript_text, user_text, temperature, num_frames, max_seg_dur, mode,
                prompt_override,
            )

        log.info("Mode: video (whole). Sampling %d frames …", num_frames)
        frames, fps = sample_frames(video_path, num_frames)
        if not frames:
            log.warning("No frames sampled from %s.", video_path)
            return {"triplets": [], "segments": [], "transcript": transcript_text}
        log.info("Frames sampled (fps=%.2f, duration=%.1fs). Running MLLM …", fps, duration)
        prompt = build_scenegraph_prompt(transcript_text, mode=mode, user_text=user_text,
                                         template_override=prompt_override)
        req = [{"prompt": prompt, "frames": frames, "fps": fps}]
        if raw_output:
            raw_text = self._run_batch_raw(req, temperature)[0]
            log.info("Video (whole) raw done (%d chars).", len(raw_text))
            # Return flat (no segments) so the UI shows one plain text block
            return {"triplets": [], "segments": [], "transcript": transcript_text, "raw_text": raw_text}
        trips = self._run_batch(req, temperature, mode)[0]
        log.info("Video (whole) done: %d triplet(s).", len(trips))
        return {
            "triplets": trips,
            "segments": [{"start": 0.0, "end": duration, "triplets": trips}],
            "transcript": transcript_text,
        }

    def _video_temporal(
        self,
        video_path: str,
        duration: float,
        transcript_text: str,
        user_text: str,
        temperature: Optional[float],
        num_frames: int,
        seg_duration: float,
        mode: str,
        prompt_override: Optional[str] = None,
        raw_output: bool = False,
    ) -> Dict[str, Any]:
        """Process a long video without ASR by splitting into equal temporal segments."""
        boundaries: List[tuple] = []
        t = 0.0
        while t < duration:
            boundaries.append((t, min(t + seg_duration, duration)))
            t += seg_duration

        log.info(
            "Temporal segmentation: %d segment(s) of up to %.1fs each.",
            len(boundaries), seg_duration,
        )

        requests: List[Dict] = []
        seg_meta: List[Dict] = []
        for start, end in boundaries:
            frames, fps = sample_frames(video_path, num_frames, start, end)
            if not frames:
                continue
            prompt = build_temporal_scenegraph_prompt(
                start, end, transcript_text, mode=mode, user_text=user_text,
                template_override=prompt_override,
            )
            requests.append({"prompt": prompt, "frames": frames, "fps": fps})
            seg_meta.append({"start": start, "end": end})

        if not requests:
            log.warning("No usable temporal segments after frame sampling.")
            return {"triplets": [], "quintuples": [], "segments": [], "transcript": transcript_text}

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

        batch_results = self._run_batch(requests, temperature, mode)
        out_segments = []
        all_trips: List[Triplet] = []
        for meta, trips in zip(seg_meta, batch_results):
            meta["triplets"] = trips
            out_segments.append(meta)
            all_trips.extend(trips)

        validated = validate_triplets(all_trips)
        log.info(
            "Temporal segmentation done: %d segment(s), %d triplet(s).",
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
        num_frames: int,
        mode: str,
        prompt_override: Optional[str] = None,
        raw_output: bool = False,
    ) -> Dict[str, Any]:
        """Build one request per ASR segment, submit ALL in a single batch call."""
        requests: List[Dict] = []
        seg_meta: List[Dict] = []

        log.info("Mode: video by segments (%d ASR segment(s)). Sampling frames …",
                 len(asr_segments))
        for seg in asr_segments:
            frames, fps = sample_frames(
                video_path, num_frames, seg["start"], seg["end"]
            )
            if not frames:
                continue
            seg_text = seg.get("text", "")
            prompt = build_scenegraph_prompt(seg_text, mode=mode, user_text=user_text,
                                             template_override=prompt_override)
            requests.append({"prompt": prompt, "frames": frames, "fps": fps})
            seg_meta.append({"start": seg["start"], "end": seg["end"]})

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

        batch_results = self._run_batch(requests, temperature, mode)
        out_segments = []
        all_trips: List[Triplet] = []
        for meta, trips in zip(seg_meta, batch_results):
            meta["triplets"] = trips
            out_segments.append(meta)
            all_trips.extend(trips)

        validated = validate_triplets(all_trips)
        log.info("Video by segments done: %d segment(s), %d total triplet(s).",
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

        if self.do_normalize and mode == "high":
            log.info("Running normalization pass on %d result(s) …", len(results))
            results = self._normalize_batch(results)
            log.info("Normalization done.")
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

    def _normalize_batch(self, all_trips: List[List[Triplet]]) -> List[List[Triplet]]:
        """Normalize all non-empty triplet lists in one batched model call."""
        indices = [i for i, t in enumerate(all_trips) if t]
        if not indices:
            return all_trips
        prompts = [build_normalize_prompt(all_trips[i]) for i in indices]
        try:
            texts = self.backend.generate_text_batch(prompts)
        except Exception:
            return all_trips
        out = list(all_trips)
        for i, text in zip(indices, texts):
            try:
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
