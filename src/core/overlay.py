"""Overlay scene-graph triplets onto video or image frames.

Refactored from overlay_triplets_to_video.py into a proper module.
The main entry point is `annotate_video_with_triplets_panel()`.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

Triplet = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

import ast
import re

_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")


def _time_to_ms(t: str) -> int:
    m = _TIME_RE.fullmatch(t.strip())
    if not m:
        raise ValueError(f"Bad SRT time: {t!r}")
    h, mi, s, ms = int(m["h"]), int(m["m"]), int(m["s"]), int(m["ms"])
    return (((h * 60 + mi) * 60) + s) * 1000 + ms


@dataclass
class SRTBlock:
    index: int
    start_ms: int
    end_ms: int
    triplets: List[Triplet]


def parse_srt_triplets(path: str) -> List[SRTBlock]:
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()
    blocks: List[SRTBlock] = []
    for raw in re.split(r"\n\s*\n", content):
        lines = [ln.rstrip("\n") for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        idx = int(lines[0].strip())
        if "-->" not in lines[1]:
            raise ValueError(f"Bad time line in block {idx}: {lines[1]!r}")
        a, b = [x.strip() for x in lines[1].split("-->")]
        text = "\n".join(lines[2:]).strip()
        try:
            obj = ast.literal_eval(text)
        except Exception as e:
            raise ValueError(f"Cannot parse triplets in block {idx}: {e}\n{text}")
        if not isinstance(obj, list):
            raise ValueError(f"Triplet payload must be a list in block {idx}")
        trips: List[Triplet] = []
        for x in obj:
            if isinstance(x, (tuple, list)) and len(x) == 3 and all(isinstance(k, str) for k in x):
                s, r, o = x[0].strip(), x[1].strip(), x[2].strip()
                if s and r and o:
                    trips.append((s, r, o))
        blocks.append(SRTBlock(index=idx, start_ms=_time_to_ms(a), end_ms=_time_to_ms(b), triplets=trips))
    blocks.sort(key=lambda bl: bl.start_ms)
    return blocks


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")


def _encode_h264(src: str, out_path: str, audio_src: Optional[str] = None) -> None:
    cmd = ["ffmpeg", "-y", "-i", src]
    if audio_src:
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", out_path]
    _run(cmd)


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_metrics(scale: float) -> Tuple[int, int]:
    """Return (char_width, line_height) for _FONT at the given scale.

    Uses 'Xg' for height so ascenders and descenders are both measured.
    """
    (cw, _), _ = cv2.getTextSize("X", _FONT, scale, 1)
    (_, ch), base = cv2.getTextSize("Xg", _FONT, scale, 1)
    return cw, ch + base + 5   # +5px inter-line gap


def _wrap(text: str, max_chars: int) -> List[str]:
    words = text.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    final: List[str] = []
    for ln in lines:
        for i in range(0, max(1, len(ln)), max_chars):
            final.append(ln[i:i + max_chars])
    return final


def _triplets_to_lines(triplets: List[Triplet], max_triplets: int, max_chars: int) -> List[str]:
    lines: List[str] = []
    for s, r, o in triplets[:max_triplets]:
        for ln in _wrap(f"({s}, {r}, {o})", max_chars):
            lines.append(ln)
    return lines


def _paginate(lines: List[str], max_lines: int, ms_now: int, seg_start: int, seg_end: int):
    if len(lines) <= max_lines:
        return lines, f"Total triplets: {len(lines)}"
    span = max(seg_end - seg_start, 1)
    frac = (ms_now - seg_start) / span
    num_pages = max(1, -(-len(lines) // max_lines))   # ceil div
    page = max(0, min(num_pages - 1, int(frac * num_pages)))
    return lines[page * max_lines:(page + 1) * max_lines], f"Page {page+1}/{num_pages}"


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _side_panel(
    panel_w: int, frame_h: int,
    triplets: List[Triplet], ms_now: int, seg_start: int, seg_end: int,
    max_triplets: int,
) -> np.ndarray:
    panel = np.full((frame_h, panel_w, 3), (18, 18, 18), dtype=np.uint8)
    x0 = 12
    usable_w = panel_w - x0 - 8
    # Scale font so ~40 chars fit per line; clamp to [0.35, 0.60]
    scale = max(0.35, min(0.60, usable_w / (40 * 15)))
    cw, lh = _text_metrics(scale)
    max_chars = min(55, max(20, usable_w // max(1, cw)))
    footer_reserve = lh + 6
    # y is the text baseline; first line starts one lh from the top
    y = lh
    max_lines = max(1, (frame_h - y - footer_reserve) // lh)
    lines, footer = _paginate(
        _triplets_to_lines(triplets, max_triplets, max_chars),
        max_lines, ms_now, seg_start, seg_end,
    )
    for ln in lines:
        cv2.putText(panel, ln, (x0, y), _FONT, scale, (230, 230, 230), 1, cv2.LINE_AA)
        y += lh
    footer_scale = max(0.30, scale * 0.75)
    cv2.putText(panel, footer, (x0, frame_h - 5), _FONT, footer_scale, (160, 160, 160), 1, cv2.LINE_AA)
    return panel


def _bottom_panel(
    frame_w: int, panel_h: int,
    triplets: List[Triplet], ms_now: int, seg_start: int, seg_end: int,
    max_triplets: int,
) -> np.ndarray:
    panel = np.full((panel_h, frame_w, 3), (18, 18, 18), dtype=np.uint8)
    x0, y0 = 12, 18
    scale = 0.50
    cw, lh = _text_metrics(scale)
    chars_per_col = 40
    col_w = chars_per_col * cw + 20
    num_cols = max(1, (frame_w - x0) // col_w)
    footer_reserve = lh + 6
    max_rows = max(1, (panel_h - y0 - footer_reserve) // lh)
    lines, footer = _paginate(
        _triplets_to_lines(triplets, max_triplets, chars_per_col),
        max_rows * num_cols, ms_now, seg_start, seg_end,
    )
    for i, ln in enumerate(lines):
        col = i // max_rows
        row = i % max_rows
        if col >= num_cols:
            break
        x = x0 + col * col_w
        y = y0 + row * lh
        cv2.putText(panel, ln, (x, y), _FONT, scale, (230, 230, 230), 1, cv2.LINE_AA)
    footer_scale = max(0.30, scale * 0.75)
    cv2.putText(panel, footer, (x0, panel_h - 5), _FONT, footer_scale, (160, 160, 160), 1, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Segment lookup
# ---------------------------------------------------------------------------

def _find_active(blocks: List[SRTBlock], ms_now: int, hint: int) -> Tuple[Optional[SRTBlock], int]:
    if not blocks:
        return None, hint
    i = max(0, min(hint, len(blocks) - 1))
    while i < len(blocks) and blocks[i].end_ms <= ms_now:
        i += 1
    if i < len(blocks) and blocks[i].start_ms <= ms_now < blocks[i].end_ms:
        return blocks[i], i
    j = min(i, len(blocks) - 1)
    while j >= 0 and blocks[j].start_ms > ms_now:
        j -= 1
    if j >= 0 and blocks[j].start_ms <= ms_now < blocks[j].end_ms:
        return blocks[j], j
    return None, i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_image_with_triplets_panel(
    image_path: str,
    triplets: List[Triplet],
    out_path: str,
    max_triplets: int = 60,
    panel_ratio: float = 0.4,
) -> None:
    """Render a triplet panel beside a static image and save to `out_path`.

    Portrait images (h > w) get a bottom panel; landscape images get a side panel.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    if h > w:
        _cw, _lh = _text_metrics(0.50)
        _col_w = 40 * _cw + 20
        _num_cols = max(1, (w - 12) // _col_w)
        _est_lines = max(1, int(len(triplets[:max_triplets]) * 1.4))
        _est_rows = -(-_est_lines // _num_cols)
        _content_h = _est_rows * _lh + 18 + _lh + 10
        panel_h = max(80, min(int(h * panel_ratio), _content_h))
        panel = _bottom_panel(w, panel_h, triplets, 0, 0, 1, max_triplets)
        cv2.imwrite(out_path, np.concatenate([img, panel], axis=0))
    else:
        panel_w = max(240, int(w * panel_ratio))
        panel = _side_panel(panel_w, h, triplets, 0, 0, 1, max_triplets)
        cv2.imwrite(out_path, np.concatenate([img, panel], axis=1))


def annotate_video_with_triplets_panel(
    video_path: str,
    merged_srt_path: str,
    out_path: str,
    panel_ratio: float = 0.45,
    max_triplets: int = 40,
    keep_audio: bool = True,
    panel_position: str = "side",   # "side" | "bottom"
) -> None:
    """Render a triplet panel alongside every frame of `video_path` and save to `out_path`."""
    if panel_position not in ("side", "bottom"):
        raise ValueError(f"panel_position must be 'side' or 'bottom', got {panel_position!r}")

    blocks = parse_srt_triplets(merged_srt_path)
    if not blocks:
        raise RuntimeError("No SRT blocks parsed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if panel_position == "side":
        panel_w = max(240, int(round(w * panel_ratio)))
        panel_h = h
        out_w, out_h = w + panel_w, h
    else:
        # Size the bottom panel to the actual content so it isn't mostly empty.
        # Use the same font metrics that _bottom_panel will use (scale=0.5).
        _cw, _lh = _text_metrics(0.50)
        _col_w = 40 * _cw + 20
        _num_cols = max(1, (w - 12) // _col_w)
        _max_t = min(max_triplets, max((len(b.triplets) for b in blocks), default=1))
        _est_lines = max(1, int(_max_t * 1.4))           # ~1.4 lines per triplet
        _est_rows = -(-_est_lines // _num_cols)           # ceil div
        _content_h = _est_rows * _lh + 18 + _lh + 10    # y0 + footer_reserve + padding
        panel_h = max(80, min(int(round(h * panel_ratio)), _content_h))
        panel_w = w
        out_w, out_h = w, h + panel_h

    tmp_dir     = os.path.dirname(out_path) or "."
    tmp_noaudio = tempfile.mktemp(suffix="_noaudio.mp4", dir=tmp_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter.")

    hint = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ms_now = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ms_now <= 0:
            ms_now = (frame_idx / fps) * 1000.0
        ms_i = int(round(ms_now))

        block, hint = _find_active(blocks, ms_i, hint)
        if block:
            trips, seg_start, seg_end = block.triplets, block.start_ms, block.end_ms
        else:
            trips, seg_start, seg_end = [], ms_i, ms_i

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        if panel_position == "side":
            panel = _side_panel(panel_w, h, trips, ms_i, seg_start, seg_end, max_triplets)
            composed = np.concatenate([frame, panel], axis=1)
        else:
            panel = _bottom_panel(w, panel_h, trips, ms_i, seg_start, seg_end, max_triplets)
            composed = np.concatenate([frame, panel], axis=0)

        writer.write(composed)
        frame_idx += 1

    cap.release()
    writer.release()

    try:
        _encode_h264(tmp_noaudio, out_path, audio_src=video_path if keep_audio else None)
    finally:
        try:
            os.remove(tmp_noaudio)
        except OSError:
            pass
