"""Discord slash commands for scene-graph generation (Function 1).

/scenegraph  media:<attachment>  [text:<str>]  [output_type:json|overlay]
             [mode:high|low]  [model:<str>]  [temperature:<float>]  [fps:<float>]  [normalize:<bool>]

Workflow:
  1. Defer the interaction (processing takes time).
  2. Download attached media to a temp file.
  3. Run SceneGraphPipeline in the thread-pool executor.
  4. Reply with JSON or the annotated video file.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time
from typing import Optional

import aiohttp
import aiofiles
import discord
from discord import app_commands
from discord.ext import commands

_STAGE_ICON = {"done": "✅", "running": "⏳", "pending": "⬜"}


def _fmt_stages(stages) -> str:
    """Render the pipeline stage list as a compact Discord checklist."""
    out = []
    for s in stages or []:
        ic = _STAGE_ICON.get(s.get("state"), "⬜")
        name = s.get("label", s.get("key"))
        extra = ""
        if s.get("state") == "running":
            if s.get("determinate"):
                extra = f" — {round(s.get('percent', 0))}%"
            d = s.get("detail")
            if d:
                extra += (" — " if not extra else " · ") + d
        elif s.get("state") == "done" and s.get("detail"):
            extra = f" — {s['detail']}"
        out.append(f"{ic} {name}{extra}")
    return "\n".join(out)


class SceneGraphCog(commands.Cog):
    """Cog for /scenegraph."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    @app_commands.command(
        name="scenegraph",
        description="Generate a scene graph from text, image, or video (with audio).",
    )
    @app_commands.describe(
        media="Video (mp4) or image file to analyse.",
        text="Text input or extra context (used alone or combined with media).",
        output_type="Output format: 'json' (default), 'overlay' (annotated video/image), or 'raw' (model text).",
        mode="Scene graph mode: 'high' for semantic/news (default), 'low' for physical/visual.",
        temperature="Sampling temperature (default from config, e.g. 0.8).",
        fps="Frame sampling rate (frames per second; default from config, e.g. 1.0). Per-call frame count = ceil(window_seconds × fps), clamped by min_frames and the context budget.",
        normalize="Run the refinement pass after generation: global entity normalization + dedup + quality filter (on by default; set False to skip and save latency).",
        model="Override model name (must be loaded; restart required to change).",
    )
    @app_commands.choices(
        output_type=[
            app_commands.Choice(name="json",    value="json"),
            app_commands.Choice(name="overlay", value="overlay"),
            app_commands.Choice(name="raw",     value="raw"),
        ],
        mode=[
            app_commands.Choice(name="high — semantic / named entities / events (default)", value="high"),
            app_commands.Choice(name="low  — physical / objects / spatial positions",       value="low"),
        ],
    )
    async def scenegraph(
        self,
        interaction: discord.Interaction,
        media: Optional[discord.Attachment] = None,
        text: Optional[str] = None,
        output_type: str = "json",
        mode: str = "high",
        temperature: Optional[float] = None,
        fps: Optional[float] = None,
        normalize: bool = True,
        model: Optional[str] = None,
    ) -> None:
        if not media and not text:
            await interaction.response.send_message(
                "Please provide at least `media` (file attachment) or `text`.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        cfg      = self.bot.cfg
        tmp_dir  = cfg["paths"]["tmp_dir"]
        out_dir  = cfg["paths"]["output_dir"]
        max_mb   = cfg["discord"]["max_upload_mb"]

        raw_output = output_type == "raw"
        eff_output = "overlay" if output_type == "overlay" else "json"

        # Live progress: edit the deferred reply with the stage checklist as the
        # pipeline reports it. The callback runs in the ML worker thread, so we
        # marshal the edit back onto the bot's event loop, throttled to avoid
        # Discord rate limits (only on stage change, ≥1.2s apart).
        loop = asyncio.get_running_loop()
        _pstate = {"sig": None, "ts": 0.0}

        def _progress(percent, stage, detail, stages):
            sig = tuple(
                (s["key"], s["state"], round(s.get("percent", 0) / 10) if s.get("determinate") else 0)
                for s in stages
            )
            now = time.monotonic()
            if sig == _pstate["sig"] or (now - _pstate["ts"]) < 1.2:
                return
            _pstate["sig"], _pstate["ts"] = sig, now
            content = "**Scene Graph** — processing…\n" + _fmt_stages(stages)
            fut = asyncio.run_coroutine_threadsafe(
                interaction.edit_original_response(content=content), loop
            )
            fut.add_done_callback(lambda f: f.exception())  # swallow edit errors

        try:
            # Download attachment
            media_path: Optional[str] = None
            if media:
                media_path = await _download_attachment(media, tmp_dir, max_mb)

            # Run pipeline (blocking → thread pool)
            out_path: Optional[str] = None
            if eff_output == "overlay" and media_path:
                ext = ".mp4" if not _is_image(media_path) else ".png"
                out_path = tempfile.mktemp(suffix=f"_sg_overlay{ext}", dir=out_dir)

            _mode = mode if mode in ("high", "low") else "high"
            result = await self.bot.run_in_thread(
                self.bot.sg_pipeline.process,
                media_path=media_path,
                text=text or "",
                output_type=eff_output,
                output_path=out_path,
                mode=_mode,
                temperature=temperature,
                fps=fps,
                normalize=normalize,
                raw_output=raw_output,
                progress_cb=_progress,
            )

            transcript = (result.get("transcript") or "").strip()

            # Raw mode: reply with the model's raw text (per segment if temporal).
            if raw_output:
                segs = result.get("segments", [])
                if segs:
                    parts = []
                    for seg in segs:
                        hdr = f"# [{seg.get('start')}-{seg.get('end')}]"
                        parts.append(f"{hdr}\n{seg.get('raw_text', '')}")
                    raw_text = "\n\n".join(parts)
                else:
                    raw_text = result.get("raw_text", "")
                files = [discord.File(fp=io.BytesIO(raw_text.encode()), filename="scene_graph_raw.txt")]
                if transcript:
                    files.append(discord.File(fp=io.BytesIO(transcript.encode()),
                                              filename="transcript.txt"))
                await interaction.followup.send(
                    f"**Scene Graph (raw)** — {len(segs) or 1} block(s)"
                    + (f" · transcript {len(transcript)} chars" if transcript else ""),
                    files=files,
                )
                return

            # Format response. Video segments carry quintuples (s, r, o, start_sec, end_sec);
            # image / text-only carry triplets (s, r, o).
            def _fmt_item(t):
                if len(t) >= 5:
                    return {
                        "subject":   t[0],
                        "relation":  t[1],
                        "object":    t[2],
                        "start_sec": float(t[3]),
                        "end_sec":   float(t[4]),
                    }
                return {"subject": t[0], "relation": t[1], "object": t[2]}

            segments    = result.get("segments", [])
            is_temporal = bool(segments)
            flat_trips  = result.get("triplets", [])  # populated for image / text-only
            total_items = (
                sum(len(seg.get("triplets", [])) for seg in segments)
                if is_temporal else len(flat_trips)
            )
            kind = f"{len(segments)} segment(s)" if is_temporal else "image/text"
            unit = "quintuple" if is_temporal else "triplet"
            msg_lines = [f"**Scene Graph** — {kind}, {total_items} {unit}(s)"]
            if transcript:
                msg_lines.append(f"🗣️ transcript: {len(transcript)} chars (attached)")

            overlay_path = result.get("overlay_path")
            overlay_err  = result.get("overlay_error")

            # Transcript attachment (parity with the web transcript panel).
            extra_files = []
            if transcript:
                extra_files.append(
                    discord.File(fp=io.BytesIO(transcript.encode()), filename="transcript.txt")
                )

            if output_type == "json" or not overlay_path:
                if is_temporal:
                    json_data = {
                        "segments": [
                            {
                                "start": seg["start"],
                                "end":   seg["end"],
                                "triplets": [_fmt_item(t) for t in seg.get("triplets", [])],
                            }
                            for seg in segments
                        ],
                    }
                else:
                    json_data = {
                        "triplets": [_fmt_item(t) for t in flat_trips],
                    }
                if transcript:
                    json_data["transcript"] = transcript
                json_bytes = json.dumps(json_data, ensure_ascii=False, indent=2).encode()

                if overlay_err:
                    msg_lines.append(f"⚠️ Overlay failed: {overlay_err}")

                await interaction.followup.send(
                    "\n".join(msg_lines),
                    files=[discord.File(fp=io.BytesIO(json_bytes), filename="scene_graph.json"),
                           *extra_files],
                )
            else:
                # Attach overlay file
                size_mb = os.path.getsize(overlay_path) / 1_048_576
                if size_mb > max_mb:
                    msg_lines.append(
                        f"⚠️ Overlay file ({size_mb:.1f} MB) exceeds Discord limit ({max_mb} MB). Sending JSON instead."
                    )
                    json_bytes = json.dumps(
                        [_fmt_item(t) for t in flat_trips],
                        ensure_ascii=False, indent=2
                    ).encode()
                    await interaction.followup.send(
                        "\n".join(msg_lines),
                        files=[discord.File(fp=io.BytesIO(json_bytes), filename="scene_graph.json"),
                               *extra_files],
                    )
                else:
                    await interaction.followup.send(
                        "\n".join(msg_lines),
                        files=[discord.File(overlay_path), *extra_files],
                    )

        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}")
        finally:
            _cleanup(media_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


async def _download_attachment(
    attachment: discord.Attachment, tmp_dir: str, max_mb: float
) -> str:
    """Stream-download a Discord attachment to a temp file. Returns local path."""
    size_mb = attachment.size / 1_048_576
    if size_mb > max_mb:
        raise ValueError(
            f"Attachment ({size_mb:.1f} MB) exceeds the configured limit of {max_mb} MB."
        )
    ext  = os.path.splitext(attachment.filename)[1] or ".bin"
    path = tempfile.mktemp(suffix=ext, dir=tmp_dir)
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            resp.raise_for_status()
            async with aiofiles.open(path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    await fh.write(chunk)
    return path


def _cleanup(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass
