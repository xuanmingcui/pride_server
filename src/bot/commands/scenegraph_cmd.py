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

import json
import os
import tempfile
from typing import Optional

import aiohttp
import aiofiles
import discord
from discord import app_commands
from discord.ext import commands


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
        output_type="Output format: 'json' (default) or 'overlay' (annotated video/image).",
        mode="Scene graph mode: 'high' for semantic/news (default), 'low' for physical/visual.",
        temperature="Sampling temperature (default from config, e.g. 0.8).",
        fps="Frame sampling rate (frames per second; default from config, e.g. 1.0). Per-call frame count = ceil(window_seconds × fps), clamped by min_frames and the context budget.",
        normalize="Run the refinement pass after generation: entity normalization + dedup + quality filter (off by default; adds latency).",
        model="Override model name (must be loaded; restart required to change).",
    )
    @app_commands.choices(
        output_type=[
            app_commands.Choice(name="json",    value="json"),
            app_commands.Choice(name="overlay", value="overlay"),
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
        normalize: bool = False,
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

        try:
            # Download attachment
            media_path: Optional[str] = None
            if media:
                media_path = await _download_attachment(media, tmp_dir, max_mb)

            # Run pipeline (blocking → thread pool)
            out_path: Optional[str] = None
            if output_type == "overlay" and media_path:
                ext = ".mp4" if not _is_image(media_path) else ".png"
                out_path = tempfile.mktemp(suffix=f"_sg_overlay{ext}", dir=out_dir)

            _mode = mode if mode in ("high", "low") else "high"
            result = await self.bot.run_in_thread(
                self.bot.sg_pipeline.process,
                media_path=media_path,
                text=text or "",
                output_type=output_type,
                output_path=out_path,
                mode=_mode,
                temperature=temperature,
                fps=fps,
                normalize=normalize,
            )

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

            overlay_path = result.get("overlay_path")
            overlay_err  = result.get("overlay_error")

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
                json_bytes = json.dumps(json_data, ensure_ascii=False, indent=2).encode()

                if overlay_err:
                    msg_lines.append(f"⚠️ Overlay failed: {overlay_err}")

                await interaction.followup.send(
                    "\n".join(msg_lines),
                    file=discord.File(fp=__import__("io").BytesIO(json_bytes), filename="scene_graph.json"),
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
                        file=discord.File(fp=__import__("io").BytesIO(json_bytes), filename="scene_graph.json"),
                    )
                else:
                    await interaction.followup.send(
                        "\n".join(msg_lines),
                        file=discord.File(overlay_path),
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
