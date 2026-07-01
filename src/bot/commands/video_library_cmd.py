"""Discord slash commands for the Video Library (parity with the web tab).

/video_index   media:<attachment> [title] [mode] [fps] [temperature] [normalize]
    Generate a scene graph for the attached video and index its rows into the
    structured, searchable library (rows embedded + the file stored for replay).

/video_search  query:<str> [top_k] [facet] [keyword] [min_score]
    Retrieve indexed videos whose scene-graph rows best match the query, with the
    same relevance threshold + hybrid (facet / keyword) filters as the web UI.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from src.bot.commands.scenegraph_cmd import _download_attachment, _cleanup


def _flatten_sg(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a pipeline result into uniform row dicts for the index."""
    rows: List[Dict[str, Any]] = []

    def _emit(t):
        if isinstance(t, (list, tuple)) and len(t) >= 5:
            rows.append({"subject": t[0], "relation": t[1], "object": t[2],
                         "start_sec": float(t[3]), "end_sec": float(t[4])})
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            rows.append({"subject": t[0], "relation": t[1], "object": t[2]})

    segs = result.get("segments") or []
    if segs:
        for seg in segs:
            for t in seg.get("triplets", []):
                _emit(t)
    else:
        for t in result.get("triplets", []):
            _emit(t)
    return rows


class VideoLibraryCog(commands.Cog):
    """Cog for /video_index and /video_search."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    # ── Index ──────────────────────────────────────────────────────────────
    @app_commands.command(
        name="video_index",
        description="Generate a scene graph for a video and add it to the searchable Video Library.",
    )
    @app_commands.describe(
        media="Video file to index.",
        title="Display title (defaults to the file name).",
        mode="Scene graph mode: 'high' (default) or 'low'.",
        fps="Frame sampling rate (default from config).",
        temperature="Sampling temperature (default from config).",
        normalize="Run the refinement pass before indexing (on by default).",
    )
    @app_commands.choices(mode=[
        app_commands.Choice(name="high — semantic / events (default)", value="high"),
        app_commands.Choice(name="low  — physical / events / anomalies", value="low"),
    ])
    async def video_index(
        self,
        interaction: discord.Interaction,
        media: discord.Attachment,
        title: Optional[str] = None,
        mode: str = "high",
        fps: Optional[float] = None,
        temperature: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        await interaction.response.defer(thinking=True)
        cfg = self.bot.cfg
        tmp_dir = cfg["paths"]["tmp_dir"]
        max_mb = cfg["discord"]["max_upload_mb"]
        _mode = mode if mode in ("high", "low") else "high"
        media_path = None
        try:
            media_path = await _download_attachment(media, tmp_dir, max_mb)
            ext = "." + media.filename.rsplit(".", 1)[-1] if "." in media.filename else ".mp4"
            disp_title = title or media.filename

            def _work():
                result = self.bot.sg_pipeline.process(
                    media_path=media_path, text="", output_type="json",
                    mode=_mode, fps=fps, temperature=temperature, normalize=normalize,
                )
                rows = _flatten_sg(result)
                return self.bot.video_index.index_video(
                    rows=rows, title=disp_title, media_src_path=media_path,
                    ext=ext, source="discord",
                )

            record = await self.bot.run_in_thread(_work)
            await interaction.followup.send(
                f"📚 **Indexed** “{record['title']}” — {record['num_rows']} rows"
                + (f", {record['duration']:.0f}s" if record.get("duration") else "")
                + f" (id `{record['id'][:8]}`). Search it with `/video_search`."
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}")
        finally:
            _cleanup(media_path)

    # ── Search ─────────────────────────────────────────────────────────────
    @app_commands.command(
        name="video_search",
        description="Search the Video Library for videos whose scene graph matches a query.",
    )
    @app_commands.describe(
        query="Free-text query (e.g. 'people fleeing during an earthquake').",
        top_k="Max videos to return (default 10).",
        facet="Restrict to a row type.",
        keyword="Require this exact word to appear in a matching row.",
        min_score="Relevance floor 0–1 (default 0.30; 0 = show all, ranked).",
    )
    @app_commands.choices(facet=[
        app_commands.Choice(name="people", value="people"),
        app_commands.Choice(name="location", value="location"),
        app_commands.Choice(name="action", value="action"),
        app_commands.Choice(name="text & speech", value="text"),
        app_commands.Choice(name="framing", value="framing"),
    ])
    async def video_search(
        self,
        interaction: discord.Interaction,
        query: str,
        top_k: int = 10,
        facet: Optional[str] = None,
        keyword: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            results = await self.bot.run_in_thread(
                self.bot.video_index.search,
                query, max_videos=max(1, top_k), facet=facet or None,
                keyword=(keyword or None), min_score=min_score,
            )
            if not results:
                await interaction.followup.send(
                    f"🔍 No videos matched “{query}” above the relevance threshold. "
                    "Try lowering `min_score` (e.g. 0) or a broader query."
                )
                return

            lines = [f"🔍 **{len(results)}** video(s) for “{query}”:"]
            for v in results:
                lines.append(
                    f"\n**{v['title']}** · score {v['score']:.2f} · {v['num_matches']} match(es)"
                )
                for m in v.get("matches", [])[:3]:
                    t = ""
                    if m.get("start_sec") is not None:
                        t = f" [{m['start_sec']:.0f}–{m['end_sec']:.0f}s]"
                    lines.append(f"  • {m['subject']} → {m['relation']} → {m['object']}{t}")
            text = "\n".join(lines)
            if len(text) > 1900:
                text = text[:1900] + "\n… (truncated; full results attached)"
            await interaction.followup.send(
                text,
                file=discord.File(
                    fp=io.BytesIO(json.dumps(results, ensure_ascii=False, indent=2).encode()),
                    filename="video_search.json",
                ),
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}")
