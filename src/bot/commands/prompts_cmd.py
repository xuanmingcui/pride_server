"""Discord slash commands for prompt template management.

Commands
--------
/prompt_list
    Show all available prompt template names and labels.

/prompt_view  name:<name>
    Display the current template (custom or default) for a named slot.

/prompt_edit  name:<name>  [template:<str>]  [file:<attachment>]
    Override a prompt template with custom text.
    Supply either `template` (inline string) or `file` (.txt attachment).

/prompt_reset  name:<name>
    Restore a prompt template to its built-in default.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import aiofiles
import discord
from discord import app_commands
from discord.ext import commands

from src.bot.commands.scenegraph_cmd import _download_attachment, _cleanup

# Template name choices — must match keys in prompts._DEFAULTS
_PROMPT_CHOICES = [
    app_commands.Choice(name="scenegraph_visual_high — Scene Graph (image, high-level)",  value="scenegraph_visual_high"),
    app_commands.Choice(name="scenegraph_visual_low  — Scene Graph (image, low/anomaly)", value="scenegraph_visual_low"),
    app_commands.Choice(name="scenegraph_video_high  — Scene Graph (video, high-level)",  value="scenegraph_video_high"),
    app_commands.Choice(name="scenegraph_video_low   — Scene Graph (video, low/anomaly)", value="scenegraph_video_low"),
    app_commands.Choice(name="scenegraph_text_high   — Scene Graph (text-only, high)",    value="scenegraph_text_high"),
    app_commands.Choice(name="scenegraph_text_low    — Scene Graph (text-only, low)",     value="scenegraph_text_low"),
    app_commands.Choice(name="identify_subjects      — Speaker/Summary pre-pass (high)",   value="identify_subjects"),
    app_commands.Choice(name="identify_subjects_low  — Event/Activity pre-pass (low)",     value="identify_subjects_low"),
    app_commands.Choice(name="canonicalize_entities  — Entity Canonicalization",          value="canonicalize_entities"),
    app_commands.Choice(name="normalize_quintuples   — Refinement (video quintuples)",    value="normalize_quintuples"),
    app_commands.Choice(name="normalize              — Refinement (triplets)",            value="normalize"),
    app_commands.Choice(name="validation             — Fact Validation Report",           value="validation"),
]


class PromptsCog(commands.Cog):
    """Cog for prompt template management commands."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    def _store(self):
        from src.core.prompts import get_store
        return get_store()

    # -----------------------------------------------------------------------
    # /prompt_list
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="prompt_list",
        description="List all prompt template slots with their labels.",
    )
    async def prompt_list(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            store = self._store()
            lines = ["**Prompt Templates**", ""]
            for name in store.names():
                meta = store.meta(name)
                custom_flag = " *(custom)*" if store.is_custom(name) else ""
                lines.append(f"• `{name}`{custom_flag}")
                lines.append(f"  {meta['label']}")
            await interaction.followup.send("\n".join(lines), ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)

    # -----------------------------------------------------------------------
    # /prompt_view
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="prompt_view",
        description="Show the current prompt template for a named slot.",
    )
    @app_commands.describe(name="Which prompt template to view.")
    @app_commands.choices(name=_PROMPT_CHOICES)
    async def prompt_view(
        self, interaction: discord.Interaction, name: str
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            store  = self._store()
            tmpl   = store.get(name)
            meta   = store.meta(name)
            status = "**custom**" if store.is_custom(name) else "default"
            header = f"**`{name}`** ({status}) — {meta['label']}\n"

            content = header + f"```\n{tmpl}\n```"
            if len(content) > 1900:
                await interaction.followup.send(
                    header + "(Template attached as file)",
                    file=discord.File(
                        fp=__import__("io").BytesIO(tmpl.encode()),
                        filename=f"{name}.txt",
                    ),
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(content, ephemeral=True)
        except KeyError:
            await interaction.followup.send(f"❌ Unknown template `{name}`.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)

    # -----------------------------------------------------------------------
    # /prompt_edit
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="prompt_edit",
        description="Override a prompt template. Supply `template` text or upload a .txt `file`.",
    )
    @app_commands.describe(
        name="Which template slot to override.",
        template="Inline replacement template text (use {variable} placeholders as needed).",
        file="Plain-text (.txt) file containing the new template (used if both given).",
    )
    @app_commands.choices(name=_PROMPT_CHOICES)
    async def prompt_edit(
        self,
        interaction: discord.Interaction,
        name: str,
        template: Optional[str] = None,
        file: Optional[discord.Attachment] = None,
    ) -> None:
        if not template and not file:
            await interaction.response.send_message(
                "Provide `template` text or a `file` attachment.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True, ephemeral=True)

        cfg      = self.bot.cfg
        tmp_dir  = cfg["paths"]["tmp_dir"]
        max_mb   = cfg["discord"]["max_upload_mb"]
        file_path: Optional[str] = None

        try:
            new_tmpl: str
            if file:
                file_path = await _download_attachment(file, tmp_dir, max_mb)
                async with aiofiles.open(file_path, "r", encoding="utf-8") as fh:
                    new_tmpl = await fh.read()
            else:
                new_tmpl = template  # type: ignore[assignment]

            new_tmpl = new_tmpl.strip()
            if not new_tmpl:
                await interaction.followup.send("Template cannot be empty.", ephemeral=True)
                return

            store = self._store()
            store.set(name, new_tmpl)

            preview = new_tmpl[:300] + ("…" if len(new_tmpl) > 300 else "")
            await interaction.followup.send(
                f"✅ Template `{name}` updated.\n```\n{preview}\n```",
                ephemeral=True,
            )
        except KeyError:
            await interaction.followup.send(f"❌ Unknown template `{name}`.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
        finally:
            _cleanup(file_path)

    # -----------------------------------------------------------------------
    # /prompt_reset
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="prompt_reset",
        description="Restore a prompt template to its built-in default.",
    )
    @app_commands.describe(name="Which template to reset.")
    @app_commands.choices(name=_PROMPT_CHOICES)
    async def prompt_reset(
        self, interaction: discord.Interaction, name: str
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            store = self._store()
            was_custom = store.is_custom(name)
            store.reset(name)
            if was_custom:
                await interaction.followup.send(
                    f"✅ Template `{name}` restored to default.", ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"ℹ️ Template `{name}` was already using the default.", ephemeral=True
                )
        except KeyError:
            await interaction.followup.send(f"❌ Unknown template `{name}`.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)
