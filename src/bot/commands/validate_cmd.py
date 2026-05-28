"""Discord slash commands for fact validation and database management (Function 2).

Commands
--------
/validate   media [text] [database] [top_k] [temperature]
    Validate a media clip or text claim against the fact database.

/add_facts  [facts] [file] [database] [tags]
    Add semicolon-separated facts, or upload a .txt file (one fact per line).

/list_facts [database] [limit] [query]
    Browse / semantically search facts stored in a database.

/delete_facts  fact_ids  [database]
    Remove specific facts by comma-separated IDs.

/create_database  name
    Create a new named fact collection.

/list_databases
    List all available fact databases and their fact counts.
"""
from __future__ import annotations

import json
from typing import Optional

import aiofiles
import discord
from discord import app_commands
from discord.ext import commands

from src.bot.commands.scenegraph_cmd import _download_attachment, _cleanup


class ValidateCog(commands.Cog):
    """Cog for all validation and database management commands."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    # -----------------------------------------------------------------------
    # /validate
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="validate",
        description="Fact-check media or text against the fact database.",
    )
    @app_commands.describe(
        media="Video (mp4) or image to fact-check.",
        text="Text claim or extra context (used alone or alongside media).",
        database="Which fact database to query (default: 'default').",
        top_k="Number of facts retrieved from the database (default 5).",
        show_facts="Also list the retrieved database facts used as context.",
    )
    async def validate(
        self,
        interaction: discord.Interaction,
        media: Optional[discord.Attachment] = None,
        text: Optional[str] = None,
        database: Optional[str] = None,
        top_k: Optional[int] = None,
        show_facts: bool = False,
    ) -> None:
        if not media and not text:
            await interaction.response.send_message(
                "Please provide `media` or `text` to validate.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        cfg     = self.bot.cfg
        db_name = database or cfg["validation"]["default_db"]
        tmp_dir = cfg["paths"]["tmp_dir"]
        max_mb  = cfg["discord"]["max_upload_mb"]

        media_path: Optional[str] = None
        try:
            if media:
                media_path = await _download_attachment(media, tmp_dir, max_mb)

            report = await self.bot.run_in_thread(
                self.bot.val_pipeline.validate,
                database=db_name,
                media_path=media_path,
                text=text or "",
                top_k=top_k,
            )

            discord_text = report.format_discord()

            if show_facts and report.retrieved_facts:
                facts_lines = ["", "**Retrieved facts:**"]
                for i, f in enumerate(report.retrieved_facts, 1):
                    facts_lines.append(f"`{i}.` {f}")
                facts_block = "\n".join(facts_lines)
                # Attach as file if combined text would be too long
                if len(discord_text) + len(facts_block) > 1900:
                    full_text = discord_text + "\n\n" + "\n".join(facts_lines[1:])
                    await interaction.followup.send(
                        discord_text,
                        file=discord.File(
                            fp=__import__("io").BytesIO(full_text.encode()),
                            filename="validation_report.txt",
                        ),
                    )
                    return
                discord_text += facts_block

            if len(discord_text) > 1900:
                await interaction.followup.send(
                    discord_text[:1900] + "\n…*(truncated — full report attached)*",
                    file=discord.File(
                        fp=__import__("io").BytesIO(
                            json.dumps(report.to_dict(), ensure_ascii=False, indent=2).encode()
                        ),
                        filename="validation_report.json",
                    ),
                )
            else:
                await interaction.followup.send(discord_text)

        except Exception as e:
            await interaction.followup.send(f"❌ Validation error: {e}")
        finally:
            _cleanup(media_path)

    # -----------------------------------------------------------------------
    # /add_facts
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="add_facts",
        description="Add facts to a database. Use semicolons to separate multiple facts.",
    )
    @app_commands.describe(
        facts="Semicolon-separated facts, e.g. 'Fact 1; Fact 2; Fact 3'.",
        file="Text file with one fact per line (overrides `facts` if both given).",
        database="Target database name (default: 'default').",
        tags="Comma-separated tags for these facts (e.g. 'politics,singapore').",
        source="Source label stored in metadata (default: 'user').",
    )
    async def add_facts(
        self,
        interaction: discord.Interaction,
        facts: Optional[str] = None,
        file: Optional[discord.Attachment] = None,
        database: Optional[str] = None,
        tags: Optional[str] = None,
        source: str = "user",
    ) -> None:
        if not facts and not file:
            await interaction.response.send_message(
                "Provide `facts` text or a `file` attachment.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        cfg     = self.bot.cfg
        db_name = database or cfg["validation"]["default_db"]
        tmp_dir = cfg["paths"]["tmp_dir"]
        max_mb  = cfg["discord"]["max_upload_mb"]

        fact_list: list[str] = []
        file_path: Optional[str] = None

        try:
            if file:
                file_path = await _download_attachment(file, tmp_dir, max_mb)
                async with aiofiles.open(file_path, "r", encoding="utf-8") as fh:
                    content = await fh.read()
                fact_list = [ln.strip() for ln in content.splitlines() if ln.strip()]
            elif facts:
                fact_list = [f.strip() for f in facts.split(";") if f.strip()]

            if not fact_list:
                await interaction.followup.send("No non-empty facts found.")
                return

            ids = await self.bot.run_in_thread(
                self.bot.db.add_facts,
                db_name,
                fact_list,
                source,
                tags or "",
            )

            await interaction.followup.send(
                f"✅ Added **{len(ids)}** fact(s) to database `{db_name}`.\n"
                f"IDs: `{', '.join(ids[:5])}{'…' if len(ids) > 5 else ''}`"
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Error adding facts: {e}")
        finally:
            _cleanup(file_path)

    # -----------------------------------------------------------------------
    # /list_facts
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="list_facts",
        description="Browse or semantically search facts in a database.",
    )
    @app_commands.describe(
        database="Database to list facts from (default: 'default').",
        limit="Maximum number of facts to show (default 15).",
        query="Optional semantic search query to filter relevant facts.",
    )
    async def list_facts(
        self,
        interaction: discord.Interaction,
        database: Optional[str] = None,
        limit: int = 15,
        query: Optional[str] = None,
    ) -> None:
        await interaction.response.defer(thinking=True)

        cfg     = self.bot.cfg
        db_name = database or cfg["validation"]["default_db"]

        try:
            facts = await self.bot.run_in_thread(
                self.bot.db.list_facts, db_name, limit, 0, query
            )
            total = await self.bot.run_in_thread(self.bot.db.count, db_name)

            if not facts:
                await interaction.followup.send(
                    f"Database `{db_name}` is empty (or no matches for query)."
                )
                return

            header = f"**Facts in `{db_name}`** (showing {len(facts)}/{total})"
            if query:
                header += f' — search: *"{query}"*'
            lines = [header, ""]
            for i, entry in enumerate(facts, 1):
                fid  = entry.get("id", "?")
                fact = entry["fact"]
                meta = entry.get("metadata", {})
                tags = meta.get("tags", "")
                tag_str = f" `[{tags}]`" if tags else ""
                lines.append(f"`{i}.` {fact}{tag_str}")
                lines.append(f"    ID: `{fid}`")

            text = "\n".join(lines)
            if len(text) > 1900:
                # Send as file
                await interaction.followup.send(
                    f"{header}\n(Full list attached)",
                    file=discord.File(
                        fp=__import__("io").BytesIO(text.encode()),
                        filename="facts.txt",
                    ),
                )
            else:
                await interaction.followup.send(text)

        except Exception as e:
            await interaction.followup.send(f"❌ Error listing facts: {e}")

    # -----------------------------------------------------------------------
    # /delete_facts
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="delete_facts",
        description="Delete specific facts from a database by their IDs.",
    )
    @app_commands.describe(
        fact_ids="Comma-separated fact IDs to delete (shown by /list_facts).",
        database="Database to delete from (default: 'default').",
    )
    async def delete_facts(
        self,
        interaction: discord.Interaction,
        fact_ids: str,
        database: Optional[str] = None,
    ) -> None:
        await interaction.response.defer(thinking=True)

        cfg     = self.bot.cfg
        db_name = database or cfg["validation"]["default_db"]
        ids     = [x.strip() for x in fact_ids.split(",") if x.strip()]

        if not ids:
            await interaction.followup.send("No valid IDs provided.", ephemeral=True)
            return

        try:
            await self.bot.run_in_thread(self.bot.db.delete_facts, db_name, ids)
            await interaction.followup.send(
                f"🗑️ Deleted {len(ids)} fact(s) from `{db_name}`."
            )
        except Exception as e:
            await interaction.followup.send(f"❌ Error deleting facts: {e}")

    # -----------------------------------------------------------------------
    # /create_database
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="create_database",
        description="Create a new named fact database.",
    )
    @app_commands.describe(name="Unique name for the new database (letters, digits, underscores).")
    async def create_database(
        self, interaction: discord.Interaction, name: str
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            await self.bot.run_in_thread(self.bot.db.create_database, name)
            await interaction.followup.send(f"✅ Database `{name}` created (or already existed).")
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}")

    # -----------------------------------------------------------------------
    # /list_databases
    # -----------------------------------------------------------------------

    @app_commands.command(
        name="list_databases",
        description="List all fact databases and their fact counts.",
    )
    async def list_databases(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True)
        try:
            names = await self.bot.run_in_thread(self.bot.db.list_databases)
            if not names:
                await interaction.followup.send("No databases created yet.")
                return
            lines = ["**Fact Databases**", ""]
            for n in names:
                count = await self.bot.run_in_thread(self.bot.db.count, n)
                lines.append(f"• `{n}` — {count} fact(s)")
            await interaction.followup.send("\n".join(lines))
        except Exception as e:
            await interaction.followup.send(f"❌ Error: {e}")
