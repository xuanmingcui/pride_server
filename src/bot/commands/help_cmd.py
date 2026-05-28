"""Discord /help command — overview of PRIDE bot capabilities."""
from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

_HELP_EMBED = discord.Embed(
    title="PRIDE Bot — Command Reference",
    description=(
        "PRIDE (**P**ropaganda & **R**easoning **I**ntelligence **D**etection **E**ngine) "
        "analyses media and text using multimodal AI.\n​"
    ),
    colour=0x5865F2,
)

_HELP_EMBED.add_field(
    name="🎬  /scenegraph",
    value=(
        "Extract a structured scene graph (subject → relation → object triplets) "
        "from a **video**, **image**, or **text**.\n"
        "Optionally returns an annotated overlay video/image.\n"
        "**Options:** `media` · `text` · `output_type` (json/overlay) · "
        "`mode` (high/low) · `temperature` · `num_frames`\n"
        "• **high** — semantic, named entities & events (news/documentary)\n"
        "• **low**  — physical objects, positions & actions (everyday video)"
    ),
    inline=False,
)

_HELP_EMBED.add_field(
    name="🔍  /validate",
    value=(
        "Fact-check a media clip or text claim against a stored **fact database**.\n"
        "Retrieves the most relevant facts via semantic search, then runs a full MLLM reasoning pass.\n"
        "**Options:** `media` · `text` · `database` · `top_k` · `temperature`"
    ),
    inline=False,
)

_HELP_EMBED.add_field(
    name="📚  Fact Database Management",
    value=(
        "`/add_facts` — add facts by text or `.txt` file upload\n"
        "`/list_facts` — browse or semantically search stored facts\n"
        "`/delete_facts` — remove facts by ID\n"
        "`/create_database` — create a new named fact collection\n"
        "`/list_databases` — show all databases and their fact counts"
    ),
    inline=False,
)

_HELP_EMBED.add_field(
    name="📝  Prompt Customization",
    value=(
        "`/prompt_list` — show all template slots (marks customized ones)\n"
        "`/prompt_view name:<slot>` — display the active template\n"
        "`/prompt_edit name:<slot> template:<text>` — override a template (or supply a `.txt` file)\n"
        "`/prompt_reset name:<slot>` — restore built-in default\n"
        "Slots: `scenegraph_visual_high/low` · `scenegraph_text_high/low` · `normalize` · `validation`"
    ),
    inline=False,
)

_HELP_EMBED.set_footer(text="All ML tasks run asynchronously — responses may take a few seconds.")


class HelpCog(commands.Cog):
    """Cog for /help."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    @app_commands.command(name="help", description="Show an overview of PRIDE bot commands.")
    async def help(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(embed=_HELP_EMBED, ephemeral=True)
