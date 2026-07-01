"""Discord slash command for misinfo fact-checking (parity with the web tab).

/misinfo claim:<str> [top_k] [mode:standard|graph]

Forwards to the standalone GraphCheck misinfo service (same upstream the web
proxy uses, MISINFO_UPSTREAM, default http://127.0.0.1:8090) and summarises the
verdict, evidence stance counts, and retrieved evidence.
"""
from __future__ import annotations

import io
import json
import os
from typing import Optional

import httpx
import discord
from discord import app_commands
from discord.ext import commands

_UPSTREAM = os.environ.get("MISINFO_UPSTREAM", "http://127.0.0.1:8090")


class MisinfoCog(commands.Cog):
    """Cog for /misinfo."""

    def __init__(self, bot: discord.Client):
        self.bot = bot

    @app_commands.command(
        name="misinfo",
        description="Fact-check a claim against the misinfo knowledge base.",
    )
    @app_commands.describe(
        claim="The claim / statement to verify.",
        top_k="Number of evidence documents to retrieve (default from service).",
        mode="'standard' (default) or 'graph' (decompose the claim, then verify).",
    )
    @app_commands.choices(mode=[
        app_commands.Choice(name="standard", value="standard"),
        app_commands.Choice(name="graph — decompose & verify", value="graph"),
    ])
    async def misinfo(
        self,
        interaction: discord.Interaction,
        claim: str,
        top_k: Optional[int] = None,
        mode: str = "standard",
    ) -> None:
        await interaction.response.defer(thinking=True)
        endpoint = "/verify_decomposition" if mode == "graph" else "/verify"
        body = {"prompt": claim}
        if top_k is not None:
            body["top_k"] = max(1, int(top_k))
        try:
            async with httpx.AsyncClient(base_url=_UPSTREAM, timeout=httpx.Timeout(120.0)) as cl:
                r = await cl.post(endpoint, json=body)
        except httpx.ConnectError:
            await interaction.followup.send(
                f"❌ Misinfo service unreachable at {_UPSTREAM}. Start it with: "
                "`conda activate misinfo && CUDA_VISIBLE_DEVICES=1 python -m misinfo.serve`"
            )
            return
        except httpx.TimeoutException:
            await interaction.followup.send("❌ Misinfo service timed out.")
            return
        if r.status_code != 200:
            err = ""
            try:
                err = r.json().get("error") or r.json().get("load_error") or ""
            except Exception:
                err = r.text[:300]
            await interaction.followup.send(f"❌ Misinfo service error ({r.status_code}): {err}")
            return

        result = r.json()
        text = self._summarize(claim, result, mode)
        if len(text) > 1900:
            text = text[:1900] + "\n… (truncated; full result attached)"
        await interaction.followup.send(
            text,
            file=discord.File(
                fp=io.BytesIO(json.dumps(result, ensure_ascii=False, indent=2).encode()),
                filename="misinfo.json",
            ),
        )

    @staticmethod
    def _summarize(claim: str, result: dict, mode: str) -> str:
        if mode == "graph":
            # Decomposition result: best-effort summary of sub-claim verdicts.
            lines = [f"🧪 **Misinfo (graph)** — “{claim}”"]
            overall = result.get("prediction") or result.get("verdict")
            if overall:
                lines.append(f"Overall: **{overall}**")
            subs = result.get("subclaims") or result.get("decomposition") or result.get("results")
            if isinstance(subs, list):
                for i, s in enumerate(subs[:8], 1):
                    if isinstance(s, dict):
                        c = s.get("claim") or s.get("prompt") or s.get("text") or ""
                        p = s.get("prediction") or s.get("verdict") or "?"
                        lines.append(f"{i}. [{p}] {c}")
            return "\n".join(lines)

        pred = result.get("prediction", "—")
        emoji = "✅" if pred == "SUPPORTED" else ("❌" if pred == "REFUTED" else "❓")
        support = len(result.get("support_indices") or [])
        refute = len(result.get("refute_indices") or [])
        unrel = len(result.get("not_related") or [])
        lines = [
            f"🧪 **Misinfo** — “{claim}”",
            f"Verdict: {emoji} **{pred}**  ·  support {support} · refute {refute} · unrelated {unrel}",
        ]
        evidence = (result.get("evidence") or "").strip()
        if evidence:
            ev_lines = [l for l in evidence.split("\n") if l.strip()][:5]
            lines.append("\n**Evidence:**")
            lines.extend(f"  {l}" for l in ev_lines)
        return "\n".join(lines)
