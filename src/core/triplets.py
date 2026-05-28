"""Triplet/quintuple utilities: parsing, validation, deduplication, formatting, and prompt builders."""
from __future__ import annotations

import ast
import re
from typing import List, Tuple

Triplet = Tuple[str, str, str]
# (subject, relation, object, start_sec, end_sec) — used for temporal video segments
Quintuple = Tuple[str, str, str, float, float]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_triplets_from_text(text: str) -> List[Triplet]:
    """Parse a Python list-of-3-tuples from raw model output.

    Tries ast.literal_eval first; falls back to regex for both single- and
    double-quoted variants when the model outputs minor format violations.
    """
    text = text.strip()
    if "[" in text and "]" in text:
        text = text[text.find("[") : text.rfind("]") + 1]
    try:
        obj = ast.literal_eval(text)
    except Exception:
        singles = re.findall(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", text)
        if not singles:
            singles = re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)', text)
        return [(a.strip(), b.strip(), c.strip()) for (a, b, c) in singles]
    if not isinstance(obj, list):
        return []
    out: List[Triplet] = []
    for x in obj:
        if isinstance(x, (tuple, list)) and len(x) == 3 and all(isinstance(k, str) for k in x):
            out.append((x[0].strip(), x[1].strip(), x[2].strip()))
    return out


# ---------------------------------------------------------------------------
# Validation / normalization
# ---------------------------------------------------------------------------

def validate_triplets(trips: List[Triplet]) -> List[Triplet]:
    """Deduplicate and remove malformed triplets (empty fields, overly long nodes)."""
    seen: set = set()
    cleaned: List[Triplet] = []
    for s, r, o in trips:
        s = re.sub(r"\s+", " ", s).strip()
        r = re.sub(r"\s+", " ", r).strip()
        o = re.sub(r"\s+", " ", o).strip()
        if not s or not r or not o:
            continue
        if len(s) > 80 or len(r) > 80 or len(o) > 80:
            continue
        key = (s.lower(), r.lower(), o.lower())
        if key not in seen:
            cleaned.append((s, r, o))
            seen.add(key)
    return cleaned


# ---------------------------------------------------------------------------
# Quintuple helpers
# ---------------------------------------------------------------------------

def triplets_to_quintuples(trips: List[Triplet], start_sec: float, end_sec: float) -> List[Quintuple]:
    return [(s, r, o, start_sec, end_sec) for s, r, o in trips]


def validate_quintuples(quints: List[Quintuple]) -> List[Quintuple]:
    seen: set = set()
    cleaned: List[Quintuple] = []
    for s, r, o, t0, t1 in quints:
        s = re.sub(r"\s+", " ", s).strip()
        r = re.sub(r"\s+", " ", r).strip()
        o = re.sub(r"\s+", " ", o).strip()
        if not s or not r or not o:
            continue
        if len(s) > 80 or len(r) > 80 or len(o) > 80:
            continue
        key = (s.lower(), r.lower(), o.lower(), round(t0, 3), round(t1, 3))
        if key not in seen:
            cleaned.append((s, r, o, t0, t1))
            seen.add(key)
    return cleaned


def format_quintuples_as_json(quints: List[Quintuple]) -> list:
    return [
        {"subject": s, "relation": r, "object": o, "start_sec": t0, "end_sec": t1}
        for s, r, o, t0, t1 in quints
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_as_python_list(trips: List[Triplet]) -> str:
    if not trips:
        return "[]"
    lines = ["["]
    for s, r, o in trips:
        lines.append(f"  ({repr(s)}, {repr(r)}, {repr(o)}),")
    lines.append("]")
    return "\n".join(lines)


def format_as_json(trips: List[Triplet]) -> list:
    return [{"subject": s, "relation": r, "object": o} for s, r, o in trips]


def triplets_to_context_text(trips: List[Triplet], max_items: int = 80) -> str:
    return "\n".join(f"- ({s}, {r}, {o})" for s, r, o in trips[:max_items])


def triplets_to_claims(trips: List[Triplet]) -> List[str]:
    """Convert triplets to natural-language claim strings for RAG validation."""
    return [f"{s} {r} {o}" for s, r, o in trips]


# ---------------------------------------------------------------------------
# SRT helpers (shared with existing overlay pipeline)
# ---------------------------------------------------------------------------

def ms_to_srt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3_600_000; ms %= 3_600_000
    m = ms // 60_000;    ms %= 60_000
    s = ms // 1_000;     ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt_from_segments(segments: list) -> str:
    """Build SRT content from a list of dicts: {start, end, triplets}."""
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        trips_str = format_as_python_list(seg.get("triplets", []))
        lines += [
            str(i),
            f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}",
            trips_str,
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_scenegraph_prompt(
    transcript_text: str = "", mode: str = "high", user_text: str = "",
    template_override: Optional[str] = None,
) -> str:
    from .prompts import get_store
    transcript_section = f"\nTRANSCRIPT:\n{transcript_text}" if transcript_text else ""
    user_text_section = f"\nCONTEXT:\n{user_text}" if user_text else ""
    if template_override:
        try:
            return template_override.format(
                transcript_section=transcript_section,
                user_text_section=user_text_section,
            )
        except (KeyError, ValueError):
            return template_override
    return get_store().render(
        f"scenegraph_visual_{mode}",
        transcript_section=transcript_section,
        user_text_section=user_text_section,
    )


def build_text_only_scenegraph_prompt(
    text: str, mode: str = "high", template_override: Optional[str] = None,
) -> str:
    from .prompts import get_store
    if template_override:
        try:
            return template_override.format(text=text)
        except (KeyError, ValueError):
            return template_override
    return get_store().render(f"scenegraph_text_{mode}", text=text)


def build_temporal_scenegraph_prompt(
    start_sec: float, end_sec: float,
    transcript_text: str = "", mode: str = "high", user_text: str = "",
    template_override: Optional[str] = None,
) -> str:
    """Prompt for a specific temporal window of a video (used when segmenting long videos)."""
    from .prompts import get_store
    preamble = f"You are analyzing the video segment from {start_sec:.1f}s to {end_sec:.1f}s.\n\n"
    transcript_section = f"\nTRANSCRIPT EXCERPT:\n{transcript_text}" if transcript_text else ""
    user_text_section = f"\nCONTEXT:\n{user_text}" if user_text else ""
    if template_override:
        try:
            rendered = template_override.format(
                transcript_section=transcript_section,
                user_text_section=user_text_section,
            )
        except (KeyError, ValueError):
            rendered = template_override
        return preamble + rendered
    return preamble + get_store().render(
        f"scenegraph_visual_{mode}",
        transcript_section=transcript_section,
        user_text_section=user_text_section,
    )


def build_normalize_prompt(trips: List[Triplet]) -> str:
    from .prompts import get_store
    lines = "\n".join(f"- ({s}, {r}, {o})" for s, r, o in trips)
    return get_store().render("normalize", triplets=lines)


def build_validation_prompt(claim: str, facts: List[str]) -> str:
    facts_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(facts))
    return f"""You are a fact-checker. Given FACTS from a trusted database, evaluate the CLAIM.

FACTS:
{facts_block}

CLAIM: {claim}

Respond with JSON only (no markdown):
{{"verdict": "TRUE" | "FALSE" | "UNCERTAIN" | "NOT_COVERED", "confidence": "HIGH" | "MEDIUM" | "LOW", "explanation": "one or two sentences"}}
"""
