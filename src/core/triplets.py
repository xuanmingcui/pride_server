"""Triplet/quintuple utilities: parsing, validation, deduplication, formatting, and prompt builders."""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

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
        if len(s) > 100 or len(r) > 80 or len(o) > 100:
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


def extract_quintuples_from_text(text: str) -> List[Quintuple]:
    """Parse a Python list of 5-tuples (s, r, o, start_sec, end_sec) from raw model output.

    Tries ast.literal_eval first; falls back to a regex that captures the three
    string fields plus two numeric fields when the model emits minor format issues.
    """
    text = text.strip()
    if "[" in text and "]" in text:
        text = text[text.find("[") : text.rfind("]") + 1]
    out: List[Quintuple] = []
    try:
        obj = ast.literal_eval(text)
    except Exception:
        # Regex fallback: 3 string fields + 2 numeric fields.
        pat_s = (
            r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*,"
            r"\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
        )
        pat_d = (
            r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,'
            r'\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)'
        )
        matches = re.findall(pat_s, text) or re.findall(pat_d, text)
        for s, r, o, t0, t1 in matches:
            try:
                out.append((s.strip(), r.strip(), o.strip(), float(t0), float(t1)))
            except ValueError:
                continue
        return out
    if not isinstance(obj, list):
        return []
    for x in obj:
        if not isinstance(x, (tuple, list)) or len(x) != 5:
            continue
        s, r, o, t0, t1 = x
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue
        try:
            t0 = float(t0); t1 = float(t1)
        except (TypeError, ValueError):
            continue
        out.append((s.strip(), r.strip(), o.strip(), t0, t1))
    return out


def shift_quintuples(quints: List[Quintuple], offset_sec: float,
                     segment_duration_sec: Optional[float] = None) -> List[Quintuple]:
    """Translate quintuple times from segment-relative to absolute video time.

    Clamps each (start, end) to [0, segment_duration_sec] before applying the
    offset so an MLLM that drifts slightly outside the window does not produce
    out-of-range absolute times.
    """
    out: List[Quintuple] = []
    for s, r, o, t0, t1 in quints:
        if segment_duration_sec is not None:
            t0 = max(0.0, min(t0, segment_duration_sec))
            t1 = max(t0, min(t1, segment_duration_sec))
        else:
            if t1 < t0:
                t1 = t0
        out.append((s, r, o, t0 + offset_sec, t1 + offset_sec))
    return out


def merge_quintuples(quints: List[Quintuple], gap_sec: float = 1.0) -> List[Quintuple]:
    """Merge identical (subject, relation, object) rows with touching time windows.

    Groups rows by case-insensitive (s, r, o) and collapses any whose windows
    overlap or sit within ``gap_sec`` of each other into a single row spanning
    [min start, max end]. This removes the degenerate pattern where an MLLM
    emits the same relationship many times in micro-stepped windows
    (0.0-0.2, 0.2-0.4, …) across a static shot, while preserving genuinely
    separated re-occurrences of the same relationship as distinct rows.
    The first-seen surface form (casing) of each group is kept; group order
    follows first appearance.
    """
    from collections import OrderedDict

    groups: "OrderedDict[tuple, list]" = OrderedDict()
    forms: Dict[tuple, Triplet] = {}
    for s, r, o, t0, t1 in quints:
        key = (s.lower(), r.lower(), o.lower())
        groups.setdefault(key, []).append((float(t0), float(t1)))
        forms.setdefault(key, (s, r, o))

    out: List[Quintuple] = []
    for key, spans in groups.items():
        spans.sort()
        s, r, o = forms[key]
        cur_s, cur_e = spans[0]
        for t0, t1 in spans[1:]:
            if t0 <= cur_e + gap_sec:
                cur_e = max(cur_e, t1)
            else:
                out.append((s, r, o, cur_s, cur_e))
                cur_s, cur_e = t0, t1
        out.append((s, r, o, cur_s, cur_e))
    return out


def validate_quintuples(quints: List[Quintuple]) -> List[Quintuple]:
    seen: set = set()
    cleaned: List[Quintuple] = []
    for s, r, o, t0, t1 in quints:
        s = re.sub(r"\s+", " ", s).strip()
        r = re.sub(r"\s+", " ", r).strip()
        o = re.sub(r"\s+", " ", o).strip()
        if not s or not r or not o:
            continue
        if len(s) > 100 or len(r) > 80 or len(o) > 100:
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


def format_quintuples_as_python_list(quints: List[Quintuple]) -> str:
    if not quints:
        return "[]"
    lines = ["["]
    for s, r, o, t0, t1 in quints:
        lines.append(f"  ({repr(s)}, {repr(r)}, {repr(o)}, {float(t0):.2f}, {float(t1):.2f}),")
    lines.append("]")
    return "\n".join(lines)


def quintuples_to_triplets(quints: List[Quintuple]) -> List[Triplet]:
    return [(s, r, o) for s, r, o, _t0, _t1 in quints]


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
    """Build SRT content from a list of dicts: {start, end, triplets}.

    The 'triplets' entry may hold either 3-tuples (subject, relation, object)
    or 5-tuple quintuples (subject, relation, object, start_sec, end_sec).
    Quintuples are downcast to 3-tuples for the SRT body since the SRT cue
    timing itself already pins each block to its segment window.
    """
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        raw = seg.get("triplets", [])
        trips: List[Triplet] = []
        for t in raw:
            if len(t) >= 3:
                trips.append((t[0], t[1], t[2]))
        trips_str = format_as_python_list(trips)
        lines += [
            str(i),
            f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}",
            trips_str,
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders — each returns a {"system": str, "user": str} pair.
# ---------------------------------------------------------------------------

PromptPair = Dict[str, str]


def _render_pair(
    name: str,
    override: Optional[PromptPair],
    variables: Dict[str, "Any"],
) -> PromptPair:
    """Render a {system, user} prompt pair, applying any per-call override.

    The override is a partial {"system"?, "user"?} dict. Missing halves fall
    back to the stored template. Unsubstitutable placeholders in an override
    are preserved verbatim by ``_safe_format``.
    """
    from .prompts import get_store, _safe_format
    base = get_store().get(name)
    out = {
        "system": _safe_format(base["system"], variables),
        "user":   _safe_format(base["user"],   variables),
    }
    if not override:
        return out
    if override.get("system") is not None:
        out["system"] = _safe_format(override["system"], variables)
    if override.get("user") is not None:
        out["user"] = _safe_format(override["user"], variables)
    return out


def build_scenegraph_prompt(
    transcript_text: str = "", mode: str = "high", user_text: str = "",
    template_override: Optional[PromptPair] = None,
) -> PromptPair:
    transcript_section = f"\nTRANSCRIPT:\n{transcript_text}" if transcript_text else ""
    user_text_section = f"\nCONTEXT:\n{user_text}" if user_text else ""
    return _render_pair(
        f"scenegraph_visual_{mode}",
        template_override,
        {"transcript_section": transcript_section,
         "user_text_section": user_text_section},
    )


def build_text_only_scenegraph_prompt(
    text: str, mode: str = "high", template_override: Optional[PromptPair] = None,
) -> PromptPair:
    return _render_pair(
        f"scenegraph_text_{mode}",
        template_override,
        {"text": text},
    )


def build_video_segment_prompt(
    segment_duration_sec: float,
    transcript_text: str = "",
    mode: str = "high",
    user_text: str = "",
    template_override: Optional[PromptPair] = None,
) -> PromptPair:
    """Build the {system, user} pair for a video clip.

    Instructions live in ``system``; the duration, ASR transcript, and
    user-supplied context live in ``user``. The MLLM is asked to emit
    5-tuples with temporal grounding (start_sec, end_sec) within the clip.
    Used for ASR-segment, temporal-window, and whole-video (single-clip) paths.
    """
    transcript_section = f"\nTRANSCRIPT:\n{transcript_text}" if transcript_text else ""
    user_text_section = f"\nCONTEXT:\n{user_text}" if user_text else ""
    return _render_pair(
        f"scenegraph_video_{mode}",
        template_override,
        {"segment_duration_sec": segment_duration_sec,
         "transcript_section": transcript_section,
         "user_text_section": user_text_section},
    )


def build_normalize_prompt(trips: List[Triplet]) -> PromptPair:
    lines = "\n".join(f"- ({s}, {r}, {o})" for s, r, o in trips)
    return _render_pair("normalize", None, {"triplets": lines})


def build_identify_subjects_prompt(transcript_text: str, user_text: str = "") -> PromptPair:
    """Prompt for the short who-is-who / who-is-speaking pre-pass (frames + ASR)."""
    t = (transcript_text or "").strip() or "(no speech detected)"
    if user_text:
        t = f"{t}\n\nADDITIONAL CONTEXT:\n{user_text}"
    return _render_pair("identify_subjects", None, {"transcript": t})


def build_canonicalize_entities_prompt(quints: List[Quintuple]) -> PromptPair:
    """Prompt the model for a compact {surface_form: canonical_name} entity map.

    The whole graph is given as context (input), but the model only emits a small
    JSON map (output) — so it never truncates the way re-emitting the full graph
    does. The map is applied deterministically by ``apply_entity_map``.
    """
    facts = "\n".join(f"- ({s}, {r}, {o})" for s, r, o, _t0, _t1 in quints)
    nodes: List[str] = []
    seen: set = set()
    for s, r, o, _t0, _t1 in quints:
        for n in (s, o):
            if n.lower() not in seen:
                seen.add(n.lower())
                nodes.append(n)
    mentions = "\n".join(f"- {n}" for n in nodes)
    return _render_pair("canonicalize_entities", None, {"facts": facts, "mentions": mentions})


def parse_entity_map(text: str) -> Dict[str, str]:
    """Parse a JSON object of {surface_form: canonical_name} from model output.

    Tries strict JSON first, then a forgiving regex over "key": "value" pairs so
    markdown fences, trailing commas, comments, or a truncated closing brace do
    not lose the whole map.
    """
    import json

    raw = text.strip()
    body = raw
    if "{" in body and "}" in body:
        body = body[body.find("{") : body.rfind("}") + 1]
    out: Dict[str, str] = {}
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    out[k] = v.strip()
    except Exception:
        pass
    if out:
        return out
    # Regex fallback: capture every "key": "value" pair from the raw text.
    for k, v in re.findall(r'"((?:[^"\\]|\\.)*)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw):
        k2 = k.replace('\\"', '"').strip()
        v2 = v.replace('\\"', '"').strip()
        if k2 and v2:
            out[k2] = v2
    return out


def apply_entity_map(quints: List[Quintuple], mapping: Dict[str, str]) -> List[Quintuple]:
    """Rewrite subject/object surface forms using a case-insensitive canonical map."""
    m = {k.lower(): v for k, v in mapping.items()}
    out: List[Quintuple] = []
    for s, r, o, t0, t1 in quints:
        out.append((m.get(s.lower(), s), r, m.get(o.lower(), o), t0, t1))
    return out


# Vague placeholder nodes that are never a real, groundable entity. Rows whose
# subject or object is one of these are dropped deterministically.
_PRONOUN_NODES = {
    "they", "them", "it", "he", "she", "someone", "somebody", "something",
    "this", "that", "these", "those", "we", "us", "you", "i", "one", "ones",
}


def drop_ungrounded_rows(quints: List[Quintuple]) -> List[Quintuple]:
    """Drop rows whose subject or object is a bare pronoun / vague placeholder."""
    out: List[Quintuple] = []
    for s, r, o, t0, t1 in quints:
        if s.strip().lower() in _PRONOUN_NODES or o.strip().lower() in _PRONOUN_NODES:
            continue
        out.append((s, r, o, t0, t1))
    return out


_LEVEL_RULE_HIGH = (
    "- LOW-LEVEL TRIVIA (this is a HIGH-LEVEL graph): remove trivial physical / appearance "
    "details that carry no narrative, identifying, or contextual significance — e.g. plain "
    "clothing colors (\"wears black shirt\"), generic posture, mundane object colors. KEEP "
    "attributes that identify or contextualize: a name, role/title, nationality/community, "
    "uniform, religious dress, a held sign/weapon/flag, on-screen text, or location."
)


def build_quality_filter_prompt(quints: List[Quintuple], mode: str = "high") -> PromptPair:
    """Prompt the model for a compact list of row numbers (1-based) to remove."""
    numbered = "\n".join(
        f"{i}. ({s}, {r}, {o})" for i, (s, r, o, _t0, _t1) in enumerate(quints, 1)
    )
    level_rule = _LEVEL_RULE_HIGH if mode == "high" else ""
    return _render_pair(
        "quality_filter", None,
        {"numbered": numbered, "mode": mode, "level_rule": level_rule},
    )


def parse_int_list(text: str) -> List[int]:
    """Parse a JSON / Python list of integers from model output."""
    text = text.strip()
    if "[" in text and "]" in text:
        text = text[text.find("[") : text.rfind("]") + 1]
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return [int(n) for n in re.findall(r"-?\d+", text)]
    if isinstance(obj, (list, tuple, set)):
        out: List[int] = []
        for x in obj:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
    return []


def drop_rows_by_number(quints: List[Quintuple], drop_numbers: List[int]) -> List[Quintuple]:
    """Drop rows by 1-based index (as emitted by build_quality_filter_prompt)."""
    drop = {n for n in drop_numbers if 1 <= n <= len(quints)}
    if not drop:
        return quints
    return [q for i, q in enumerate(quints, 1) if i not in drop]


def build_normalize_quintuples_prompt(quints: List[Quintuple], mode: str = "high") -> PromptPair:
    lines = "\n".join(
        f"- ({s}, {r}, {o}, {float(t0):.2f}, {float(t1):.2f})"
        for s, r, o, t0, t1 in quints
    )
    level_rule = _LEVEL_RULE_HIGH if mode == "high" else ""
    return _render_pair("normalize_quintuples", None,
                        {"quintuples": lines, "level_rule": level_rule})


def build_validation_prompt(claim: str, facts: List[str]) -> PromptPair:
    facts_block = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(facts))
    system = (
        "You are a fact-checker. Given FACTS from a trusted database, "
        "evaluate the CLAIM. Respond with JSON only (no markdown): "
        '{"verdict": "TRUE" | "FALSE" | "UNCERTAIN" | "NOT_COVERED", '
        '"confidence": "HIGH" | "MEDIUM" | "LOW", '
        '"explanation": "one or two sentences"}.'
    )
    user = f"FACTS:\n{facts_block}\n\nCLAIM: {claim}\n"
    return {"system": system, "user": user}
