"""Named prompt templates with optional user overrides persisted to disk.

Each prompt slot has TWO parts:
- ``system`` — the role / rules / examples / output format (static instructions
  for the MLLM, sent in the system role of the chat template).
- ``user``   — the request-specific input (transcript, user text, duration,
  triplets to refine, etc.), sent in the user role.

Templates use ``{variable}`` placeholders for dynamic content; available
variables and which part they belong to are documented in ``_META``.

Custom overrides are stored as ``{name: {"system": str, "user": str}}`` in
``{db_dir}/prompts.json``. Legacy single-string entries are still loaded for
backwards compatibility (treated as user-only with the default system part).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("pride.prompts")

# Each rendered prompt is returned as this {"system", "user"} dict.
PromptPair = Dict[str, str]


# ── Default templates ─────────────────────────────────────────────────────────
# Defined inside a function so jurigged can hot-patch them without a restart.
# Use {variable_name} for dynamic values. Escape a literal brace with {{ or }}.

def _defaults() -> Dict[str, PromptPair]:
    return {

        "scenegraph_visual_high": {
            "system": """\
You are an expert multimodal scene graph extractor focused on high-level, semantically meaningful relationships.

Your task is to analyze an image or a small set of video frames provided by the user and extract the most important relationships between entities.

Rules:
1. PRIORITIZE HIGH-LEVEL EVENTS AND ACTIONS: capture what is *happening* — political decisions, negotiations, conflicts, endorsements, protests, announcements — not low-level physical positions.
2. SUBJECTS AND OBJECTS must be named entities when identifiable (specific people by name, organizations, countries, policies, events) or meaningful roles ("protesters", "delegation", "opposition leader"). Avoid vague descriptors like "man" or "person" unless the identity cannot be determined.
3. RELATIONS must convey high-level semantic meaning: "negotiates with", "signs", "accuses", "endorses", "protests against", "announces", "condemns", "meets with", "leads", "opposes", "imposes sanctions on". Only use physical/spatial relations when they carry significant narrative meaning.
4. One triplet per distinct relationship. Prioritize informationally valuable relationships.
5. CONSISTENT NAMING: use the same identifier for the same entity across all triplets.
6. OUTPUT: ONLY a Python list of (subject, relation, object) 3-tuples. No markdown, no explanation.

Example of GOOD output (high-level, semantically informative):
[
  ("US President", "negotiates", "nuclear deal"),
  ("US", "imposes sanctions on", "Iran"),
  ("NATO", "responds to", "Russian invasion"),
  ("protesters", "demand release of", "political prisoners"),
  ("Secretary of State", "meets with", "Chinese counterpart"),
  ("EU", "condemns", "US tariff policy"),
]

Example of BAD output (trivial, low-level — do NOT do this):
[
  ("man", "stands next to", "microphone"),
  ("woman", "wears", "blue jacket"),
  ("person", "talks to", "camera"),
]
""",
            "user": """\
{transcript_section}{user_text_section}

Output scene graph triplets:
""",
        },

        "scenegraph_visual_low": {
            "system": """\
You are a visual scene graph extractor focused on concrete, observable relationships.

Your task is to analyze an image or a small set of video frames provided by the user and extract the most important physical, spatial, and visual relationships.

Rules:
1. FOCUS ON VISUAL/PHYSICAL RELATIONSHIPS: how objects and people are positioned, what actions are being performed, what attributes (color, size, state) are notable.
2. SUBJECTS AND OBJECTS must be concrete physical entities: people (by appearance or role if unnamed), objects, animals, vehicles, locations. Generic labels like "man", "woman", "dog", "car" are fine.
3. RELATIONS must be physical/visual/spatial: "sits on", "holds", "stands next to", "wears", "runs toward", "places on", "looks at", "carries", "opens", "is located in", "is part of", "is colored", "points at".
4. AVOID abstract political or social concepts unless they are physically depicted (e.g., a flag or sign in the frame).
5. One triplet per distinct observable relationship. Prioritize visually prominent or action-defining ones.
6. CONSISTENT NAMING: use the same label for the same entity across all triplets.
7. OUTPUT: ONLY a Python list of (subject, relation, object) 3-tuples. No markdown, no explanation.

Example of GOOD output (concrete, visual):
[
  ("man", "sits on", "chair"),
  ("woman", "holds", "coffee cup"),
  ("dog", "runs toward", "ball"),
  ("child", "wears", "red backpack"),
  ("car", "is parked in front of", "building"),
  ("cat", "lies on top of", "table"),
]

Example of BAD output (abstract — do NOT do this):
[
  ("government", "announces", "new policy"),
  ("CEO", "negotiates", "merger"),
  ("protesters", "demand", "reform"),
]
""",
            "user": """\
{transcript_section}{user_text_section}

Output scene graph triplets:
""",
        },

        "scenegraph_video_high": {
            "system": """\
You are an expert multimodal scene graph extractor focused on high-level, semantically meaningful relationships.

Your task is to analyze a video clip (and, when provided, its audio transcript or extra context) given by the user, and extract the most important relationships between entities — each grounded in a temporal window.

Rules:
1. PRIORITIZE HIGH-LEVEL EVENTS AND ACTIONS: capture what is *happening* — political decisions, negotiations, conflicts, endorsements, protests, announcements — not low-level physical positions.
2. SUBJECTS AND OBJECTS must be named entities when identifiable (specific people by name, organizations, countries, policies, events) or meaningful roles ("protesters", "delegation", "opposition leader"). Avoid vague descriptors like "man" or "person" unless the identity cannot be determined.
3. RELATIONS must convey high-level semantic meaning: "negotiates with", "signs", "accuses", "endorses", "protests against", "announces", "condemns", "meets with", "leads", "opposes", "imposes sanctions on". Only use physical/spatial relations when they carry significant narrative meaning.
4. TEMPORAL GROUNDING: for each relationship, estimate the start and end SECOND of the video at which the relationship is observed. Both values must be floats with 0.0 <= start_sec <= end_sec <= video duration (the duration is given in the user message).
5. CONSISTENT NAMING: use the same identifier for the same entity across all quintuples.
6. Aim for roughly 5–25 high-quality, non-redundant quintuples for a typical clip; more for long videos with many distinct events. Quality over quantity — do not invent or repeat.
7. OUTPUT: ONLY a Python list of (subject, relation, object, start_sec, end_sec) 5-tuples. No markdown, no explanation.

Example of GOOD output (high-level, semantically informative, with times in seconds):
[
  ("US President", "negotiates", "nuclear deal", 0.0, 3.5),
  ("US", "imposes sanctions on", "Iran", 4.0, 7.2),
  ("NATO", "responds to", "Russian invasion", 7.5, 10.0),
  ("protesters", "demand release of", "political prisoners", 0.5, 6.0),
]

Example of BAD output (trivial, low-level — do NOT do this):
[
  ("man", "stands next to", "microphone", 0.0, 2.0),
  ("woman", "wears", "blue jacket", 0.0, 5.0),
]
""",
            "user": """\
The video is {segment_duration_sec:.2f} seconds long.{transcript_section}{user_text_section}

Output scene graph quintuples:
""",
        },

        "scenegraph_video_low": {
            "system": """\
You are a visual scene graph extractor focused on concrete, observable relationships.

Your task is to analyze a video clip (and, when provided, its audio transcript or extra context) given by the user, and extract the most important physical, spatial, and visual relationships — each grounded in a temporal window.

Rules:
1. FOCUS ON VISUAL/PHYSICAL RELATIONSHIPS: how objects and people are positioned, what actions are being performed, what attributes (color, size, state) are notable.
2. SUBJECTS AND OBJECTS must be concrete physical entities: people (by appearance or role if unnamed), objects, animals, vehicles, locations. Generic labels like "man", "woman", "dog", "car" are fine.
3. RELATIONS must be physical/visual/spatial: "sits on", "holds", "stands next to", "wears", "runs toward", "places on", "looks at", "carries", "opens", "is located in", "is part of", "is colored", "points at".
4. TEMPORAL GROUNDING: for each relationship, estimate the start and end SECOND of the video at which the relationship is observed. Both values must be floats with 0.0 <= start_sec <= end_sec <= video duration (the duration is given in the user message).
5. AVOID abstract political or social concepts unless they are physically depicted (e.g., a flag or sign in the frame).
6. CONSISTENT NAMING: use the same label for the same entity across all quintuples.
7. Aim for roughly 5–25 high-quality, non-redundant quintuples for a typical clip; more for long videos with many distinct events.
8. OUTPUT: ONLY a Python list of (subject, relation, object, start_sec, end_sec) 5-tuples. No markdown, no explanation.

Example of GOOD output (concrete, visual, with times in seconds):
[
  ("man", "sits on", "chair", 0.0, 4.2),
  ("woman", "holds", "coffee cup", 1.0, 3.5),
  ("dog", "runs toward", "ball", 2.0, 5.0),
  ("car", "is parked in front of", "building", 0.0, 8.0),
]

Example of BAD output (abstract — do NOT do this):
[
  ("government", "announces", "new policy", 0.0, 5.0),
  ("CEO", "negotiates", "merger", 0.0, 5.0),
]
""",
            "user": """\
The video is {segment_duration_sec:.2f} seconds long.{transcript_section}{user_text_section}

Output scene graph quintuples:
""",
        },

        "scenegraph_text_high": {
            "system": """\
You are a knowledge extraction system building a high-level SCENE GRAPH from text.

Extract the most important entities and relationships from the user's text as (subject, relation, object) triplets.

Rules:
1. Focus on high-level, semantically meaningful relationships: events, decisions, actions, affiliations, oppositions, and agreements.
2. Prefer named entities as subjects and objects: specific people, organizations, countries, policies, and events.
3. Use informative relation phrases that capture the nature of the interaction: "opposes", "negotiates with", "signs", "accuses", "announces", "supports", "leads", "controls", "threatens", "allies with".
4. Be selective — prioritize relationships that carry informational value. Avoid trivial or obvious facts.
5. OUTPUT: ONLY a Python list of 3-tuples. No extra text.
""",
            "user": """\
TEXT:
{text}

Output:
""",
        },

        "scenegraph_text_low": {
            "system": """\
You are a knowledge extraction system building a visual SCENE GRAPH from text.

Extract the most important concrete, physical relationships described in the user's text as (subject, relation, object) triplets.

Rules:
1. Focus on physical/spatial/visual relationships: positions, actions, attributes, and interactions between tangible entities.
2. Subjects and objects should be concrete physical entities: people (generic labels OK), objects, animals, vehicles, places.
3. Use physical relation phrases: "sits on", "holds", "stands next to", "wears", "carries", "opens", "is located in", "runs toward".
4. Avoid abstract political, social, or conceptual relationships.
5. OUTPUT: ONLY a Python list of 3-tuples. No extra text.
""",
            "user": """\
TEXT:
{text}

Output:
""",
        },

        "normalize": {
            "system": """\
You refine scene-graph triplets. Apply ALL of the following passes.

OUTPUT FORMAT (strict — your response is parsed)
- Emit ONLY a Python list of 3-tuples, each shaped exactly as: ("subject", "relation", "object").
- Every element must be a 3-tuple of strings — never 2-tuples, never 4-tuples, never dicts, never trailing booleans or notes.
- Use double-quoted strings. Escape any internal double-quote as \\".
- The list must be terminated with `]`. No markdown, no code fences, no commentary before or after.
- If after refinement nothing survives, output exactly `[]`.

ENTITY NORMALIZATION
- Pick one canonical surface form per real-world entity and rewrite the others to match.
- Make nodes atomic (simple nouns or named entities; split compound nodes when they bundle independent facts).
- Resolve obvious coreference ("he", "the president", "Mr. Smith" → the canonical name) only when the referent is unambiguous from the surrounding triplets.

RELATION CLEANUP
- Rewrite vague or generic relations into concise, informative ones ("is" / "has" / "does" → a more specific verb when clear; "and" / "with" → a precise relation).
- Keep relations short (1–4 words is ideal). No flowery wording, no punctuation, no quotes inside the relation.

DEDUPLICATION & REDUNDANCY REMOVAL
- Drop exact duplicates and triplets that restate the same fact under slightly different wording.
- When two triplets cover the same relationship, keep the more informative one.
- Drop triplets trivially implied by another (e.g. if (A, "is president of", "US") is present, drop (A, "is", "politician")).

QUALITY FILTER
- Drop low-information triplets: tautologies, vague descriptors ("man stands", "thing exists"), and anything you would not consider a fact worth recording.
- Drop triplets whose subject or object cannot be identified confidently from the input.

HARD CONSTRAINTS
- Do NOT invent new entities, relations, or facts not supported by the input.
- Do NOT add explanation, headers, or markdown. The first character of your response must be `[`, the last must be `]`.

IN-CONTEXT EXAMPLE

Input triplets:
- (Joseph R. Biden Jr., is, US President)
- (Biden, signs, infrastructure bill)
- (President Biden, signed, the infrastructure bill)
- (he, met with, Xi Jinping)
- (Xi, leads, China)
- (Xi Jinping, is, leader of People's Republic of China)
- (man, stands at, podium)
- (US, imposes sanctions on, Russia)
- (United States, sanctions, Russian Federation)

Output:
[
  ("Joe Biden", "is president of", "United States"),
  ("Joe Biden", "signs", "infrastructure bill"),
  ("Joe Biden", "meets with", "Xi Jinping"),
  ("Xi Jinping", "leads", "China"),
  ("United States", "imposes sanctions on", "Russia"),
]

Notice in the example: the two "signs / signed infrastructure bill" entries were merged; "he" was resolved to "Joe Biden"; "Xi" and "Xi Jinping" were unified; the trivially-implied "Xi is leader of People's Republic of China" was dropped (already covered by the leads-China triplet); the low-information "man stands at podium" was dropped; "US"/"United States" and "Russia"/"Russian Federation" were normalized.
""",
            "user": """\
Triplets:
{triplets}

Output:
""",
        },

        "normalize_quintuples": {
            "system": """\
You refine scene-graph quintuples (subject, relation, object, start_sec, end_sec). Apply ALL of the following passes.

OUTPUT FORMAT (strict — your response is parsed)
- Emit ONLY a Python list of 5-tuples, each shaped exactly as: ("subject", "relation", "object", start_sec, end_sec).
- Every element must be a 5-tuple — never 3-tuples or 4-tuples, never dicts, never trailing fields.
- The first three positions are double-quoted strings. The last two are unquoted floats (e.g. 3.5, not "3.5"); use 0.0 if the input had an integer.
- Use double-quoted strings. Escape any internal double-quote as \\".
- The list must be terminated with `]`. No markdown, no code fences, no commentary before or after.
- If after refinement nothing survives, output exactly `[]`.

ENTITY NORMALIZATION
- Pick one canonical surface form per real-world entity and rewrite the others to match.
- Make nodes atomic (simple nouns or named entities; split compound nodes when they bundle independent facts).
- Resolve obvious coreference ("he", "the president", "Mr. Smith" → the canonical name) only when the referent is unambiguous from the surrounding quintuples.

RELATION CLEANUP
- Rewrite vague or generic relations into concise, informative ones.
- Keep relations short (1–4 words is ideal). No flowery wording, no punctuation, no quotes inside the relation.

DEDUPLICATION & REDUNDANCY REMOVAL
- Drop exact duplicates and quintuples that restate the same fact under slightly different wording.
- When two quintuples cover the same relationship in overlapping or adjacent time windows, MERGE them into a single quintuple that spans [min(start_sec), max(end_sec)].
- Drop quintuples trivially implied by another (e.g. if (A, "is president of", "US", t0, t1) is present, drop (A, "is", "politician", t0, t1)).

QUALITY FILTER
- Drop low-information quintuples: tautologies, vague descriptors ("man stands"), and anything you would not consider a fact worth recording.
- Drop quintuples whose subject or object cannot be identified confidently from the input.

TIMESTAMPS
- PRESERVE the original start_sec and end_sec for surviving quintuples; do not move or invent times.
- When merging, use the UNION of the input time windows: start_sec = min over merged rows, end_sec = max over merged rows.
- start_sec must be <= end_sec. Always emit floats (e.g. 4.00, not 4).

HARD CONSTRAINTS
- Do NOT invent new entities, relations, facts, or times not supported by the input.
- Do NOT add explanation, headers, or markdown. The first character of your response must be `[`, the last must be `]`.

IN-CONTEXT EXAMPLE

Input quintuples:
- (Joseph R. Biden Jr., is, US President, 0.00, 2.50)
- (Biden, signs, infrastructure bill, 3.00, 5.20)
- (President Biden, signed, the infrastructure bill, 4.80, 6.10)
- (he, met with, Xi Jinping, 7.00, 9.50)
- (Xi, leads, China, 7.50, 9.00)
- (Xi Jinping, is, leader of People's Republic of China, 7.50, 9.00)
- (man, stands at, podium, 0.00, 2.50)
- (US, imposes sanctions on, Russia, 10.00, 12.40)
- (United States, sanctions, Russian Federation, 10.20, 12.80)

Output:
[
  ("Joe Biden", "is president of", "United States", 0.00, 2.50),
  ("Joe Biden", "signs", "infrastructure bill", 3.00, 6.10),
  ("Joe Biden", "meets with", "Xi Jinping", 7.00, 9.50),
  ("Xi Jinping", "leads", "China", 7.50, 9.00),
  ("United States", "imposes sanctions on", "Russia", 10.00, 12.80),
]

Notice in the example: the two overlapping "signs / signed infrastructure bill" entries were merged into [3.00, 6.10] (union of their windows); "he" was resolved to "Joe Biden"; "Xi" and "Xi Jinping" were unified; the trivially-implied "Xi Jinping is leader of People's Republic of China" was dropped; the low-information "man stands at podium" was dropped; "US"/"United States" and "Russia"/"Russian Federation" were normalized and their overlapping windows merged.
""",
            "user": """\
Quintuples:
{quintuples}

Output:
""",
        },

        "validation": {
            "system": """\
You are a fact-checker with access to a trusted database. Given retrieved facts and an input submitted by the user, analyze the input's factual accuracy based SOLELY on the retrieved facts (plus your prior knowledge, used only to flag claims as apparently false).

Write a clear, natural-language fact-check report covering:
- What information is SUPPORTED by the retrieved facts (cite which fact)
- What appears INACCURATE or CONTRADICTED (cite which fact)
- What CANNOT BE DETERMINED from the available facts
- What is apparently false based on your prior knowledge

Be specific and concise. Do not invent information not present in the facts.
""",
            "user": """\
RETRIEVED FACTS from database "{database}":
{facts_block}

Please analyze {input_desc}.{context_block}
""",
        },
    }


# Human-readable metadata — shown in the UI and returned by GET /api/prompts.
# `variables` is split into the part of the prompt each placeholder belongs to.
_META: Dict[str, Dict[str, Any]] = {
    "scenegraph_visual_high": {
        "label": "Scene Graph — Visual, High-Level Mode",
        "description": (
            "Used when extracting scene graphs from IMAGE input with mode='high'."
        ),
        "system_variables": {},
        "user_variables": {
            "transcript_section": (
                "ASR transcript block, e.g. '\\nTRANSCRIPT:\\n...'. "
                "Empty string when no audio transcript is available."
            ),
            "user_text_section": (
                "User-provided context block, e.g. '\\nCONTEXT:\\n...'. "
                "Empty string when the user did not supply additional text."
            ),
        },
    },
    "scenegraph_visual_low": {
        "label": "Scene Graph — Visual, Low-Level Mode",
        "description": (
            "Used when extracting scene graphs from IMAGE input with mode='low'."
        ),
        "system_variables": {},
        "user_variables": {
            "transcript_section": (
                "ASR transcript block, e.g. '\\nTRANSCRIPT:\\n...'. "
                "Empty string when no audio transcript is available."
            ),
            "user_text_section": (
                "User-provided context block, e.g. '\\nCONTEXT:\\n...'. "
                "Empty string when the user did not supply additional text."
            ),
        },
    },
    "scenegraph_video_high": {
        "label": "Scene Graph — Video, High-Level Mode",
        "description": (
            "Used when extracting scene graphs from VIDEO input with mode='high'. "
            "Asks the MLLM to output 5-tuples with temporal grounding (start_sec, end_sec)."
        ),
        "system_variables": {},
        "user_variables": {
            "segment_duration_sec": (
                "Duration of the video clip shown to the MLLM, in seconds (float). "
                "The MLLM is asked to output times in [0.0, segment_duration_sec]."
            ),
            "transcript_section": (
                "ASR transcript block, e.g. '\\nTRANSCRIPT:\\n...'. "
                "Empty string when no audio transcript is available."
            ),
            "user_text_section": (
                "User-provided context block, e.g. '\\nCONTEXT:\\n...'. "
                "Empty string when the user did not supply additional text."
            ),
        },
    },
    "scenegraph_video_low": {
        "label": "Scene Graph — Video, Low-Level Mode",
        "description": (
            "Used when extracting scene graphs from VIDEO input with mode='low'. "
            "Asks the MLLM to output 5-tuples with temporal grounding (start_sec, end_sec)."
        ),
        "system_variables": {},
        "user_variables": {
            "segment_duration_sec": (
                "Duration of the video clip shown to the MLLM, in seconds (float). "
                "The MLLM is asked to output times in [0.0, segment_duration_sec]."
            ),
            "transcript_section": (
                "ASR transcript block, e.g. '\\nTRANSCRIPT:\\n...'. "
                "Empty string when no audio transcript is available."
            ),
            "user_text_section": (
                "User-provided context block, e.g. '\\nCONTEXT:\\n...'. "
                "Empty string when the user did not supply additional text."
            ),
        },
    },
    "scenegraph_text_high": {
        "label": "Scene Graph — Text-Only, High-Level Mode",
        "description": "Used for text-only inputs with mode='high'.",
        "system_variables": {},
        "user_variables": {
            "text": "The input text submitted by the user.",
        },
    },
    "scenegraph_text_low": {
        "label": "Scene Graph — Text-Only, Low-Level Mode",
        "description": "Used for text-only inputs with mode='low'.",
        "system_variables": {},
        "user_variables": {
            "text": "The input text submitted by the user.",
        },
    },
    "normalize": {
        "label": "Refinement Pass — Triplets",
        "description": (
            "Second MLLM pass that refines the first-round triplets: normalizes entity names, "
            "cleans up vague relations, removes duplicates and redundant restatements, and drops "
            "low-quality / trivially-implied facts."
        ),
        "system_variables": {},
        "user_variables": {
            "triplets": "Bullet list of (subject, relation, object) triplets from the first pass.",
        },
    },
    "normalize_quintuples": {
        "label": "Refinement Pass — Video Quintuples",
        "description": (
            "Second MLLM pass that refines the first-round VIDEO quintuples: normalizes entity "
            "names, cleans up vague relations, merges near-duplicates (taking the union of their "
            "time windows), drops low-quality / trivially-implied facts."
        ),
        "system_variables": {},
        "user_variables": {
            "quintuples": "Bullet list of (subject, relation, object, start_sec, end_sec) quintuples from the first pass.",
        },
    },
    "validation": {
        "label": "Fact Validation",
        "description": (
            "Used by the validation pipeline. The model receives retrieved facts from the "
            "database and writes a fact-check report on the submitted input."
        ),
        "system_variables": {},
        "user_variables": {
            "database":      "Name of the fact database being queried.",
            "facts_block":   "Numbered list of retrieved facts, e.g. '  1. Fact one\\n  2. Fact two'.",
            "input_desc":    "Natural-language description of input types, e.g. 'the video frames and the audio transcript'.",
            "context_block": "Transcript and/or submitted text block appended after the instruction. Empty string if neither was provided.",
        },
    },
}


def _coerce_to_pair(value: Any, default_pair: PromptPair) -> PromptPair:
    """Normalize a custom-stored value into a {system, user} pair.

    Accepts the legacy single-string format (treated as user-only override
    with the default system part), and the new dict-of-{system, user} format
    where either field may be omitted to inherit the default.
    """
    if isinstance(value, str):
        return {"system": default_pair["system"], "user": value}
    if isinstance(value, dict):
        return {
            "system": value.get("system", default_pair["system"]),
            "user":   value.get("user",   default_pair["user"]),
        }
    return dict(default_pair)


# ── PromptStore ───────────────────────────────────────────────────────────────

class PromptStore:
    """Manages named prompt templates; custom overrides persist to disk.

    Each slot has two halves: a static ``system`` instruction string and a
    request-shaped ``user`` template string. ``get`` / ``set`` / ``render``
    all operate on the pair as a unit.
    """

    def __init__(self, storage_path: str):
        self._path = storage_path
        self._custom: Dict[str, PromptPair] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
        except Exception as e:
            log.warning("Could not load custom prompts from %s: %s", self._path, e)
            return
        defs = _defaults()
        for name, value in raw.items():
            if name not in defs:
                continue
            self._custom[name] = _coerce_to_pair(value, defs[name])
        log.info("Loaded %d custom prompt(s) from %s.", len(self._custom), self._path)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._custom, f, indent=2)
        except Exception as e:
            log.error("Could not save custom prompts to %s: %s", self._path, e)

    def names(self) -> List[str]:
        return list(_defaults().keys())

    def get(self, name: str) -> PromptPair:
        defs = _defaults()
        if name not in defs:
            raise KeyError(name)
        if name in self._custom:
            return dict(self._custom[name])
        return dict(defs[name])

    def get_default(self, name: str) -> PromptPair:
        defs = _defaults()
        if name not in defs:
            raise KeyError(name)
        return dict(defs[name])

    def is_custom(self, name: str) -> bool:
        if name not in self._custom:
            return False
        defs = _defaults()[name]
        return self._custom[name] != defs

    def set(self, name: str, system: Optional[str] = None, user: Optional[str] = None) -> None:
        """Save a custom override. Unspecified halves fall back to the default."""
        defs = _defaults()
        if name not in defs:
            raise KeyError(name)
        pair = {
            "system": system if system is not None else defs[name]["system"],
            "user":   user   if user   is not None else defs[name]["user"],
        }
        self._custom[name] = pair
        self._save()

    def reset(self, name: str) -> None:
        if name not in _defaults():
            raise KeyError(name)
        self._custom.pop(name, None)
        self._save()

    def meta(self, name: str) -> Dict[str, Any]:
        if name not in _defaults():
            raise KeyError(name)
        return _META[name]

    def render(self, name: str, **kwargs: Any) -> PromptPair:
        """Substitute ``{variable}`` placeholders in both halves of the prompt.

        Variables that don't appear in a given half are ignored for that half
        (so callers can pass the union of system+user variables once).
        """
        pair = self.get(name)
        return {
            "system": _safe_format(pair["system"], kwargs),
            "user":   _safe_format(pair["user"],   kwargs),
        }


def _safe_format(template: str, kwargs: Dict[str, Any]) -> str:
    """``str.format``-style substitution that ignores unknown placeholders.

    Callers pass the union of variables required by either half of a prompt,
    so a key required by the user half may legitimately be absent from the
    system half (and vice versa). Missing keys are left as-is rather than
    raising.
    """
    try:
        return template.format(**kwargs)
    except KeyError:
        # Fallback: substitute only the keys the template actually references.
        import string
        formatter = string.Formatter()
        out = []
        for literal, field, fmt, conv in formatter.parse(template):
            out.append(literal)
            if field is None:
                continue
            if field in kwargs:
                value = kwargs[field]
                if conv:
                    value = formatter.convert_field(value, conv)
                out.append(formatter.format_field(value, fmt or ""))
            else:
                # Leave unresolved placeholder intact for transparency.
                out.append("{" + field + (("!" + conv) if conv else "") + ((":" + fmt) if fmt else "") + "}")
        return "".join(out)


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[PromptStore] = None


def get_store() -> PromptStore:
    if _store is None:
        raise RuntimeError("PromptStore not initialized — call init_store() first.")
    return _store


def init_store(db_dir: str) -> PromptStore:
    global _store
    _store = PromptStore(os.path.join(db_dir, "prompts.json"))
    return _store
