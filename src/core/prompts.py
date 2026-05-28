"""Named prompt templates with optional user overrides persisted to disk.

Six template slots cover every MLLM call made by the system.
Each template uses {variable} placeholders for dynamic content;
available variables are documented in _META.

Custom templates are loaded from / saved to {db_dir}/prompts.json so they
survive container restarts (the file lives on the DATA_DIR volume).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("pride.prompts")


# ── Default templates ─────────────────────────────────────────────────────────
# Defined inside a function so jurigged can hot-patch them without a restart.
# Use {variable_name} for dynamic values. Escape a literal brace with {{ or }}.

def _defaults() -> Dict[str, str]:
    return {

        "scenegraph_visual_high": """\
You are an expert multimodal scene graph extractor focused on high-level, semantically meaningful relationships.

Analyze the provided video frames and extract the most important relationships between entities.

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
]{transcript_section}{user_text_section}

Output scene graph triplets:
""",

        "scenegraph_visual_low": """\
You are a visual scene graph extractor focused on concrete, observable relationships.

Analyze the provided video frames and extract the most important physical, spatial, and visual relationships.

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
]{transcript_section}{user_text_section}

Output scene graph triplets:
""",

        "scenegraph_text_high": """\
You are a knowledge extraction system building a high-level SCENE GRAPH from text.

Extract the most important entities and relationships as (subject, relation, object) triplets.

Rules:
1. Focus on high-level, semantically meaningful relationships: events, decisions, actions, affiliations, oppositions, and agreements.
2. Prefer named entities as subjects and objects: specific people, organizations, countries, policies, and events.
3. Use informative relation phrases that capture the nature of the interaction: "opposes", "negotiates with", "signs", "accuses", "announces", "supports", "leads", "controls", "threatens", "allies with".
4. Be selective — prioritize relationships that carry informational value. Avoid trivial or obvious facts.
5. OUTPUT: ONLY a Python list of 3-tuples. No extra text.

TEXT:
{text}

Output:
""",

        "scenegraph_text_low": """\
You are a knowledge extraction system building a visual SCENE GRAPH from text.

Extract the most important concrete, physical relationships described in the text as (subject, relation, object) triplets.

Rules:
1. Focus on physical/spatial/visual relationships: positions, actions, attributes, and interactions between tangible entities.
2. Subjects and objects should be concrete physical entities: people (generic labels OK), objects, animals, vehicles, places.
3. Use physical relation phrases: "sits on", "holds", "stands next to", "wears", "carries", "opens", "is located in", "runs toward".
4. Avoid abstract political, social, or conceptual relationships.
5. OUTPUT: ONLY a Python list of 3-tuples. No extra text.

TEXT:
{text}

Output:
""",

        "normalize": """\
Clean and normalize these scene-graph triplets:

- Make entity naming consistent across all triplets.
- Make nodes atomic (simple nouns/named entities).
- Remove exact and near-duplicate triplets.
- Do NOT invent new entities or relations.
- OUTPUT: ONLY a Python list of (subject, relation, object) 3-tuples. No extra text.

Triplets:
{triplets}

Output:
""",

        "validation": """\
You are a fact-checker with access to a trusted database.

RETRIEVED FACTS from database "{database}":
{facts_block}

Analyze {input_desc} and report on its factual accuracy based solely on the retrieved facts above.{context_block}

Write a clear, natural-language fact-check report covering:
- What information is SUPPORTED by the retrieved facts (cite which fact)
- What appears INACCURATE or CONTRADICTED (cite which fact)
- What CANNOT BE DETERMINED from the available facts
- What is apparently false based on your prior knowledge

Be specific and concise. Do not invent information not present in the facts.\
""",
    }


# Human-readable metadata — shown in the UI and returned by GET /api/prompts
_META: Dict[str, Dict[str, Any]] = {
    "scenegraph_visual_high": {
        "label": "Scene Graph — Visual, High-Level Mode",
        "description": (
            "Used when extracting scene graphs from video or image input with mode='high'. "
            "Also applied to individual temporal segments of long videos."
        ),
        "variables": {
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
            "Used when extracting scene graphs from video or image input with mode='low'. "
            "Also applied to individual temporal segments of long videos."
        ),
        "variables": {
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
        "variables": {
            "text": "The input text submitted by the user.",
        },
    },
    "scenegraph_text_low": {
        "label": "Scene Graph — Text-Only, Low-Level Mode",
        "description": "Used for text-only inputs with mode='low'.",
        "variables": {
            "text": "The input text submitted by the user.",
        },
    },
    "normalize": {
        "label": "Entity Normalization Pass",
        "description": (
            "Second MLLM pass that deduplicates and normalizes entity names across all triplets. "
            "Runs after scene graph extraction when 'normalize_pass' is enabled in config."
        ),
        "variables": {
            "triplets": "Bullet list of (subject, relation, object) triplets from the first pass.",
        },
    },
    "validation": {
        "label": "Fact Validation",
        "description": (
            "Used by the validation pipeline. The model receives retrieved facts from the "
            "database and writes a fact-check report on the submitted input."
        ),
        "variables": {
            "database":      "Name of the fact database being queried.",
            "facts_block":   "Numbered list of retrieved facts, e.g. '  1. Fact one\\n  2. Fact two'.",
            "input_desc":    "Natural-language description of input types, e.g. 'the video frames and the audio transcript'.",
            "context_block": "Transcript and/or submitted text block appended after the instruction. Empty string if neither was provided.",
        },
    },
}


# ── PromptStore ───────────────────────────────────────────────────────────────

class PromptStore:
    """Manages named prompt templates; custom overrides persist to disk."""

    def __init__(self, storage_path: str):
        self._path = storage_path
        self._custom: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self._path):
            try:
                with open(self._path) as f:
                    self._custom = json.load(f)
                log.info("Loaded %d custom prompt(s) from %s.", len(self._custom), self._path)
            except Exception as e:
                log.warning("Could not load custom prompts from %s: %s", self._path, e)
                self._custom = {}

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._custom, f, indent=2)
        except Exception as e:
            log.error("Could not save custom prompts to %s: %s", self._path, e)

    def names(self) -> List[str]:
        return list(_defaults().keys())

    def get(self, name: str) -> str:
        defs = _defaults()
        if name not in defs:
            raise KeyError(name)
        return self._custom.get(name, defs[name])

    def get_default(self, name: str) -> str:
        defs = _defaults()
        if name not in defs:
            raise KeyError(name)
        return defs[name]

    def is_custom(self, name: str) -> bool:
        return name in self._custom

    def set(self, name: str, template: str) -> None:
        if name not in _defaults():
            raise KeyError(name)
        self._custom[name] = template
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

    def render(self, name: str, **kwargs: Any) -> str:
        template = self.get(name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            log.error("Prompt '%s' references unknown variable %s — fix the template.", name, e)
            raise


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
