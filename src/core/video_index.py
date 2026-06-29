"""Structured, searchable index of scene-graph rows keyed by source video.

Each indexed video contributes one searchable *row* per scene-graph triplet /
quintuple. A row is rendered as a short natural-language string
("subject relation object") and embedded in the shared multimodal space with the
same encoder used by the validation pipeline. Because every row carries its
``video_id`` (and, for video sources, its ``start_sec`` / ``end_sec``) in
metadata, a free-text query — "man holding a flag in a market", "police", a
location, a person — retrieves the most relevant rows, which are then aggregated
back up to rank the videos they came from and to surface the matching moments.

Storage layout (under ``index_path``)
-------------------------------------
``chroma/``            ChromaDB persistent store; one collection per embedding
                       model (rows live here, one per triplet).
``videos.json``        Registry of per-video records (title, file, duration,
                       row count, …) for fast listing without scanning Chroma.
``media/<id>.<ext>``   The stored video file, served back for in-browser playback.

The per-model collection naming mirrors :class:`FactDatabase` so switching the
embedding model starts a fresh (empty) index rather than mixing incompatible
vector spaces; re-index videos after a model change.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from .embedder import MultimodalEmbedder

log = logging.getLogger("pride.video_index")

_REGISTRY_FILENAME = "videos.json"
_MAX_SLUG_LEN = 40
_MAX_COL_LEN = 63
# Minimum cosine similarity a row must have with the query to count as a match.
# Below this a video is considered irrelevant and excluded. Tunable per-request
# via the `min_score` search argument (0 disables filtering → ranks all videos).
_DEFAULT_MIN_SCORE = 0.30


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:_MAX_SLUG_LEN]


def _row_text(subject: str, relation: str, obj: str) -> str:
    """Render a triplet as the short string that gets embedded and searched."""
    return f"{subject} {relation} {obj}".strip()


# Coarse facet detection. These are deterministic heuristics (no extra model
# call): the relation predicate reveals location / action / text / framing rows,
# and a person/role noun in either entity flags a "people" row. A row may carry
# several facets (e.g. "man holding flag" → people + action). Unmatched rows get
# no facet and only appear under the unfiltered ("Any") search. Facets are stored
# as boolean metadata (``f_<facet>``) so ChromaDB ``where`` can filter on them
# alongside the vector search; recall stays heuristic, so it is a coarse sieve,
# not a guarantee.
FACET_KEYS = ("people", "location", "action", "text", "framing")

_SPATIAL = {
    "in", "at", "on", "near", "inside", "outside", "behind", "beside", "under",
    "above", "below", "by", "across", "around", "within", "into", "onto",
    "located", "positioned", "amid", "among", "amongst", "atop",
}
_SPATIAL_PHRASES = ("in front of", "next to", "close to", "located in",
                    "in the background", "on top of", "in the foreground")
_TEXT = {
    "reads", "read", "says", "said", "states", "stated", "displays", "displayed",
    "captioned", "caption", "labeled", "labelled", "written", "writes",
    "announces", "announced", "claims", "claimed", "quotes", "quoted",
    "headline", "subtitle", "titled", "text", "spells", "narrates",
}
_TEXT_PHRASES = ("on-screen text", "on screen text", "caption reads",
                 "text reads", "sign reads", "subtitle reads")
_FRAMING = {
    "portrays", "portray", "depicts", "depict", "frames", "targets", "target",
    "suggests", "suggest", "implies", "imply", "promotes", "promote", "blames",
    "blame", "accuses", "accuse", "demonizes", "glorifies", "criticizes",
    "praises", "mocks", "warns", "represents", "stereotypes", "vilifies",
}
_FRAMING_PHRASES = ("narrative targets", "frames as", "portrayed as", "depicted as")
_ACTION = {
    "holding", "holds", "hold", "carrying", "carries", "carry", "walking",
    "walks", "running", "runs", "speaking", "speaks", "speak", "talking",
    "talks", "shaking", "shakes", "waving", "waves", "pointing", "points",
    "throwing", "throws", "raising", "raises", "marching", "marches",
    "protesting", "protests", "standing", "stands", "sitting", "sits",
    "gesturing", "grabbing", "pushing", "pulling", "kicking", "jumping",
    "dancing", "singing", "clapping", "fighting", "attacking", "hugging",
    "kissing", "eating", "drinking", "driving", "riding", "flying", "chanting",
    "cheering", "gathering", "blocking", "burning", "shouting", "praying",
}
_PERSON = {
    "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child",
    "children", "kid", "kids", "people", "person", "persons", "crowd", "mob",
    "group", "groups", "protester", "protesters", "protestor", "protestors",
    "officer", "officers", "police", "policeman", "policemen", "soldier",
    "soldiers", "worker", "workers", "vendor", "vendors", "driver", "drivers",
    "leader", "leaders", "president", "minister", "ministers", "official",
    "officials", "citizen", "citizens", "immigrant", "immigrants", "migrant",
    "migrants", "community", "communities", "family", "families", "individual",
    "individuals", "resident", "residents", "demonstrator", "demonstrators",
    "indian", "indians", "chinese", "malay", "malays", "muslim", "muslims",
    "christian", "christians", "hindu", "hindus", "audience", "spectators",
    "fans", "passenger", "passengers", "pedestrian", "pedestrians", "guard",
    "guards", "reporter", "reporters", "journalist", "journalists",
}


def _words(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def classify_facets(subject: str, relation: str, obj: str) -> Dict[str, bool]:
    """Return the ``{facet: True}`` flags that apply to one triplet."""
    rel = relation.lower()
    rel_words = _words(rel)
    flags: Dict[str, bool] = {}
    if rel_words & _SPATIAL or any(p in rel for p in _SPATIAL_PHRASES):
        flags["location"] = True
    if rel_words & _TEXT or any(p in rel for p in _TEXT_PHRASES):
        flags["text"] = True
    if rel_words & _FRAMING or any(p in rel for p in _FRAMING_PHRASES):
        flags["framing"] = True
    if rel_words & _ACTION or (
        rel.endswith("ing") and "location" not in flags and "text" not in flags
    ):
        flags["action"] = True
    if _words(f"{subject} {obj}") & _PERSON:
        flags["people"] = True
    return flags


class VideoIndex:
    """Embeds scene-graph rows per video and retrieves videos by free-text query.

    Args:
        index_path: Directory for the Chroma store, registry, and media files.
        embedder:   Active embedding model (shared with the validation pipeline).
    """

    def __init__(self, index_path: str, embedder: MultimodalEmbedder):
        self._path = index_path
        self._embedder = embedder
        self._chroma_dir = os.path.join(index_path, "chroma")
        self._media_dir = os.path.join(index_path, "media")
        self._registry_path = os.path.join(index_path, _REGISTRY_FILENAME)
        os.makedirs(self._chroma_dir, exist_ok=True)
        os.makedirs(self._media_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._chroma_dir)
        self._registry = self._load_registry()

    # ------------------------------------------------------------------
    # Registry persistence
    # ------------------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.isfile(self._registry_path):
            with open(self._registry_path) as fh:
                return json.load(fh)
        return {"videos": {}}

    def _save_registry(self) -> None:
        with open(self._registry_path, "w") as fh:
            json.dump(self._registry, fh, indent=2)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _collection(self) -> "chromadb.Collection":
        slug = _model_slug(self._embedder.model_name)
        name = f"videoidx__{slug}"[:_MAX_COL_LEN]
        return self._client.get_or_create_collection(name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_video(
        self,
        rows: List[Dict[str, Any]],
        title: str,
        media_src_path: Optional[str] = None,
        ext: Optional[str] = None,
        source: str = "user",
    ) -> Dict[str, Any]:
        """Index one video's scene-graph rows and (optionally) store its file.

        Args:
            rows:           List of ``{subject, relation, object[, start_sec,
                            end_sec]}`` dicts (the flattened scene graph).
            title:          Display title (typically the original filename).
            media_src_path: Path to the uploaded video to copy into the store
                            for later playback. ``None`` indexes metadata only.
            ext:            File extension (with dot) for the stored media file.
            source:         Provenance tag stored on the video record.

        Returns the created video record. Rows with empty subject/relation/object
        are skipped; a video with zero usable rows is rejected.
        """
        clean: List[Dict[str, Any]] = []
        for r in rows:
            s = str(r.get("subject", "")).strip()
            rel = str(r.get("relation", "")).strip()
            o = str(r.get("object", "")).strip()
            if not s or not rel or not o:
                continue
            t0 = r.get("start_sec")
            t1 = r.get("end_sec")
            clean.append({
                "subject": s, "relation": rel, "object": o,
                "start_sec": float(t0) if t0 is not None else -1.0,
                "end_sec": float(t1) if t1 is not None else -1.0,
            })
        if not clean:
            raise ValueError("No usable scene-graph rows to index.")

        video_id = str(uuid.uuid4())

        # Store the media file for playback, if provided.
        stored_ext = ""
        if media_src_path and os.path.isfile(media_src_path):
            stored_ext = (ext or os.path.splitext(media_src_path)[1] or ".mp4").lower()
            dest = os.path.join(self._media_dir, f"{video_id}{stored_ext}")
            shutil.copyfile(media_src_path, dest)

        # Embed and write the rows.
        texts = [_row_text(r["subject"], r["relation"], r["object"]) for r in clean]
        embs = self._embedder.embed_texts(texts)
        ids = [f"{video_id}:{i}" for i in range(len(clean))]
        metas = []
        for r in clean:
            meta = {
                "video_id": video_id,
                "subject": r["subject"],
                "relation": r["relation"],
                "object": r["object"],
                "start_sec": r["start_sec"],
                "end_sec": r["end_sec"],
            }
            # Only True flags are stored, so a `where={"f_x": True}` filter
            # matches exactly the rows carrying that facet.
            for facet in classify_facets(r["subject"], r["relation"], r["object"]):
                meta[f"f_{facet}"] = True
            metas.append(meta)
        self._collection().add(
            documents=texts, embeddings=embs.tolist(), metadatas=metas, ids=ids
        )

        duration = max((r["end_sec"] for r in clean), default=0.0)
        record = {
            "id": video_id,
            "title": title,
            "ext": stored_ext,
            "has_media": bool(stored_ext),
            "num_rows": len(clean),
            "duration": round(duration, 2) if duration > 0 else None,
            "source": source,
            "model": self._embedder.model_name,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._registry["videos"][video_id] = record
        self._save_registry()
        log.info("Indexed video '%s' (%d rows, id=%s).", title, len(clean), video_id)
        return record

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_videos: int = 10,
        row_pool: int = 60,
        matches_per_video: int = 6,
        facet: Optional[str] = None,
        keyword: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve videos whose scene-graph rows best match ``query``.

        Embeds the query, pulls the ``row_pool`` closest rows across all videos,
        scores each by exact cosine similarity to the query, and KEEPS ONLY rows
        at or above ``min_score``. Rows are grouped by video; a video is scored by
        its single best matching row and is excluded entirely if none of its rows
        clear the threshold (this is what makes search a filter, not just a
        ranker — without it a small library returns every video for any query).

        Args:
            min_score: cosine-similarity floor in [0, 1]; defaults to
                       :data:`_DEFAULT_MIN_SCORE`. Pass 0 to disable filtering and
                       rank all videos (the old behaviour).

        Optional hybrid filters applied *inside* the same vector search:
            facet:   restrict to rows tagged with this facet (one of
                     :data:`FACET_KEYS`) via a metadata ``where`` clause.
            keyword: require this word/substring to appear literally in the row
                     text via a ``where_document`` ``$contains`` clause.
        """
        query = (query or "").strip()
        if not query:
            return []
        col = self._collection()
        count = col.count()
        if count == 0:
            return []

        thr = _DEFAULT_MIN_SCORE if min_score is None else float(min_score)
        where = {f"f_{facet}": True} if facet in FACET_KEYS else None
        kw = (keyword or "").strip()
        where_document = {"$contains": kw} if kw else None

        qvec = self._embedder.embed_query(query).flatten().astype(np.float32)
        qn = float(np.linalg.norm(qvec)) or 1.0
        qvec = qvec / qn

        res = col.query(
            query_embeddings=[qvec.tolist()],
            n_results=min(row_pool, count),
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        embs = res.get("embeddings", [[]])[0]

        grouped: Dict[str, Dict[str, Any]] = {}
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            vid = meta.get("video_id")
            if vid is None:
                continue
            # Exact cosine similarity (both vectors unit-normalised). This is
            # metric-agnostic, so the threshold is meaningful regardless of the
            # collection's configured distance space.
            if i < len(embs) and embs[i] is not None:
                rv = np.asarray(embs[i], dtype=np.float32)
                rn = float(np.linalg.norm(rv)) or 1.0
                cos = float(np.dot(qvec, rv / rn))
            else:  # fall back to distance→score if embeddings are unavailable
                cos = 1.0 / (1.0 + float(dist))
            if cos < thr:
                continue
            g = grouped.setdefault(vid, {"video_id": vid, "matches": [], "best": cos})
            g["best"] = max(g["best"], cos)
            start = meta.get("start_sec", -1.0)
            end = meta.get("end_sec", -1.0)
            g["matches"].append({
                "text": doc,
                "subject": meta.get("subject"),
                "relation": meta.get("relation"),
                "object": meta.get("object"),
                "start_sec": None if start is None or start < 0 else float(start),
                "end_sec": None if end is None or end < 0 else float(end),
                "facets": [k[2:] for k in meta if k.startswith("f_") and meta[k]],
                "distance": float(dist),
                "score": round(cos, 4),
            })

        out: List[Dict[str, Any]] = []
        for vid, g in grouped.items():
            rec = self._registry["videos"].get(vid)
            if rec is None:
                continue  # row references a deleted video
            g["matches"].sort(key=lambda m: m["score"], reverse=True)
            out.append({
                **rec,
                "score": round(float(g["best"]), 4),
                "num_matches": len(g["matches"]),
                "matches": g["matches"][:matches_per_video],
            })
        out.sort(key=lambda v: v["score"], reverse=True)
        return out[:max_videos]

    # ------------------------------------------------------------------
    # Listing / deletion / media
    # ------------------------------------------------------------------

    def list_videos(self) -> List[Dict[str, Any]]:
        """All indexed video records, most recently indexed first."""
        vids = list(self._registry["videos"].values())
        vids.sort(key=lambda v: v.get("indexed_at", ""), reverse=True)
        return vids

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        return self._registry["videos"].get(video_id)

    def media_path(self, video_id: str) -> Optional[str]:
        rec = self._registry["videos"].get(video_id)
        if not rec or not rec.get("ext"):
            return None
        path = os.path.join(self._media_dir, f"{video_id}{rec['ext']}")
        return path if os.path.isfile(path) else None

    def delete_video(self, video_id: str) -> bool:
        """Remove a video's rows, its media file, and its registry record."""
        if video_id not in self._registry["videos"]:
            return False
        try:
            self._collection().delete(where={"video_id": video_id})
        except Exception as exc:
            log.warning("Could not delete rows for video '%s': %s", video_id, exc)
        media = self.media_path(video_id)
        if media:
            try:
                os.remove(media)
            except OSError:
                pass
        self._registry["videos"].pop(video_id, None)
        self._save_registry()
        log.info("Deleted video '%s'.", video_id)
        return True

    def count_videos(self) -> int:
        return len(self._registry["videos"])
