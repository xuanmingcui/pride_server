"""ChromaDB-backed fact store with per-embedding-model variant caching.

Each logical database (e.g. "default") can hold up to ``max_variants``
embedding-model-specific sub-collections in ChromaDB.  Variants are keyed
by the embedder's model name, enabling seamless model switching without
blocking recomputation.

Variant lifecycle
-----------------
First use of model M on database D
    A new ChromaDB collection is created for (D, M).  If other variants of D
    already exist, all their documents are re-embedded with M and inserted
    into the new collection (backfill).  This is the only moment where
    cross-model recomputation occurs.

Subsequent add_facts calls
    Facts are written into the *current* model's variant only.  Other cached
    variants become stale with respect to newly added facts — that is the
    accepted cache trade-off.

LRU eviction
    When a (max_variants + 1)-th model is requested for the same database,
    the least-recently-used variant's ChromaDB collection is deleted.
    Documents that were *only* in the evicted variant are no longer
    searchable via that model, but they remain in all other active variants.

delete_facts
    Deletes the given IDs from *all* active variants so they don't reappear
    if a previously evicted model is reactivated.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from ..core.embedder import MultimodalEmbedder

log = logging.getLogger("pride.database")

_META_FILENAME  = "variants_meta.json"
_MAX_SLUG_LEN   = 40   # chars reserved for the model slug in a collection name
_MAX_COL_LEN    = 63   # ChromaDB collection name limit


def _model_slug(model_name: str) -> str:
    """Turn a HuggingFace model ID into a safe ChromaDB name fragment."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:_MAX_SLUG_LEN]


def _collection_name(db_name: str, slug: str) -> str:
    available = _MAX_COL_LEN - len(slug) - 2   # 2 for the "__" separator
    return f"{db_name[:available]}__{slug}"


class FactDatabase:
    """Manages named ChromaDB collections of facts with per-model variant caching.

    Args:
        db_path:      Directory for ChromaDB persistent storage.
        embedder:     Active embedding model instance.
        max_variants: Maximum number of model-specific indexes to keep per
                      logical database before LRU eviction (default 2).
    """

    def __init__(
        self,
        db_path: str,
        embedder: MultimodalEmbedder,
        max_variants: int = 2,
    ):
        self._client       = chromadb.PersistentClient(path=db_path)
        self._embedder     = embedder
        self._max_variants = max_variants
        self._meta_path    = os.path.join(db_path, _META_FILENAME)
        self._meta         = self._load_meta()

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def _load_meta(self) -> Dict[str, Any]:
        if os.path.isfile(self._meta_path):
            with open(self._meta_path) as fh:
                return json.load(fh)
        return {"databases": {}}

    def _save_meta(self) -> None:
        with open(self._meta_path, "w") as fh:
            json.dump(self._meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Variant management
    # ------------------------------------------------------------------

    def _variants(self, db_name: str) -> List[Dict[str, Any]]:
        """Return the list of active variant records for a logical database."""
        return self._meta["databases"].setdefault(db_name, [])

    def _current_variant(self, db_name: str) -> Optional[Dict[str, Any]]:
        model = self._embedder.model_name
        return next((v for v in self._variants(db_name) if v["model"] == model), None)

    def _touch(self, variant: Dict[str, Any]) -> None:
        variant["last_used"] = datetime.now(timezone.utc).isoformat()
        self._save_meta()

    def _get_or_create_variant(self, db_name: str) -> "chromadb.Collection":
        """Return the ChromaDB collection for (db_name, current model).

        Creates and backfills a new variant if this model hasn't been used
        for this database before.  Evicts the LRU variant when the cache is full.
        """
        model   = self._embedder.model_name
        variant = self._current_variant(db_name)

        if variant is not None:
            self._touch(variant)
            return self._client.get_or_create_collection(variant["collection"])

        # ── New variant needed ────────────────────────────────────────────────
        slug     = _model_slug(model)
        col_name = _collection_name(db_name, slug)
        variants = self._variants(db_name)

        # Evict LRU when at capacity
        if len(variants) >= self._max_variants:
            lru = min(variants, key=lambda v: v["last_used"])
            log.info(
                "Variant cache full for '%s' (max=%d). "
                "Evicting LRU variant (model='%s', last used %s).",
                db_name, self._max_variants, lru["model"], lru["last_used"],
            )
            try:
                self._client.delete_collection(lru["collection"])
            except Exception as exc:
                log.warning("Could not delete evicted collection '%s': %s",
                            lru["collection"], exc)
            variants.remove(lru)

        # Create the new collection
        new_col = self._client.get_or_create_collection(col_name)

        # Backfill from any existing variant that has facts
        source = next(
            (v for v in variants if self._client.get_or_create_collection(
                v["collection"]).count() > 0),
            None,
        )
        if source:
            src_col = self._client.get_or_create_collection(source["collection"])
            result  = src_col.get(include=["documents", "metadatas", "ids"])
            if result["ids"]:
                log.info(
                    "Backfilling %d fact(s) from model '%s' into new variant "
                    "for model '%s' (database '%s') …",
                    len(result["ids"]), source["model"], model, db_name,
                )
                new_embs = self._embedder.embed_texts(result["documents"])
                new_col.add(
                    documents  = result["documents"],
                    embeddings = new_embs.tolist(),
                    metadatas  = result["metadatas"],
                    ids        = result["ids"],
                )
                log.info("Backfill complete.")

        # Register and persist
        new_entry = {
            "model":      model,
            "collection": col_name,
            "last_used":  datetime.now(timezone.utc).isoformat(),
        }
        variants.append(new_entry)
        self._save_meta()
        log.info("Created variant for model '%s' on database '%s'.", model, db_name)
        return new_col

    # ------------------------------------------------------------------
    # Public: collection management
    # ------------------------------------------------------------------

    def list_databases(self) -> List[str]:
        """Return logical database names (independent of how many variants each has)."""
        return list(self._meta["databases"].keys())

    def create_database(self, name: str) -> None:
        """Register a logical database name (the variant is created lazily on first use)."""
        self._meta["databases"].setdefault(name, [])
        self._save_meta()

    def delete_database(self, name: str) -> None:
        """Delete all variants of a logical database."""
        for v in self._variants(name):
            try:
                self._client.delete_collection(v["collection"])
            except Exception:
                pass
        self._meta["databases"].pop(name, None)
        self._save_meta()

    # ------------------------------------------------------------------
    # Public: CRUD on facts
    # ------------------------------------------------------------------

    def add_facts(
        self,
        database: str,
        facts: List[str],
        source: str = "user",
        tags: str = "",
    ) -> List[str]:
        """Embed and insert facts into the current model's variant of `database`.

        Returns the generated IDs.  The same IDs are used across all variants
        so that delete_facts can remove a fact from every variant by ID.
        """
        if not facts:
            return []
        col  = self._get_or_create_variant(database)
        embs = self._embedder.embed_texts(facts)
        now  = datetime.now(timezone.utc).isoformat()
        ids  = [str(uuid.uuid4()) for _ in facts]
        meta = [{"source": source, "added_at": now, "tags": tags} for _ in facts]
        col.add(documents=facts, embeddings=embs.tolist(), metadatas=meta, ids=ids)
        return ids

    def search_facts(
        self, database: str, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Text-query search using the current embedder."""
        return self.search_by_embedding(database, self._embedder.embed_query(query), top_k)

    def search_by_embedding(
        self,
        database: str,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the current model's variant by a precomputed embedding vector."""
        col   = self._get_or_create_variant(database)
        count = col.count()
        if count == 0:
            return []
        k       = min(top_k, count)
        results = col.query(
            query_embeddings = [embedding.flatten().tolist()],
            n_results        = k,
            include          = ["documents", "metadatas", "distances"],
        )
        return [
            {"fact": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def list_facts(
        self,
        database: str,
        limit: int = 20,
        offset: int = 0,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if query:
            return self.search_facts(database, query, top_k=limit)
        col   = self._get_or_create_variant(database)
        total = col.count()
        if total == 0:
            return []
        results = col.get(
            limit=limit, offset=offset, include=["documents", "metadatas"]
        )
        return [
            {"id": fid, "fact": doc, "metadata": meta}
            for doc, meta, fid in zip(
                results["documents"], results["metadatas"], results["ids"]
            )
        ]

    def delete_facts(self, database: str, ids: List[str]) -> None:
        """Delete facts by ID from ALL active variants so they don't reappear."""
        for v in self._variants(database):
            try:
                col = self._client.get_or_create_collection(v["collection"])
                col.delete(ids=ids)
            except Exception as exc:
                log.warning("Could not delete from variant '%s': %s", v["collection"], exc)

    def count(self, database: str) -> int:
        """Count facts in the current model's variant."""
        try:
            return self._get_or_create_variant(database).count()
        except Exception:
            return 0
