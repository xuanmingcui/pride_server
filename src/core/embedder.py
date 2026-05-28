"""Multimodal embedder for fact retrieval in the validation pipeline.

Wraps a Qwen3-Embedding (or compatible) model to produce dense vectors from
text and optionally from visual frames.  Two backends are supported:

  transformers  (default)
      Loads the model locally via HuggingFace Transformers.  Uses last-token
      pooling — the strategy required by decoder-style embedding models such
      as Qwen3-Embedding.

  vllm
      Runs the model through a local vLLM LLM instance with task="embed".
      Supports the same interface as the transformers backend and enables
      tensor-parallel acceleration.  Cannot share a process with the
      generation-mode MLLM, so it starts its own LLM instance.

Multimodal queries (video / image + text)
-----------------------------------------
Qwen3-Embedding is a text model; visual frames are not directly encoded.
For retrieval, the query vector is built from:
    • the audio transcript  (richest signal when audio is present)
    • the user-supplied text
    • a short visual placeholder tag when frames are present but no text exists

The MLLM validation step that follows sees the actual frames for visual
reasoning.  Retrieval accuracy comes primarily from the transcript.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("pride.embedder")

# Query instruction for Qwen3-Embedding (instruction-aware model).
_QUERY_INSTRUCTION = (
    "Instruct: Retrieve relevant facts for verifying the following content\nQuery: "
)
# No instruction needed when embedding the documents (facts) themselves.
_DOC_INSTRUCTION = ""

_embedder_instance: Optional["MultimodalEmbedder"] = None


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _last_token_pool(last_hidden_state, attention_mask):
    """Last-token pooling for decoder-style models (e.g. Qwen3-Embedding)."""
    import torch
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size  = last_hidden_state.size(0)
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lengths]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class MultimodalEmbedder:
    """Dense encoder for text (+ optional visual context) queries and fact documents.

    Args:
        model_name:    HuggingFace model ID (default: Qwen/Qwen3-Embedding).
        backend:       "transformers" or "vllm".
        device:        PyTorch device string ("cpu", "cuda:1", …).
        max_length:    Maximum token length for the encoder.
        batch_size:    Texts per forward pass (transformers backend only).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding",
        backend: str = "vllm",
        device: str = "cuda",
        max_length: int = 4096,
        batch_size: int = 32,
    ):
        self.model_name  = model_name
        self.backend     = backend
        self.device      = device
        self.max_length  = max_length
        self.batch_size  = batch_size

        self._tokenizer  = None
        self._model      = None
        self._vllm_llm   = None

        self._load()

    def _load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModel

        log.info("Loading embedding model %s via transformers on %s …",
                 self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            device_map={"": self.device},
            trust_remote_code=True,
        ).eval()
        log.info("Embedding model loaded (transformers).")

    def _load_vllm(self) -> None:
        from vllm import LLM

        log.info("Loading embedding model %s via vLLM on %s …",
                 self.model_name, self.device)
        self._vllm_llm = LLM(
            model=self.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        log.info("Embedding model loaded (vLLM).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of fact strings for database indexing.

        Returns float32 array of shape (N, dim), L2-normalised.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._encode(texts, instruction=_DOC_INSTRUCTION)

    def embed_query(
        self,
        text: str,
        frames: Optional[List[Any]] = None,  # List[PIL.Image]
    ) -> np.ndarray:
        """Embed a (possibly multimodal) query for retrieval.

        For text-only queries the text is encoded directly.  When frames are
        present but no text is provided a visual-placeholder prefix is added so
        the model knows visual context exists.

        Returns float32 array of shape (1, dim), L2-normalised.
        """
        query = text.strip()
        if not query and frames:
            query = "[visual content submitted for fact-checking]"
        elif frames and query:
            # Prepend a short signal so the model weights the visual context
            query = f"[with visual content] {query}"
        return self._encode([query], instruction=_QUERY_INSTRUCTION)

    @property
    def dim(self) -> int:
        """Embedding dimension (inferred on first call)."""
        if not hasattr(self, "_dim"):
            sample = self._encode(["ping"], instruction="")
            self._dim = sample.shape[1]
        return self._dim

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str], instruction: str) -> np.ndarray:
        if self.backend == "vllm":
            return self._encode_vllm(texts, instruction)
        return self._encode_transformers(texts, instruction)

    def _encode_transformers(self, texts: List[str], instruction: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        prefixed = [instruction + t for t in texts]
        all_embs: List[np.ndarray] = []

        for i in range(0, len(prefixed), self.batch_size):
            batch = prefixed[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            emb = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.float().cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def _encode_vllm(self, texts: List[str], instruction: str) -> np.ndarray:
        prefixed = [instruction + t for t in texts]
        outputs  = self._vllm_llm.encode(prefixed)
        embs     = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        norms    = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

def get_embedder(config: Dict[str, Any]) -> MultimodalEmbedder:
    """Return (and cache) the embedder specified in the validation config."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultimodalEmbedder(
            model_name=config.get("embed_model",   "Qwen/Qwen3-Embedding"),
            backend=   config.get("embed_backend", "transformers"),
            device=    config.get("embed_device",  "cpu"),
            max_length=config.get("embed_max_length", 4096),
            batch_size=config.get("embed_batch_size", 32),
        )
    return _embedder_instance
