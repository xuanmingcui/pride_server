"""Multimodal embedder for fact retrieval in the validation pipeline.

Wraps a Qwen3-VL-Embedding model to produce dense vectors in a *single shared
space* from text, images, and video frames.  Because the query and the stored
facts live in the same space, an image-only or video-only query retrieves
relevant text facts directly — no transcript or caption is required.

Backends
--------
transformers  (default, required for multimodal queries)
    Loads ``Qwen3VLModel`` + ``Qwen3VLProcessor`` locally.  Inputs are wrapped
    in the model's chat template (system = task instruction, user = the typed
    text / image / video parts), encoded, then last-token pooled and
    L2-normalised — the exact recipe the model was trained with.

vllm  (text-only)
    Runs the model through a local vLLM instance with task="embed" for fast
    text embedding.  Visual frames are NOT encoded on this backend; image /
    video queries fall back to a text placeholder.  Use the transformers
    backend when multimodal retrieval accuracy matters.

Instruction awareness
---------------------
Qwen3-VL-Embedding is instruction-aware: a short task instruction is supplied
as the system prompt.  Queries use a retrieval instruction; fact documents use
a neutral representation instruction.  Query and document instructions may
differ — the shared space is preserved regardless.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("pride.embedder")

# Instruction (system prompt) used when embedding a retrieval query.
_QUERY_INSTRUCTION = (
    "Retrieve facts from the knowledge base that help verify the claims "
    "in the user's input."
)
# Instruction used when embedding a fact document for indexing.
_DOC_INSTRUCTION = "Represent this fact for retrieval."

# Vision budgets (mirrors the model's reference implementation; one factor unit
# = 32px after the patch-16 / 2x merge).  Kept conservative so CPU/GPU memory
# stays bounded for multi-frame video queries.
_IMAGE_FACTOR        = 32
_MIN_PIXELS          = 4 * _IMAGE_FACTOR * _IMAGE_FACTOR
_MAX_PIXELS          = 1800 * _IMAGE_FACTOR * _IMAGE_FACTOR
_FRAME_MAX_PIXELS    = 768 * _IMAGE_FACTOR * _IMAGE_FACTOR
_MAX_TOTAL_PIXELS    = 10 * _FRAME_MAX_PIXELS
# Image-patch size the vision tower expects (used by process_vision_info).
_IMAGE_PATCH_SIZE    = 16
# Upper bound on frames encoded for a video query (evenly subsampled). Keeps the
# query forward pass bounded regardless of clip length / sampling fps.
_MAX_QUERY_FRAMES    = 16

# Placeholder query used only on the text-only vLLM backend when an image /
# video is submitted without any accompanying text.
_VISUAL_PLACEHOLDER  = "[visual content submitted for fact-checking]"

_embedder_instance: Optional["MultimodalEmbedder"] = None


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _subsample(items: List[Any], max_count: int) -> List[Any]:
    """Evenly pick at most ``max_count`` items, preserving order and endpoints."""
    if len(items) <= max_count:
        return items
    idx = np.linspace(0, len(items) - 1, max_count).round().astype(int)
    return [items[i] for i in idx]


def _last_token_pool(last_hidden_state, attention_mask):
    """Pool the embedding of the final non-pad token (robust to right padding)."""
    import torch
    flipped = attention_mask.flip(dims=[1])
    last_one = flipped.argmax(dim=1)
    col = attention_mask.shape[1] - last_one - 1
    row = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[row, col]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class MultimodalEmbedder:
    """Dense encoder over text (+ optional image/video) for queries and facts.

    Args:
        model_name:    HuggingFace model ID (a Qwen3-VL-Embedding checkpoint).
        backend:       "transformers" (multimodal) or "vllm" (text-only).
        device:        PyTorch device string ("cpu", "cuda", "cuda:1", …).
        max_length:    Maximum token length for the encoder.
        batch_size:    Conversations per forward pass (transformers backend).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        backend: str = "transformers",
        device: str = "cuda",
        max_length: int = 8192,
        batch_size: int = 32,
    ):
        self.model_name  = model_name
        self.backend     = backend
        self.device      = device
        self.max_length  = max_length
        self.batch_size  = batch_size

        self._processor  = None
        self._model      = None
        self._vllm_llm   = None

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import Qwen3VLModel, Qwen3VLProcessor

        log.info("Loading multimodal embedder %s via transformers on %s …",
                 self.model_name, self.device)
        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        device_map  = "cpu" if self.device == "cpu" else {"": self.device}
        self._model = Qwen3VLModel.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self._processor = Qwen3VLProcessor.from_pretrained(
            self.model_name, padding_side="right", trust_remote_code=True,
        )
        log.info("Multimodal embedder loaded (transformers).")

    def _load_vllm(self) -> None:
        from vllm import LLM

        log.info("Loading embedder %s via vLLM on %s (text-only) …",
                 self.model_name, self.device)
        self._vllm_llm = LLM(
            model=self.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        log.info("Embedder loaded (vLLM, text-only).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of fact strings for database indexing.

        Returns float32 array of shape (N, dim), L2-normalised.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        if self.backend == "vllm":
            return self._encode_vllm_text(texts, _DOC_INSTRUCTION)
        items = [{"text": t} for t in texts]
        return self._encode_multimodal(items, _DOC_INSTRUCTION)

    def embed_query(
        self,
        text: str,
        frames: Optional[List[Any]] = None,  # List[PIL.Image]
    ) -> np.ndarray:
        """Embed a (possibly multimodal) query for retrieval.

        On the transformers backend the frames are encoded directly: a single
        frame is treated as an image, multiple frames as a video clip.  On the
        vLLM (text-only) backend frames are ignored and a placeholder is used
        when no text is present.

        Returns float32 array of shape (1, dim), L2-normalised.
        """
        query = (text or "").strip()

        if self.backend == "vllm":
            if not query and frames:
                query = _VISUAL_PLACEHOLDER
            return self._encode_vllm_text([query], _QUERY_INSTRUCTION)

        image = video = None
        if frames:
            rgb = [f.convert("RGB") for f in frames]
            if len(rgb) == 1:
                image = rgb[0]
            else:
                video = _subsample(rgb, _MAX_QUERY_FRAMES)
        item = {"text": query or None, "image": image, "video": video}
        return self._encode_multimodal([item], _QUERY_INSTRUCTION)

    @property
    def dim(self) -> int:
        """Embedding dimension (inferred on first call)."""
        if not hasattr(self, "_dim"):
            self._dim = int(self.embed_texts(["ping"]).shape[1])
        return self._dim

    # ------------------------------------------------------------------
    # Transformers (multimodal) encoding
    # ------------------------------------------------------------------

    def _build_conversation(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        video: Optional[List[Any]] = None,
        instruction: str = _DOC_INSTRUCTION,
    ) -> List[Dict[str, Any]]:
        """Build a single chat-template conversation for one input item."""
        instr = (instruction or "").strip()
        if instr and not unicodedata.category(instr[-1]).startswith("P"):
            instr += "."

        content: List[Dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instr}]},
            {"role": "user",   "content": content},
        ]
        if not text and image is None and not video:
            content.append({"type": "text", "text": "NULL"})
            return conversation
        if video:
            content.append({
                "type": "video", "video": video, "total_pixels": _MAX_TOTAL_PIXELS,
            })
        if image is not None:
            content.append({
                "type": "image", "image": image,
                "min_pixels": _MIN_PIXELS, "max_pixels": _MAX_PIXELS,
            })
        if text:
            content.append({"type": "text", "text": text})
        return conversation

    def _preprocess(self, conversations: List[List[Dict[str, Any]]]):
        from qwen_vl_utils.vision_process import process_vision_info

        text = self._processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False,
        )
        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=_IMAGE_PATCH_SIZE,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as e:  # text-only batches can trip some versions
            log.debug("process_vision_info skipped (text-only?): %s", e)
            images, video_inputs, video_kwargs = None, None, {}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self._processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    def _encode_multimodal(
        self, items: List[Dict[str, Any]], instruction: str
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        all_embs: List[np.ndarray] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            convs = [
                self._build_conversation(
                    text=it.get("text"), image=it.get("image"),
                    video=it.get("video"), instruction=instruction,
                )
                for it in batch
            ]
            inputs = self._preprocess(convs)
            inputs = {
                k: (v.to(self._model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.float().cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    # ------------------------------------------------------------------
    # vLLM (text-only) encoding
    # ------------------------------------------------------------------

    def _encode_vllm_text(self, texts: List[str], instruction: str) -> np.ndarray:
        # Wrap each text in the instruction as a lightweight system prefix.
        prefixed = [f"{instruction}\n{t}" if instruction else t for t in texts]
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
            model_name=config.get("embed_model",   "Qwen/Qwen3-VL-Embedding-2B"),
            backend=   config.get("embed_backend", "transformers"),
            device=    config.get("embed_device",  "cuda"),
            max_length=config.get("embed_max_length", 8192),
            batch_size=config.get("embed_batch_size", 32),
        )
    return _embedder_instance
