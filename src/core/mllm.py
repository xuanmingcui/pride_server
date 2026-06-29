"""Batched MLLM backends (vLLM and HuggingFace Transformers).

Key design: both backends expose `generate_batch(requests)` which accepts a
*list* of request dicts and submits them in a single forward pass / single
vLLM generate call.  This is the primary efficiency gain over the old
one-at-a-time loop in generate_news_scenegraph.py.

Request dict schema:
    {
        "prompt":        str,                       # user-role text
        "system_prompt": Optional[str],             # system-role text; default = generic assistant
        "frames":        Optional[List[PIL.Image]], # None for text-only requests
        "fps":           Optional[float],
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from PIL import Image

from .triplets import Triplet, extract_triplets_from_text, validate_triplets

_backend_instance: Optional["BaseMLLM"] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseMLLM:
    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[List[Triplet]]:
        raise NotImplementedError

    def generate_batch_raw(self, requests: List[Dict[str, Any]]) -> List[str]:
        """Batched generation returning raw model text (no triplet parsing).

        Default falls back to sequential generate_raw calls.
        Subclasses may override for true batching.
        """
        return [
            self.generate_raw(r["prompt"], r.get("frames"), r.get("fps"),
                              system_prompt=r.get("system_prompt"))
            for r in requests
        ]

    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> str:
        """Text-only generation for refinement / normalization prompts."""
        raise NotImplementedError

    def generate_text_batch(
        self,
        user_prompts: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Batched text-only generation. Default falls back to sequential."""
        if system_prompts is None:
            system_prompts = [None] * len(user_prompts)
        return [
            self.generate_text(up, system_prompt=sp, max_tokens=max_tokens)
            for up, sp in zip(user_prompts, system_prompts)
        ]

    def generate_raw(
        self,
        user_prompt: str,
        frames: Optional[List[Image.Image]] = None,
        fps: Optional[float] = None,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Multimodal generation returning raw text (no triplet parsing).

        Used by the validation pipeline to produce free-form reports.
        Subclasses that support visual input should include the frames;
        text-only subclasses may ignore them.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# vLLM backend  (primary / recommended)
# ---------------------------------------------------------------------------

class VLLMBackend(BaseMLLM):
    """
    Submits all requests in a single llm.generate() call so vLLM can fill
    GPU batches optimally.  Supports mixed text-only and video/image requests
    in the same batch.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        enable_thinking: bool = False,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Qwen3 / Qwen3.5 "thinking" toggle, forwarded to apply_chat_template.
        self.enable_thinking = enable_thinking
        self.SamplingParams = SamplingParams

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1, "image": 16},
            max_model_len=max_model_len,
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    _DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def _build_request(
        self,
        user_prompt: str,
        frames: Optional[List[Image.Image]],
        fps: Optional[float],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        if frames:
            frames_rgb = [f.convert("RGB") for f in frames]
            user_content = [
                {"type": "text", "text": user_prompt},
                {"type": "video", "video": frames_rgb, "fps": fps or 1.0},
            ]
        else:
            user_content = [{"type": "text", "text": user_prompt}]

        sys_text = system_prompt or self._DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_content},
        ]
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        if frames:
            _, video_inputs, _ = process_vision_info(
                messages, return_video_kwargs=True, return_video_metadata=True
            )
            return {"prompt": prompt_str, "multi_modal_data": {"video": video_inputs}}
        return {"prompt": prompt_str}

    def _sg_sampling(self):
        """SamplingParams for scene-graph (multimodal) calls.

        repetition_penalty discourages the model from locking onto a single
        subject or relation and emitting the same pattern until the token
        budget is exhausted.
        """
        return self.SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            # Mild penalty only: windowed extraction keeps each call short, and a
            # dense comprehensive graph legitimately repeats subjects/relations
            # (e.g. the same group across many rows). An aggressive penalty here
            # suppresses that and caps recall.
            repetition_penalty=1.1,
        )

    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[List[Triplet]]:
        if not requests:
            return []
        sampling = self._sg_sampling()
        vllm_reqs = [
            self._build_request(r["prompt"], r.get("frames"), r.get("fps"),
                                system_prompt=r.get("system_prompt"))
            for r in requests
        ]
        outputs = self.llm.generate(vllm_reqs, sampling_params=sampling)
        results: List[List[Triplet]] = []
        for out in outputs:
            text = out.outputs[0].text
            trips = extract_triplets_from_text(text)
            results.append(validate_triplets(trips))
        return results

    def generate_batch_raw(self, requests: List[Dict[str, Any]]) -> List[str]:
        if not requests:
            return []
        sampling = self._sg_sampling()
        vllm_reqs = [
            self._build_request(r["prompt"], r.get("frames"), r.get("fps"),
                                system_prompt=r.get("system_prompt"))
            for r in requests
        ]
        outputs = self.llm.generate(vllm_reqs, sampling_params=sampling)
        return [out.outputs[0].text for out in outputs]

    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> str:
        # Allow enough room to re-emit a full quintuple list during normalization.
        sampling = self.SamplingParams(
            max_tokens=max_tokens or self.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.1,
        )
        req = self._build_request(user_prompt, None, None, system_prompt=system_prompt)
        outputs = self.llm.generate([req], sampling_params=sampling)
        return outputs[0].outputs[0].text

    def generate_text_batch(
        self,
        user_prompts: List[str],
        system_prompts: Optional[List[Optional[str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        if not user_prompts:
            return []
        sampling = self.SamplingParams(
            max_tokens=max_tokens or self.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.1,
        )
        if system_prompts is None:
            system_prompts = [None] * len(user_prompts)
        reqs = [
            self._build_request(up, None, None, system_prompt=sp)
            for up, sp in zip(user_prompts, system_prompts)
        ]
        outputs = self.llm.generate(reqs, sampling_params=sampling)
        return [o.outputs[0].text for o in outputs]

    def generate_raw(
        self,
        user_prompt: str,
        frames: Optional[List[Image.Image]] = None,
        fps: Optional[float] = None,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        sampling = self.SamplingParams(max_tokens=max_tokens, temperature=0.0)
        req = self._build_request(user_prompt, frames, fps, system_prompt=system_prompt)
        outputs = self.llm.generate([req], sampling_params=sampling)
        return outputs[0].outputs[0].text


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend  (fallback / CPU)
# ---------------------------------------------------------------------------

class TransformersVLBackend(BaseMLLM):
    """
    Sequential fallback using HuggingFace Transformers.
    generate_batch() still accepts a list but processes requests one-by-one
    (GPU memory is usually the bottleneck, not Python overhead, so this is fine
    as a fallback).

    NOTE: For Qwen3-VL, the processor expects the chat-template format used
    here.  Other VL models may need minor adjustments to the message format.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        enable_thinking: bool = False,
    ):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking

        # Resolve device_map: pin to an explicit device rather than "auto" so
        # that the MLLM does not silently spread across GPUs intended for other
        # components (Whisper, embedding model).
        if device == "cpu":
            device_map: object = "cpu"
        elif device == "auto":
            device_map = "auto"
        else:
            # "cuda" → GPU 0; "cuda:N" → GPU N (dict form tells HF to put every
            # layer on that single device).
            target = device if ":" in device else "cuda:0"
            device_map = {"": target}
        self._device = torch.device(device if device != "auto" else
                                    ("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype, torch.float16)
        # Use the generic image-text-to-text auto class so newer architectures
        # (e.g. Qwen3.5's Qwen3_5ForConditionalGeneration) load without a
        # hardcoded class; fall back to the Qwen2.5-VL class on older transformers.
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except Exception:
            model_cls = Qwen2_5_VLForConditionalGeneration
        self.model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    _DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def _run_one(
        self,
        user_prompt: str,
        frames: Optional[List[Image.Image]],
        fps: Optional[float],
        system_prompt: Optional[str] = None,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        if frames:
            frames_rgb = [f.convert("RGB") for f in frames]
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "video", "video": frames_rgb, "fps": fps or 1.0},
            ]
        else:
            content = [{"type": "text", "text": user_prompt}]

        messages = [
            {"role": "system", "content": system_prompt or self._DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return self.processor.decode(gen[0][input_len:], skip_special_tokens=True)

    def generate_batch(self, requests: List[Dict[str, Any]]) -> List[List[Triplet]]:
        results: List[List[Triplet]] = []
        for r in requests:
            text = self._run_one(r["prompt"], r.get("frames"), r.get("fps"),
                                 system_prompt=r.get("system_prompt"))
            trips = extract_triplets_from_text(text)
            results.append(validate_triplets(trips))
        return results

    def generate_text(self, user_prompt: str, system_prompt: Optional[str] = None,
                      max_tokens: Optional[int] = None) -> str:
        if max_tokens is None:
            return self._run_one(user_prompt, None, None, system_prompt=system_prompt)
        old = self.max_new_tokens
        self.max_new_tokens = max_tokens
        try:
            return self._run_one(user_prompt, None, None, system_prompt=system_prompt)
        finally:
            self.max_new_tokens = old

    def generate_raw(
        self,
        user_prompt: str,
        frames: Optional[List[Image.Image]] = None,
        fps: Optional[float] = None,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        old = self.max_new_tokens
        self.max_new_tokens = max_tokens
        try:
            return self._run_one(user_prompt, frames, fps, system_prompt=system_prompt)
        finally:
            self.max_new_tokens = old


# ---------------------------------------------------------------------------
# Factory / singleton
# ---------------------------------------------------------------------------

def get_backend(config: Dict[str, Any]) -> BaseMLLM:
    """Return (and cache) the MLLM backend specified in config."""
    global _backend_instance
    if _backend_instance is None:
        backend_type = config.get("backend", "vllm")
        model_name   = config["name"]
        common = {
            "model_name":     model_name,
            "max_new_tokens": config.get("max_new_tokens", 2048),
            "temperature":    config.get("temperature", 0.8),
            "enable_thinking": config.get("enable_thinking", False),
        }
        if backend_type == "vllm":
            _backend_instance = VLLMBackend(
                **common,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                max_model_len=config.get("max_model_len", 32768),
            )
        else:
            # Normalise "cuda" → "cuda:0" so TransformersVLBackend is pinned to
            # GPU 0 rather than letting device_map="auto" spread across all GPUs.
            raw_device = config.get("device", "cuda")
            device = raw_device if (raw_device in ("cpu", "auto") or ":" in raw_device) else "cuda:0"
            _backend_instance = TransformersVLBackend(
                **common,
                device=device,
                dtype=config.get("dtype", "auto"),
            )
    return _backend_instance


def reset_backend() -> None:
    """Force the next get_backend() call to reload the model (e.g., after config change)."""
    global _backend_instance
    _backend_instance = None
