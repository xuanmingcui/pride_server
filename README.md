# PRIDE Server

**Propaganda & Reasoning Intelligence Detection Engine**

A multimodal AI server with two primary capabilities:

1. **Scene Graph Generation** — Given any combination of video, image, audio, or text, extract a structured graph of (subject, relation, object) triplets describing what is happening.
2. **Fact Validation** — Given a user-curated knowledge database and any multimodal input, determine whether the input is consistent with the stored facts and generate a structured report.

Both capabilities are exposed via a **web UI** (FastAPI + SPA) and a **Discord bot**. Both interfaces share one set of loaded GPU models so no model is loaded twice.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [Entry Points](#entry-points)
- [Configuration](#configuration)
- [Core Modules](#core-modules)
  - [services.py](#servicespyservicespy)
  - [core/mllm.py](#coremllmpy)
  - [core/scenegraph.py](#corescenegraphpy)
  - [core/triplets.py](#coretripletspy)
  - [core/embedder.py](#coreembedderpy)
  - [core/audio.py](#coreaudiopy)
  - [core/overlay.py](#coreoverlaypy)
  - [core/gpu_utils.py](#coregpu_utilspy)
  - [validation/pipeline.py](#validationpipelinepy)
  - [validation/database.py](#validationdatabasepy)
- [Web API](#web-api)
  - [app.py](#apyapppy)
  - [task_queue.py](#apitask_queuepy)
  - [Routes](#routes)
  - [Prompt Templates](#prompts--apiprompts)
- [Web Frontend](#web-frontend)
- [Discord Bot](#discord-bot)
- [Running the Server](#running-the-server)
- [Docker Deployment](#docker-deployment)
- [Dependencies](#dependencies)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                              │
│         Web Browser               Discord                   │
└────────────┬──────────────────────────┬────────────────────┘
             │ HTTP                      │ WebSocket
             ▼                          ▼
┌────────────────────┐      ┌──────────────────────┐
│   FastAPI (uvicorn)│      │     PrideBot          │
│   src/api/app.py   │      │  src/bot/main.py      │
└─────────┬──────────┘      └──────────┬────────────┘
          │                            │
          └────────────┬───────────────┘
                       ▼
            ┌──────────────────┐
            │  ModelServices   │  ← single instance shared by both
            │ src/services.py  │
            └────┬──────┬──────┘
                 │      │
       ┌─────────┘      └─────────┐
       ▼                          ▼
┌──────────────┐        ┌──────────────────┐
│SceneGraph    │        │ ValidationPipeline│
│Pipeline      │        │ + FactDatabase    │
│(vLLM/MLLM)  │        │ (ChromaDB + embed)│
└──────────────┘        └──────────────────┘
```

**Key design decisions:**

- **Single model load** — `ModelServices` initialises all GPU models once. Both the web API and Discord bot receive the same instance.
- **Thread-pool executor** — ML inference runs in a `ThreadPoolExecutor(max_workers=1)` so async coroutines never block the event loop.
- **Task queue** — Long operations (scene graph, validation, embedding) return a `task_id` immediately. The client polls `GET /api/tasks/{id}` until the task is `done` or `error`.
- **Batched inference** — All segments of a video are batched into a single `llm.generate()` call, including the normalisation pass.

---

## Directory Structure

```
pride_server/
├── config.yaml              # Main configuration (see Configuration)
├── requirements.txt         # Python dependencies
├── run.py                   # Main entry: Discord bot + web + misinfo, one shared vLLM
│
├── frontend/
│   ├── index.html           # Single-page app (3 tabs)
│   ├── styles.css           # Dark theme, component styles
│   └── app.js               # Tab logic, polling, result rendering
│
├── src/
│   ├── services.py          # ModelServices dataclass + factory
│   │
│   ├── api/
│   │   ├── app.py           # FastAPI factory (create_app)
│   │   ├── task_queue.py    # Async single-worker task queue
│   │   └── routes/
│   │       ├── scenegraph.py   # POST /api/scenegraph
│   │       ├── validation.py   # POST /api/validate
│   │       ├── database.py     # CRUD  /api/databases/…
│   │       └── tasks.py        # GET  /api/tasks/{id}[/file]
│   │
│   ├── core/
│   │   ├── mllm.py          # VLLMBackend / TransformersVLBackend
│   │   ├── scenegraph.py    # SceneGraphPipeline
│   │   ├── triplets.py      # Parsing, validation, prompt builders
│   │   ├── embedder.py      # MultimodalEmbedder (text + visual)
│   │   ├── audio.py         # Whisper transcription helpers
│   │   ├── overlay.py       # Video/image overlay rendering (OpenCV + ffmpeg)
│   │   └── gpu_utils.py     # GPU assignment policy
│   │
│   ├── validation/
│   │   ├── pipeline.py      # ValidationPipeline (RAG fact-check)
│   │   └── database.py      # FactDatabase (ChromaDB wrapper)
│   │
│   └── bot/
│       ├── main.py          # PrideBot (discord.py)
│       ├── config.py        # Config loader with env-var overrides
│       └── commands/
│           ├── scenegraph_cmd.py  # /scenegraph slash command
│           ├── validate_cmd.py    # /validate + DB management commands
│           └── help_cmd.py        # /help
│
├── data/db/                 # ChromaDB persistent storage
├── output/                  # Generated overlay videos/images
└── tmp/                     # Temporary uploads and WAV extracts
```

---

## Entry Points

### `run.py` — Main entry (Discord bot + web + misinfo, one shared vLLM)

```
DISCORD_TOKEN=your_token python run.py [--host 0.0.0.0] [--port 8080]
```

Loads `ModelServices` once (a single vLLM engine), then runs the FastAPI server (`uvicorn.Server.serve()` with `loop="none"`), the Discord bot (`PrideBot.start()`), and the in-process misinfo verifier — all sharing the one model — concurrently via `asyncio.gather()`. Requires `DISCORD_TOKEN` in the environment.

**ML cache location** is configurable via env (or `.env`): set `PRIDE_CACHE_DIR` (default `/workspace/.cache`). run.py applies it to `HF_HOME`, `TORCH_HOME`, `VLLM_CACHE_ROOT`, `TMPDIR`, etc. before any ML import; an explicitly-exported `HF_HOME`/`TORCH_HOME`/… still takes precedence. (This is separate from `paths.tmp_dir` in `config.yaml`, the app's downloaded-media directory.)

### Hot-reloading (development)

```
python -m jurigged -w src/ run.py
```

[jurigged](https://github.com/breuleux/jurigged) patches changed function bodies in-place without restarting the process, so GPU models stay loaded. Suitable for iterating on prompts, pipeline logic, and API routes. Changes to imports or class structure still require a restart.

---

## Configuration

All settings live in `config.yaml`. Any key can be overridden with an environment variable prefixed `PRIDE_` using double underscores for nesting (e.g. `PRIDE_MODEL__TEMPERATURE=0.5`).

```yaml
model:
  name: "Qwen/Qwen3-VL-4B-Instruct-FP8"
  backend: "vllm"          # "vllm" (recommended) | "transformers" (CPU fallback)
  device: "cuda"
  dtype: "auto"
  max_new_tokens: 2048
  temperature: 0.8
  tensor_parallel_size: 1
  max_model_len: 32768

whisper:
  model_size: "large-v3"
  device: "cuda"
  compute_type: "float16"
  language: null             # null = auto-detect
  beam_size: 5
  vad_filter: true

scenegraph:
  num_frames: 16             # Frames sampled per video segment
  normalize_pass: true       # Second MLLM pass for entity normalisation
  temporal_target_fps: 1     # Target FPS for long-video segmentation
  tokens_per_frame: 256      # Used to compute context-budget limit
  prompt_overhead_tokens: 4096

validation:
  top_k: 5                   # Facts retrieved per query
  default_db: "default"
  num_frames: 16
  embed_model: "Qwen/Qwen3-VL-Embedding-2B"
  embed_backend: "transformers"
  embed_max_length: 8192
  embed_batch_size: 32

paths:
  db_dir: "./data/db"
  tmp_dir: "./tmp"
  output_dir: "./output"

discord:
  max_upload_mb: 10
  guild_id: null             # Set to your server ID for instant command sync
```

### GPU assignment policy (`core/gpu_utils.py`)

`assign_gpus(model_config)` returns `{mllm, whisper, embed}` device strings:

- MLLM occupies GPUs `0 … tensor_parallel_size-1`.
- Whisper gets the next free GPU, or shares GPU 0 if none are free.
- Embedding gets the next free GPU after Whisper, or falls back to CPU.

---

## Core Modules

### `src/services.py`

Container for all loaded models. The entry point (`run.py`) calls `await ModelServices.create(cfg)` once and passes the result around.

```python
@dataclass
class ModelServices:
    cfg:          Dict[str, Any]
    executor:     ThreadPoolExecutor   # max_workers=1, "pride-ml"
    backend:      BaseMLLM
    sg_pipeline:  SceneGraphPipeline
    embedder:     MultimodalEmbedder
    db:           FactDatabase
    val_pipeline: ValidationPipeline

    @classmethod
    async def create(cls, cfg) -> "ModelServices": ...

    async def run_in_thread(self, fn, *args, **kwargs): ...
```

---

### `core/mllm.py`

MLLM backends. Both expose the same interface; only `VLLMBackend` does true batching.

#### `BaseMLLM` (abstract)

| Method | Description |
|---|---|
| `generate_batch(requests)` | Accepts a list of `{prompt, frames, fps}` dicts; returns `List[List[Triplet]]` |
| `generate_text(prompt)` | Text-only generation; temperature=0, max_tokens=512 |
| `generate_text_batch(prompts)` | Batched text generation (default: sequential fallback) |
| `generate_raw(prompt, frames, fps, max_tokens)` | Free-form text output; used by ValidationPipeline |

#### `VLLMBackend`

```python
VLLMBackend(
    model_name: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
)
```

All requests in a `generate_batch()` call are submitted to vLLM in a single `llm.generate()` invocation, letting the scheduler fill GPU batches optimally. `generate_text_batch()` is similarly batched with `temperature=0`.

#### `TransformersVLBackend`

```python
TransformersVLBackend(
    model_name: str,
    device: str = "cuda:0",
    dtype: str = "auto",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
)
```

Sequential fallback for environments without vLLM. Pins the model to a single explicit GPU device.

#### Factory

```python
get_backend(config: Dict) -> BaseMLLM   # singleton
reset_backend() -> None                 # force reload on next call
```

---

### `core/scenegraph.py`

Orchestrates frame sampling, audio transcription, batched MLLM inference, normalisation, and optional overlay rendering.

#### `SceneGraphPipeline`

```python
SceneGraphPipeline(
    backend: BaseMLLM,
    whisper_config: Dict,
    scenegraph_config: Dict,
    tmp_dir: str = "/tmp",
)
```

**Public entry point:**

```python
pipeline.process(
    media_path: Optional[str] = None,  # video, image, or None
    text: str = "",                     # additional context or text-only input
    output_type: str = "json",          # "json" | "overlay"
    output_path: Optional[str] = None,
    temperature: Optional[float] = None,
    num_frames: Optional[int] = None,
    mode: str = "high",                 # "high" (semantic) | "low" (visual)
) -> Dict   # {triplets, segments, transcript, overlay_path, overlay_error}
```

**Dispatch logic:**

```
media_path is None      → _text_only()
media_path is image     → _image()
media_path is video
  has ASR segments      → _video_by_segments()   # one request per ASR segment
  no ASR, short video   → _video_whole()          # single request
  no ASR, long video    → _video_temporal()       # equal-length temporal splits
```

**Batching:** All segment requests are collected into a list and sent in **one** `backend.generate_batch()` call. The normalisation pass (`normalize_pass: true`) is also batched via `generate_text_batch()`.

**Helper functions:**

| Function | Description |
|---|---|
| `sample_frames(video_path, num_frames, start_sec, end_sec)` | Uniform frame sampling; returns `(List[PIL.Image], sample_fps)` |
| `_video_duration(video_path)` | Returns duration in seconds via OpenCV |
| `_max_segment_duration(num_frames)` | Adaptive limit based on temporal coverage and model context budget |

---

### `core/triplets.py`

All triplet-related utilities: parsing, validation, formatting, and prompt construction.

#### Types

```python
Triplet   = Tuple[str, str, str]                       # (subject, relation, object)
Quintuple = Tuple[str, str, str, float, float]         # + (start_sec, end_sec)
```

#### Parsing & validation

| Function | Description |
|---|---|
| `extract_triplets_from_text(text)` | `ast.literal_eval` with regex fallback for minor format violations |
| `validate_triplets(trips)` | Deduplicate, strip whitespace, drop malformed or oversized nodes |
| `validate_quintuples(quints)` | Same as above for quintuples |

#### Formatting

| Function | Description |
|---|---|
| `format_as_python_list(trips)` | Python list-of-tuples string |
| `format_as_json(trips)` | List of `{subject, relation, object}` dicts |
| `triplets_to_context_text(trips, max_items)` | Bullet-list string for prompt context |
| `triplets_to_claims(trips)` | List of `"subject relation object"` strings for RAG |
| `ms_to_srt_time(ms)` | `HH:MM:SS,mmm` for SRT files |
| `build_srt_from_segments(segments)` | Full SRT file content from `[{start, end, triplets}]` |

#### Prompt builders

| Function | Key parameter | Purpose |
|---|---|---|
| `build_scenegraph_prompt(transcript_text, mode)` | `mode="high"\|"low"` | Image / video-segment prompt |
| `build_text_only_scenegraph_prompt(text, mode)` | `mode` | Text-only scene graph prompt |
| `build_temporal_scenegraph_prompt(start, end, transcript_text, mode)` | `mode` | Temporal window prompt |
| `build_normalize_prompt(trips)` | — | Entity-normalisation second pass |

**Mode semantics:**
- `"high"` — Named entities, events, decisions, political/social relationships. Avoids generic visual descriptors.
- `"low"` — Physical objects, spatial positions, visual actions. Avoids abstract concepts.

---

### `core/embedder.py`

Wraps the embedding model used by the validation pipeline for semantic retrieval.

```python
MultimodalEmbedder(
    model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
    backend: str = "transformers",   # "transformers" | "vllm"
    device: str = "cuda",
    max_length: int = 4096,
    batch_size: int = 32,
)
```

| Method | Description |
|---|---|
| `embed_texts(texts: List[str])` | Embed facts for indexing; returns `(N, dim)` float32 L2-normalised |
| `embed_query(text, frames)` | Embed a query (text + optional visual frames); returns `(1, dim)` |
| `dim` (property) | Embedding dimension, inferred on first call |

When both text and frames are provided to `embed_query`, the text is prefixed with `"[with visual content]"`. Visual-only queries use the placeholder `"[visual content submitted for fact-checking]"`.

**Factory:** `get_embedder(config) -> MultimodalEmbedder` (singleton).

---

### `core/audio.py`

Audio extraction and Whisper transcription.

| Function | Description |
|---|---|
| `transcribe_video(mp4_path, config, tmp_dir)` | Extract WAV, transcribe, clean up; returns `{segments, language, language_probability}` |
| `transcribe_wav(wav_path, config)` | Transcribe a WAV file; tries faster-whisper (GPU) → transformers (GPU) → faster-whisper (CPU int8) |
| `get_full_text(segments)` | Concatenate all segment texts |
| `extract_wav(mp4_path, wav_path, sr=16000)` | ffmpeg audio extraction |

Each segment in the returned list has `{start, end, text}` in seconds.

---

### `core/overlay.py`

Renders scene graph triplets as an annotation panel on images and videos.

#### Public API

```python
annotate_image_with_triplets_panel(
    image_path: str,
    triplets: List[Triplet],
    out_path: str,
    max_triplets: int = 60,
    panel_ratio: float = 0.4,
) -> None
# Portrait image (h > w): bottom panel; landscape: right-side panel.

annotate_video_with_triplets_panel(
    video_path: str,
    merged_srt_path: str,
    out_path: str,
    panel_ratio: float = 0.45,
    max_triplets: int = 40,
    keep_audio: bool = True,
    panel_position: str = "side",   # "side" | "bottom"
) -> None
# Panel height is computed from actual SRT content to avoid empty space.
# Output is H.264-encoded via ffmpeg for browser compatibility.
```

#### Panel sizing

- **Bottom panel height** is derived from the maximum triplet count across all SRT segments, the font metrics at `scale=0.5`, and the number of columns that fit in the frame width. This prevents a 3-triplet segment from occupying a 500 px empty panel.
- **Side panel width** defaults to `max(240, frame_w × panel_ratio)`.

#### Internal helpers

| Function | Description |
|---|---|
| `_text_metrics(scale)` | Returns `(char_width, line_height)` using `cv2.getTextSize("Xg", …)` to include descenders |
| `_side_panel(…)` | Adaptive font scale, single column |
| `_bottom_panel(…)` | Fixed `scale=0.5`, multi-column layout |
| `_paginate(lines, max_lines, ms_now, seg_start, seg_end)` | Page-advance based on playback position within segment |
| `parse_srt_triplets(path)` | Parse SRT file into `List[SRTBlock]` |
| `_encode_h264(src, out_path, audio_src)` | ffmpeg H.264 re-encode with optional audio mux |

---

### `core/gpu_utils.py`

```python
assign_gpus(model_config: Dict) -> Dict[str, str]
# Returns {"mllm": "cuda:0", "whisper": "cuda:1"|"cpu", "embed": "cuda:2"|"cpu"}

whisper_compute_type(device: str, preferred: str = "float16") -> str
# Returns "float16" on GPU, "int8" on CPU.
```

---

### `validation/pipeline.py`

RAG-based fact-checking pipeline.

#### `ValidationReport`

```python
@dataclass
class ValidationReport:
    database: str
    report: str               # Free-form MLLM analysis
    retrieved_facts: List[str]
    transcript: str = ""
    num_facts_found: int = 0

    def format_discord(self, max_chars=1900) -> str: ...
    def to_dict(self) -> Dict: ...
```

#### `ValidationPipeline`

```python
ValidationPipeline(
    backend: BaseMLLM,
    db: FactDatabase,
    whisper_config: Dict,
    top_k: int = 5,
    num_frames: int = 8,
    tmp_dir: str = "/tmp",
)
```

```python
pipeline.validate(
    database: str,
    media_path: Optional[str] = None,
    text: str = "",
    top_k: Optional[int] = None,
) -> ValidationReport
```

**Validation flow:**

1. If `media_path` is a video: extract and transcribe audio.
2. If `media_path` is an image: sample frames for embedding.
3. Embed the query (text + transcript + optional frames) via `MultimodalEmbedder`.
4. Retrieve `top_k` most similar facts from `FactDatabase`.
5. Call `backend.generate_raw()` with facts + input to produce a free-form analysis report.
6. Return `ValidationReport`.

---

### `validation/database.py`

ChromaDB wrapper with **variant caching**: the same logical database can have embeddings stored under multiple embedding models simultaneously (up to `max_variants=2`). When a new embedding model is used, it backfills from the most recent existing variant. The least-recently-used variant is evicted when the cache is full.

```python
FactDatabase(
    db_path: str,
    embedder: MultimodalEmbedder,
    max_variants: int = 2,
)
```

| Method | Description |
|---|---|
| `list_databases()` | `List[str]` of logical database names |
| `create_database(name)` | Register a database (creates no collection until facts are added) |
| `delete_database(name)` | Delete all variants |
| `add_facts(db_name, facts, source, tags)` | Embed and insert; returns assigned IDs |
| `list_facts(db_name, limit, offset, query)` | Paginated list or semantic search |
| `delete_facts(db_name, ids)` | Remove from all active variants |
| `count(db_name)` | Total fact count in current variant |

Variant metadata is persisted in `db_path/variants_meta.json`.

---

## Web API

### `src/api/app.py`

```python
create_app(services: ModelServices) -> FastAPI
```

- Mounts all routers under `/api`.
- Serves the `frontend/` directory at `/` (SPA).
- Starts the `TaskQueue` worker on startup.

### `src/api/task_queue.py`

```python
class TaskQueue:
    async def start() -> None
    async def submit(fn: Callable[[], Any]) -> str   # returns task_id
    def get(task_id: str) -> Optional[TaskInfo]
```

`TaskInfo` statuses: `pending` → `running` → `done` | `error`. Completed tasks are purged after 1 hour (TTL). Overlay files are deleted on purge.

### Routes

#### Scene Graph — `POST /api/scenegraph`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Upload | — | Video, image, or audio |
| `text` | string | — | Additional context or text-only input |
| `output_type` | string | `"json"` | `"json"` or `"overlay"` |
| `mode` | string | `"high"` | `"high"` (semantic) or `"low"` (visual) |
| `num_frames` | int | — | Override frames per segment |
| `temperature` | float | — | Override sampling temperature |

Returns `{"task_id": "…"}`. Poll `GET /api/tasks/{task_id}`.

Result shape when done:
```json
{
  "segments": [
    {
      "start": 0.0, "end": 12.4,
      "triplets": [{"subject": "…", "relation": "…", "object": "…"}]
    }
  ],
  "overlay_path": "/tmp/…_overlay.mp4" | null
}
```

#### Validation — `POST /api/validate`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | Upload | — | Video, image, or audio |
| `text` | string | — | Claim or context |
| `database` | string | config default | Database to search |
| `top_k` | int | config default | Facts to retrieve |

Returns `{"task_id": "…"}`. Result is `ValidationReport.to_dict()`.

#### Databases — `/api/databases/…`

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/databases` | List all databases with fact counts |
| `POST` | `/api/databases` | Create database `{"name": "…"}` |
| `DELETE` | `/api/databases/{name}` | Delete database and all its facts |
| `GET` | `/api/databases/{name}/facts` | List facts; optional `?query=` for semantic search |
| `POST` | `/api/databases/{name}/facts` | Add facts (file upload or `facts_text`, semicolon-separated); returns `{task_id}` |
| `DELETE` | `/api/databases/{name}/facts` | Delete facts by ID; body: `["id1", "id2", …]` |

#### Tasks — `/api/tasks/…`

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/tasks/{id}` | Poll task; returns `{task_id, status, result, error}` |
| `GET` | `/api/tasks/{id}/file` | Download overlay file once task is `done` |

#### Prompts — `/api/prompts/…`

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/prompts` | List all prompt slots with metadata and current templates |
| `PUT` | `/api/prompts/{name}` | Save a custom template; body: `{"template": "…"}` |
| `DELETE` | `/api/prompts/{name}` | Reset slot to built-in default; returns the restored template |

Each entry in the `GET` response includes:

```json
{
  "name": "scenegraph_visual_high",
  "label": "Scene Graph — Visual, High-Level Mode",
  "description": "…",
  "variables": { "transcript_section": "…" },
  "template": "… current active template …",
  "default_template": "… built-in default …",
  "is_custom": false
}
```

Custom templates are persisted to `{DATA_DIR}/data/prompts.json` and survive restarts.

---

## Web Frontend

Four-tab SPA served from `frontend/`.

**Scene Graph tab**
- Drag-and-drop upload (video, image, audio) or text-only input.
- Controls: output type, frames per segment, temperature, mode (high / low).
- Results: segment cards with triplet rows; overlay media player with download button.

**Validate tab**
- Media upload or text claim.
- Database and top-K selectors.
- Results: validation report from the model, retrieved facts list, transcript.

**Databases tab**
- Create / delete databases.
- Add facts via text input (semicolon-separated) or `.txt` file upload (one fact per line).
- Browse facts with pagination; semantic search; checkbox multi-select delete.

**Prompts tab**
- Six collapsible prompt cards — one per MLLM operation.
- Each card shows: label, description, available `{variable}` placeholders and their descriptions.
- Prompts are read-only by default; click **Edit** to open the textarea for editing.
- **Save** persists the custom template to disk immediately (takes effect for all subsequent requests).
- **Reset to default** reverts to the built-in template.
- Cards with active customizations show a **Modified** badge.

**Polling pattern:** Long operations POST to return a `task_id`, then `app.js` calls `GET /api/tasks/{id}` every 2 seconds until status is `done` or `error`.

---

## Discord Bot

### Setup

Set `DISCORD_TOKEN` in the environment (or `.env` file). Set `discord.guild_id` in `config.yaml` for instant command registration; omit for global (up to 1 hour propagation).

### Commands

| Command | Description |
|---|---|
| `/scenegraph [media] [text] [output_type] [temperature] [num_frames]` | Generate a scene graph. Responds with JSON or uploads overlay file. |
| `/validate [media] [text] [database] [top_k] [show_facts]` | Fact-check against a database. |
| `/add_facts [facts] [file] [database] [tags] [source]` | Add facts as semicolon-separated text or `.txt` attachment. |
| `/list_facts [database] [limit] [offset] [query]` | List or semantically search facts. |
| `/delete_facts <fact_ids> [database]` | Delete facts by comma-separated IDs. |
| `/create_database <name>` | Create a new fact database. |
| `/list_databases` | List all databases with counts. |
| `/help` | Show command overview. |

File uploads are capped at `discord.max_upload_mb` (default 10 MB). Overlays exceeding the Discord upload limit automatically fall back to JSON.

---

## Running the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (Discord bot + web + misinfo, one shared vLLM)
DISCORD_TOKEN=your_token python run.py --port 8080

# Relocate all ML caches / temp files (optional)
PRIDE_CACHE_DIR=/data/.cache PRIDE_TMP_DIR=/data/tmp \
    DISCORD_TOKEN=your_token python run.py --port 8080

# Development: hot-patch code without restarting (GPU models stay loaded)
python -m jurigged -w src/ run.py --port 8080
```

### Accessing remotely (Vast.ai / Docker)

The server binds to `0.0.0.0` but inside a container. A Caddy reverse proxy forwards external port → container port with basic-auth. Check `VAST_TCP_PORT_*` environment variables for the mapped external port.

---

## Docker Deployment

The recommended way to run PRIDE on a fresh machine. The image bundles all Python dependencies and the CUDA runtime — only NVIDIA drivers need to be on the host.

### Prerequisites

#### 1. NVIDIA drivers

```bash
# Ubuntu/Debian — install the driver recommended for your GPU
sudo apt-get install -y nvidia-driver-550
sudo reboot
nvidia-smi          # verify: should print your GPU and driver version
```

For other distributions or driver versions see the [NVIDIA driver download page](https://www.nvidia.com/Download/index.aspx).

#### 2. Docker Engine

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # run Docker without sudo
newgrp docker                   # apply the group change in this shell
```

#### 3. NVIDIA Container Toolkit

Required so Docker containers can access the GPU.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Quick sanity-check — should print your GPU table
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

### Deployment

#### 1. Get the code

```bash
git clone <repo-url> pride_server
cd pride_server
```

Or copy the project directory to the machine however you prefer.

#### 2. Configure `.env`

```bash
cp .env.example .env
nano .env
```

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | Yes | Your Discord bot token |
| `HUGGING_FACE_HUB_TOKEN` | Only for gated models | HuggingFace access token |
| `PRIDE_CACHE_DIR` | No | Base dir for ML caches + scratch tmp (default `/workspace/.cache`) |
| `DATA_DIR` | No | Host path for the database (defaults to `./pride-data` next to `docker-compose.yml`) |

`.env` holds only secrets and storage locations. All model / pipeline settings (model name, backend, fps, temperature, whisper, paths, …) live in `config.yaml` — edit them there.

#### 3. Build the image

```bash
docker compose build
```

Takes 5–15 minutes on first run while Python packages are installed. Model weights (~20 GB) are **not** downloaded during the build — they are fetched from HuggingFace on first startup.

#### 4. Run

The stack runs as a single service (Discord bot + web + misinfo, one shared vLLM). Requires `DISCORD_TOKEN` in `.env`:

```bash
docker compose up -d
```

The UI is available at `http://localhost:8080`. To expose it on the network, open port 8080 in your firewall / security group.

#### 5. Watch the first-run model download

The first startup downloads the configured model weights from HuggingFace (~20 GB for the default `Qwen3-VL-4B-Instruct-FP8`). Watch progress with:

```bash
docker compose logs -f
```

Subsequent starts reuse the cached weights from the `hf_cache` Docker volume and are much faster.

---

### Day-to-day operations

**View live logs:**

```bash
docker compose logs -f
```

**Stop the server:**

```bash
docker compose down
```

**Update to a new version:**

```bash
git pull                            # or copy in new files
docker compose build --no-cache
docker compose up -d
```

The `hf_cache` (model weights) and database volumes are preserved across rebuilds.

---

### Database location

By default the ChromaDB database is stored in `./pride-data/` next to `docker-compose.yml`. To store it on a different drive or path, set `DATA_DIR` in `.env`:

```
DATA_DIR=/mnt/external-drive/pride-db
```

To back up or migrate the database, copy the directory at that path while the server is stopped.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch`, `transformers`, `accelerate` | Core deep learning / HuggingFace |
| `vllm` | High-throughput batched inference (primary backend) |
| `qwen-vl-utils` | Qwen3-VL frame preprocessing |
| `faster-whisper` | CTranslate2-based Whisper ASR |
| `Pillow`, `opencv-python`, `numpy` | Image/video processing and overlay rendering |
| `chromadb` | Vector database for fact storage |
| `sentence-transformers` | Embedding utilities |
| `fastapi`, `uvicorn`, `python-multipart` | Web API server |
| `discord.py`, `aiohttp`, `aiofiles` | Discord bot |
| `PyYAML`, `python-dotenv` | Configuration |
| `jurigged` | Hot-patching for development |
