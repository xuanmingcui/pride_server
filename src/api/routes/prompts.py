"""GET / PUT / DELETE endpoints for prompt template management.

Each prompt slot has two halves — ``system`` and ``user``. The GET response
returns both, along with the built-in defaults so the UI can show diff/reset
state. PUT accepts a partial body: omitted halves inherit the default.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["prompts"])


class PromptUpdate(BaseModel):
    """Payload for PUT /api/prompts/{name}.

    Either field may be omitted to reset that half to the built-in default
    while customizing only the other half.
    """
    system: Optional[str] = None
    user: Optional[str] = None


def _slot_payload(store, name: str):
    meta = store.meta(name)
    return {
        "name":             name,
        "label":            meta["label"],
        "description":      meta["description"],
        "system_variables": meta.get("system_variables", {}),
        "user_variables":   meta.get("user_variables", {}),
        "template":         store.get(name),           # {"system", "user"}
        "default_template": store.get_default(name),   # {"system", "user"}
        "is_custom":        store.is_custom(name),
    }


@router.get("/prompts")
async def list_prompts(request: Request):
    store = request.app.state.services.prompt_store
    return [_slot_payload(store, name) for name in store.names()]


@router.put("/prompts/{name}")
async def update_prompt(name: str, body: PromptUpdate, request: Request):
    store = request.app.state.services.prompt_store
    try:
        store.set(name, system=body.system, user=body.user)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt slot: {name!r}")
    return {"name": name, "is_custom": True, "template": store.get(name)}


@router.delete("/prompts/{name}")
async def reset_prompt(name: str, request: Request):
    store = request.app.state.services.prompt_store
    try:
        store.reset(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt slot: {name!r}")
    return {"name": name, "is_custom": False, "template": store.get(name)}
