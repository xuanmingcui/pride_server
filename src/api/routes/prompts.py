"""GET / PUT / DELETE endpoints for prompt template management."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["prompts"])


class PromptUpdate(BaseModel):
    template: str


@router.get("/prompts")
async def list_prompts(request: Request):
    store = request.app.state.services.prompt_store
    return [
        {
            "name":             name,
            "label":            store.meta(name)["label"],
            "description":      store.meta(name)["description"],
            "variables":        store.meta(name)["variables"],
            "template":         store.get(name),
            "default_template": store.get_default(name),
            "is_custom":        store.is_custom(name),
        }
        for name in store.names()
    ]


@router.put("/prompts/{name}")
async def update_prompt(name: str, body: PromptUpdate, request: Request):
    store = request.app.state.services.prompt_store
    try:
        store.set(name, body.template)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt slot: {name!r}")
    return {"name": name, "is_custom": True}


@router.delete("/prompts/{name}")
async def reset_prompt(name: str, request: Request):
    store = request.app.state.services.prompt_store
    try:
        store.reset(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown prompt slot: {name!r}")
    return {"name": name, "is_custom": False, "template": store.get(name)}
