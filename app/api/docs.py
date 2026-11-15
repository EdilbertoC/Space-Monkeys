# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Human documentation built from the hand-written contract, docs/api/openapi.yaml.

FastAPI's own schema (/openapi.json) is generated from the code and stays as is; the
contract adds the units, conventions and real example values written for people.

- /docs: Swagger UI over the contract, with "Try it out" pointed at this server.
- /redoc: the same contract as a reading page.
- /contract.json: the contract as JSON.
"""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "api" / "openapi.yaml"


@lru_cache
def _contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


@router.get("/contract.json")
def contract() -> dict:
    if not CONTRACT.exists():
        raise HTTPException(404, "docs/api/openapi.yaml is not deployed with this server")
    spec = deepcopy(_contract())
    # A relative server URL makes "Try it out" call the host and scheme the page was
    # opened with. Behind an HTTPS proxy the app itself sees plain HTTP, and an absolute
    # http:// URL from an https:// page is blocked by the browser ("Failed to fetch").
    spec["servers"] = [{"url": "/", "description": "This server"}] + spec.get("servers", [])
    return spec


@router.get("/docs", response_class=HTMLResponse)
def swagger() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/contract.json",
        title="OrbitBurn API — docs",
        swagger_ui_parameters={
            "docExpansion": "none",
            "defaultModelsExpandDepth": 0,
            "displayRequestDuration": True,
            "tryItOutEnabled": True,
        },
    )


@router.get("/redoc", response_class=HTMLResponse)
def redoc() -> HTMLResponse:
    return get_redoc_html(openapi_url="/contract.json", title="OrbitBurn API — reference")
