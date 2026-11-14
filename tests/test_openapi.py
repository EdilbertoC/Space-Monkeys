# SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README.
"""Contract tests for docs/api/openapi.yaml, the API contract with the frontend."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
jsonschema = pytest.importorskip("jsonschema")
validator = pytest.importorskip("openapi_spec_validator")
referencing = pytest.importorskip("referencing")

from referencing.jsonschema import DRAFT202012  # noqa: E402

from app.main import app  # noqa: E402

SPEC_PATH = Path(__file__).resolve().parent.parent / "docs" / "api" / "openapi.yaml"
SPEC = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _examples():
    for path, item in SPEC["paths"].items():
        for method, op in item.items():
            for resp in op.get("responses", {}).values():
                for media in resp.get("content", {}).values():
                    for name, ex in (media.get("examples") or {}).items():
                        yield pytest.param(
                            media["schema"], ex["value"], id=f"{method} {path} {name}"
                        )


def test_spec_is_valid_openapi():
    validator.validate(SPEC)


@pytest.mark.parametrize(("schema", "value"), list(_examples()))
def test_examples_match_their_schema(schema, value):
    registry = referencing.Registry().with_resource(
        "urn:spec", referencing.Resource.from_contents(SPEC, default_specification=DRAFT202012)
    )
    if "$ref" in schema:
        schema = {"$ref": "urn:spec" + schema["$ref"]}
    jsonschema.Draft202012Validator(
        schema, registry=registry, format_checker=jsonschema.FormatChecker()
    ).validate(value)


def test_every_running_route_is_in_the_contract():
    running = {(path, method) for path, item in app.openapi()["paths"].items() for method in item}
    documented = {(path, method) for path, item in SPEC["paths"].items() for method in item}
    assert running <= documented, f"routes missing from the contract: {running - documented}"
