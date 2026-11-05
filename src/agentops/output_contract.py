"""Per-agent output contract: bounded, safe JSON / JSON Schema validation.

Mirrors ``packages/shared/src/guardrails/outputContract.ts`` (the server /
live-scan implementation). Both are exercised by one shared fixture file
(``packages/shared/src/guardrails/__fixtures__/outputContractCases.json``) so
they cannot drift.

The schema validator is a small, bounded SUBSET of JSON Schema, written from
scratch on purpose:

* It never fetches anything. ``$ref`` may only point inside the same document
  (``#``, ``#/$defs/x`` ...); any other ``$ref`` is refused, because resolving
  a remote reference from a customer-supplied schema is an SSRF vector.
* It never compiles user regular expressions (``pattern`` /
  ``patternProperties`` are refused): the validated text is model output, so a
  hostile pattern is a ReDoS vector that cannot be timed out.
* Unsupported keywords are refused instead of silently ignored.
* Schema size, depth, node count, recursion depth and evaluation steps are
  capped; exceeding a cap fails closed (invalid) rather than hanging.

Errors carry JSON-pointer paths and keyword names only, never output content.
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

MAX_SCHEMA_BYTES = 16_384
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_NODES = 500
MAX_ERRORS = 5
_MAX_RECURSION = 200
_MAX_STEPS = 100_000

_ANNOTATIONS = frozenset(
    {
        "$schema",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        # ``format`` is an annotation in JSON Schema 2020-12 and is NOT asserted.
        "format",
    }
)
_SUPPORTED = frozenset(
    {
        "type", "enum", "const", "properties", "required", "additionalProperties",
        "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
        "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
        "exclusiveMaximum", "minProperties", "maxProperties", "anyOf", "oneOf",
        "allOf", "not", "$ref", "$defs", "definitions",
    }
)
_TYPE_NAMES = frozenset({"null", "boolean", "object", "array", "number", "integer", "string"})


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _resolve_local_ref(root: dict, ref: str) -> Optional[dict]:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    cur: Any = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, list):
            if not (token.isdigit() and (token == "0" or not token.startswith("0"))):
                return None
            idx = int(token)
            if idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict) and token in cur:
            cur = cur[token]
        else:
            return None
    return cur if isinstance(cur, dict) else None


def validate_schema_document(schema: Any) -> list[str]:
    """Return problems with the schema document itself (empty = acceptable)."""
    if not isinstance(schema, dict):
        return ["schema must be a JSON object"]
    try:
        size = len(json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return ["schema is not serializable JSON"]
    if size > MAX_SCHEMA_BYTES:
        return [f"schema is {size} bytes; the limit is {MAX_SCHEMA_BYTES}"]

    problems: list[str] = []
    nodes = 0

    def fail(msg: str) -> None:
        if len(problems) < 20:
            problems.append(msg)

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal nodes
        if len(problems) >= 20:
            return
        if depth > MAX_SCHEMA_DEPTH:
            fail(f"{path or '/'}: nested deeper than {MAX_SCHEMA_DEPTH} levels")
            return
        if not isinstance(node, dict):
            fail(f"{path or '/'}: a schema must be an object")
            return
        nodes += 1
        if nodes > MAX_SCHEMA_NODES:
            if nodes == MAX_SCHEMA_NODES + 1:
                fail(f"schema has more than {MAX_SCHEMA_NODES} nodes")
            return
        for key in node:
            if key in _ANNOTATIONS:
                continue
            if key not in _SUPPORTED:
                why = (
                    "regular expressions are not supported"
                    if key in ("pattern", "patternProperties")
                    else "unsupported keyword"
                )
                fail(f'{path or "/"}: "{key}": {why}')

        t = node.get("type")
        if "type" in node:
            names = t if isinstance(t, list) else [t]
            if len(names) == 0 or not all(isinstance(n, str) and n in _TYPE_NAMES for n in names):
                fail(f'{path or "/"}: "type" must be a JSON type name or a list of them')
        if "enum" in node and not isinstance(node["enum"], list):
            fail(f'{path or "/"}: "enum" must be an array')
        if "required" in node:
            r = node["required"]
            if not isinstance(r, list) or not all(isinstance(x, str) for x in r):
                fail(f'{path or "/"}: "required" must be an array of strings')
        for k in ("minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"):
            if k in node and not (_is_int(node[k]) and node[k] >= 0):
                fail(f'{path or "/"}: "{k}" must be a non-negative integer')
        for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if k in node and not (_is_number(node[k]) and math.isfinite(node[k])):
                fail(f'{path or "/"}: "{k}" must be a finite number')
        if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
            fail(f'{path or "/"}: "uniqueItems" must be a boolean')

        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                fail(f'{path or "/"}: "$ref" must be a string')
            elif not ref.startswith("#"):
                fail(
                    f'{path or "/"}: "$ref" must point inside this schema (start with "#"); '
                    "remote references are never fetched"
                )
            elif _resolve_local_ref(schema, ref) is None:
                fail(f'{path or "/"}: "$ref" "{ref}" does not resolve in this schema')

        if "properties" in node:
            if not isinstance(node["properties"], dict):
                fail(f'{path or "/"}: "properties" must be an object')
            else:
                for k, v in node["properties"].items():
                    walk(v, f"{path}/properties/{k}", depth + 1)
        for k in ("$defs", "definitions"):
            if k in node:
                if not isinstance(node[k], dict):
                    fail(f'{path or "/"}: "{k}" must be an object')
                else:
                    for dk, dv in node[k].items():
                        walk(dv, f"{path}/{k}/{dk}", depth + 1)
        if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
            walk(node["additionalProperties"], f"{path}/additionalProperties", depth + 1)
        if "items" in node:
            walk(node["items"], f"{path}/items", depth + 1)
        if "not" in node:
            walk(node["not"], f"{path}/not", depth + 1)
        for k in ("prefixItems", "anyOf", "oneOf", "allOf"):
            if k in node:
                lst = node[k]
                if not isinstance(lst, list) or len(lst) == 0:
                    fail(f'{path or "/"}: "{k}" must be a non-empty array')
                else:
                    for i, v in enumerate(lst):
                        walk(v, f"{path}/{k}/{i}", depth + 1)

    walk(schema, "", 0)
    return problems


def parse_output_contract(raw: Any) -> tuple[Optional[dict], Optional[str]]:
    """Parse an untrusted ``outputContract`` value into ``(contract, error)``."""
    if not isinstance(raw, dict):
        return None, "outputContract must be an object"
    extra = [k for k in raw if k not in ("kind", "schema")]
    if extra:
        return None, f'outputContract has unknown field "{extra[0]}"'
    kind = raw.get("kind")
    if kind in ("none", "json"):
        if "schema" in raw:
            return None, f'outputContract kind "{kind}" takes no schema'
        return {"kind": kind}, None
    if kind == "json_schema":
        problems = validate_schema_document(raw.get("schema"))
        if problems:
            return None, "outputContract schema is invalid: " + "; ".join(problems[:5])
        return {"kind": "json_schema", "schema": raw["schema"]}, None
    return None, 'outputContract.kind must be "none", "json" or "json_schema"'


def resolve_output_contract(post_output: Optional[dict]) -> dict:
    """The contract in force for a ``postOutput`` block (see the TS twin)."""
    if not isinstance(post_output, dict):
        return {"kind": "none"}
    if "outputContract" in post_output:
        contract, _err = parse_output_contract(post_output["outputContract"])
        return contract if contract is not None else {"kind": "none"}
    return {"kind": "json"} if post_output.get("mustBeJson") is True else {"kind": "none"}


class _LimitExceeded(Exception):
    pass


def _json_equal(a: Any, b: Any) -> bool:
    """Strict JSON equality: ``True != 1``, order-insensitive objects."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return len(a) == len(b) and all(k in b and _json_equal(v, b[k]) for k, v in a.items())
    if _is_number(a) and _is_number(b):
        return a == b
    if type(a) is not type(b):
        return False
    return a == b


def _type_matches(name: str, v: Any) -> bool:
    if name == "null":
        return v is None
    if name == "boolean":
        return isinstance(v, bool)
    if name == "string":
        return isinstance(v, str)
    if name == "array":
        return isinstance(v, list)
    if name == "object":
        return isinstance(v, dict)
    if name == "number":
        return _is_number(v)
    if name == "integer":
        return _is_int(v) or (isinstance(v, float) and v.is_integer())
    return False


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _reject_constant(name: str) -> Any:  # NaN / Infinity are not JSON
    raise ValueError(name)


def check_output_contract(contract: dict, text: str) -> dict:
    """Check output text against a contract.

    Returns ``{"ok": True}`` or ``{"ok": False, "code": "json_invalid" |
    "schema_invalid", "errors": [{"path", "keyword"}, ...]}``.
    """
    kind = contract.get("kind")
    if kind == "none":
        return {"ok": True}
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return {"ok": False, "code": "json_invalid", "errors": [{"path": "", "keyword": "json"}]}
    if kind == "json":
        return {"ok": True}

    root = contract["schema"]
    errors: list[dict] = []
    steps = 0

    def note(path: str, keyword: str) -> None:
        if len(errors) < MAX_ERRORS:
            errors.append({"path": path, "keyword": keyword})

    def validate(schema: dict, inst: Any, path: str, depth: int, record: bool) -> bool:
        nonlocal steps
        steps += 1
        if steps > _MAX_STEPS or depth > _MAX_RECURSION:
            raise _LimitExceeded()
        ok = True

        def bad(keyword: str) -> None:
            nonlocal ok
            ok = False
            if record:
                note(path, keyword)

        if isinstance(schema.get("$ref"), str):
            target = _resolve_local_ref(root, schema["$ref"])
            if target is None:
                bad("$ref")
            elif not validate(target, inst, path, depth + 1, record):
                ok = False

        if "type" in schema:
            t = schema["type"]
            names = t if isinstance(t, list) else [t]
            if not any(isinstance(n, str) and _type_matches(n, inst) for n in names):
                bad("type")
        if isinstance(schema.get("enum"), list):
            if not any(_json_equal(e, inst) for e in schema["enum"]):
                bad("enum")
        if "const" in schema:
            if not _json_equal(schema["const"], inst):
                bad("const")

        if _is_number(inst):
            if _is_number(schema.get("minimum")) and inst < schema["minimum"]:
                bad("minimum")
            if _is_number(schema.get("maximum")) and inst > schema["maximum"]:
                bad("maximum")
            if _is_number(schema.get("exclusiveMinimum")) and inst <= schema["exclusiveMinimum"]:
                bad("exclusiveMinimum")
            if _is_number(schema.get("exclusiveMaximum")) and inst >= schema["exclusiveMaximum"]:
                bad("exclusiveMaximum")

        if isinstance(inst, str):
            n = len(inst)  # code points, like the TS twin
            if _is_int(schema.get("minLength")) and n < schema["minLength"]:
                bad("minLength")
            if _is_int(schema.get("maxLength")) and n > schema["maxLength"]:
                bad("maxLength")

        if isinstance(inst, list):
            if _is_int(schema.get("minItems")) and len(inst) < schema["minItems"]:
                bad("minItems")
            if _is_int(schema.get("maxItems")) and len(inst) > schema["maxItems"]:
                bad("maxItems")
            if schema.get("uniqueItems") is True:
                dup = False
                for i in range(len(inst)):
                    for j in range(i):
                        steps += 1
                        if steps > _MAX_STEPS:
                            raise _LimitExceeded()
                        if _json_equal(inst[i], inst[j]):
                            dup = True
                            break
                    if dup:
                        break
                if dup:
                    bad("uniqueItems")
            prefix = schema["prefixItems"] if isinstance(schema.get("prefixItems"), list) else []
            for i, s in enumerate(prefix):
                if i < len(inst) and isinstance(s, dict):
                    if not validate(s, inst[i], f"{path}/{i}", depth + 1, record):
                        ok = False
            if isinstance(schema.get("items"), dict):
                for i in range(len(prefix), len(inst)):
                    if not validate(schema["items"], inst[i], f"{path}/{i}", depth + 1, record):
                        ok = False

        if isinstance(inst, dict):
            keys = list(inst.keys())
            if _is_int(schema.get("minProperties")) and len(keys) < schema["minProperties"]:
                bad("minProperties")
            if _is_int(schema.get("maxProperties")) and len(keys) > schema["maxProperties"]:
                bad("maxProperties")
            if isinstance(schema.get("required"), list):
                for r in schema["required"]:
                    if isinstance(r, str) and r not in inst:
                        ok = False
                        if record:
                            note(f"{path}/{_escape(r)}", "required")
            props = schema["properties"] if isinstance(schema.get("properties"), dict) else {}
            for k, sub in props.items():
                if k in inst and isinstance(sub, dict):
                    if not validate(sub, inst[k], f"{path}/{_escape(k)}", depth + 1, record):
                        ok = False
            if "additionalProperties" in schema:
                ap = schema["additionalProperties"]
                for k in keys:
                    if k in props:
                        continue
                    if ap is False:
                        ok = False
                        if record:
                            note(f"{path}/{_escape(k)}", "additionalProperties")
                    elif isinstance(ap, dict):
                        if not validate(ap, inst[k], f"{path}/{_escape(k)}", depth + 1, record):
                            ok = False

        if isinstance(schema.get("allOf"), list):
            for s in schema["allOf"]:
                if isinstance(s, dict) and not validate(s, inst, path, depth + 1, record):
                    ok = False
        if isinstance(schema.get("anyOf"), list):
            if not any(isinstance(s, dict) and validate(s, inst, path, depth + 1, False) for s in schema["anyOf"]):
                bad("anyOf")
        if isinstance(schema.get("oneOf"), list):
            matches = sum(1 for s in schema["oneOf"] if isinstance(s, dict) and validate(s, inst, path, depth + 1, False))
            if matches != 1:
                bad("oneOf")
        if isinstance(schema.get("not"), dict):
            if validate(schema["not"], inst, path, depth + 1, False):
                bad("not")
        return ok

    try:
        if validate(root, value, "", 0, True):
            return {"ok": True}
        return {
            "ok": False,
            "code": "schema_invalid",
            "errors": errors or [{"path": "", "keyword": "schema"}],
        }
    except (_LimitExceeded, RecursionError):
        return {
            "ok": False,
            "code": "schema_invalid",
            "errors": [{"path": "", "keyword": "evaluationLimit"}],
        }
