"""
Knowledge-table helpers: parse JSON table payloads and search rows client-side.

Table bodies match the AgentOps UI format: ``{"columns": [...], "rows": [...]}``
with stable column ``id`` keys on each row (see ``web/src/features/knowledge-tables/lib/types.ts``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, MutableMapping, Sequence, TypedDict


TABLES_PREFIX = "tables/"
CMS_PREFIX = "cms/"


class TableColumn(TypedDict, total=False):
    id: str
    name: str
    type: str
    required: bool


class TableData(TypedDict):
    columns: list[TableColumn]
    rows: list[dict[str, Any]]


class SimilarityHit(TypedDict):
    """One row from :func:`table_similarity_rows`."""

    row: dict[str, Any]
    score: float
    matched_column: str | None


def ensure_tables_name(name: str) -> str:
    """Return ``tables/...`` — add prefix if missing."""

    n = name.strip().lstrip("/")
    if n.startswith(TABLES_PREFIX):
        return n
    return f"{TABLES_PREFIX}{n}"


def ensure_cms_name(name: str) -> str:
    """Return ``cms/...`` — add prefix if missing."""

    n = name.strip().lstrip("/")
    if n.startswith(CMS_PREFIX):
        return n
    return f"{CMS_PREFIX}{n}"


def _prompt_body(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, Mapping):
        for key in ("prompt", "content", "body"):
            v = result.get(key)
            if isinstance(v, str):
                return v
        return ""
    for attr in ("prompt", "content", "body"):
        v = getattr(result, attr, None)
        if isinstance(v, str):
            return v
    return ""


def parse_table_from_prompt_response(result: Any) -> TableData:
    """Extract and validate table JSON from a :meth:`~agentops.client.AgentOps.get_prompt` / ``get_table`` result."""

    raw = _prompt_body(result)
    if not raw.strip():
        raise ValueError("Empty prompt body; cannot parse table JSON")
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("Prompt body is not valid JSON for a table") from e
    if not isinstance(parsed, dict):
        raise ValueError("Table JSON must be an object")
    cols = parsed.get("columns")
    rows = parsed.get("rows")
    if not isinstance(cols, list) or not isinstance(rows, list):
        raise ValueError("Table JSON must have 'columns' and 'rows' arrays")
    return TableData(columns=cols, rows=rows)


def row_as_name_map(data: TableData, row: Mapping[str, Any]) -> dict[str, Any]:
    """Map a row from column-id keys to human-readable column names."""

    out: dict[str, Any] = {}
    for col in data["columns"]:
        cid = col.get("id")
        name = col.get("name", cid)
        if cid is not None and cid in row:
            out[str(name)] = row[cid]
    return out


def table_lookup(
    data: TableData,
    column_name: str,
    value: Any,
) -> list[dict[str, Any]]:
    """Return rows where ``column_name`` equals ``value`` (exact, by column **name**)."""

    col = next(
        (c for c in data["columns"] if c.get("name") == column_name),
        None,
    )
    if not col:
        return []
    cid = col.get("id")
    if not cid:
        return []
    return [r for r in data["rows"] if r.get(cid) == value]


def table_search(
    data: TableData,
    query: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Substring match (case-insensitive) across all columns; each row at most once."""

    q = query.lower()
    matches: list[dict[str, Any]] = []
    for row in data["rows"]:
        for col in data["columns"]:
            cid = col.get("id")
            if cid is None:
                continue
            val = str(row.get(cid, "")).lower()
            if q in val:
                matches.append(row)
                break
        if limit is not None and len(matches) >= limit:
            break
    return matches


def _levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    matrix: list[list[int]] = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        matrix[i][0] = i
    for j in range(lb + 1):
        matrix[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
    return matrix[la][lb]


def _fuzzy_score(needle: str, haystack: str) -> float:
    n = needle.lower()
    h = haystack.lower()
    if h == n:
        return 1.0
    if h.startswith(n):
        return 0.95
    if n in h:
        return 0.9
    dist = _levenshtein(n, h)
    max_len = max(len(n), len(h))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - dist / max_len)


def table_similarity_rows(
    data: TableData,
    query: str,
    *,
    limit: int = 20,
    threshold: float = 0.4,
) -> list[SimilarityHit]:
    """
    Fuzzy similarity across all columns (aligned with product ``lookupRows`` fuzzy mode).

    Rows are sorted by descending score; at most ``limit`` hits are returned.
    """

    results: list[SimilarityHit] = []
    for row in data["rows"]:
        best_score = 0.0
        best_col: str | None = None
        for col in data["columns"]:
            cid = col.get("id")
            if cid is None:
                continue
            cell = row.get(cid)
            if cell is None:
                continue
            score = _fuzzy_score(query, str(cell))
            if score > best_score:
                best_score = score
                best_col = str(col.get("name", cid))
        if best_score > threshold:
            results.append(
                SimilarityHit(
                    row=row,
                    score=best_score,
                    matched_column=best_col,
                )
            )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def table_similarity_named(
    data: TableData,
    query: str,
    *,
    limit: int = 20,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Like :func:`table_similarity_rows` but each ``row`` uses column **names** as keys."""

    hits = table_similarity_rows(
        data, query, limit=limit, threshold=threshold
    )
    out: list[dict[str, Any]] = []
    for h in hits:
        named = row_as_name_map(data, h["row"])
        named["_score"] = h["score"]
        named["_matched_column"] = h["matched_column"]
        out.append(named)
    return out
