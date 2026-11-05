"""One view of a replay file, as text (``show``) and as HTML (``render``).

``entry_view`` reads an entry once (title, prompts, tools, replies, repeats,
feedback with its assessments); ``summary`` and ``render_html`` only format
it. ``run.events`` (large, only for publishing) is never shown. Every HTML
value is escaped: replies, notes and traces are untrusted text.
"""

from __future__ import annotations

import json
import textwrap
from html import escape
from typing import Any, Optional

# Characters kept of a message, reply or note in ``show`` unless ``--full``.
TRUNCATE = 600


def _versioned(ref: Optional[dict]) -> Optional[tuple]:
    if not ref or not ref.get("name") or ref.get("candidate") or ref.get("version") is None:
        return None
    return (ref["name"], ref["version"])


def describe_prompt(ref: Optional[dict]) -> str:
    """``support v7``, ``support candidate (base v7)`` or ``no AOS prompt``."""
    if not ref or not ref.get("name"):
        return "no AOS prompt"
    if ref.get("candidate"):
        return f"{ref['name']} candidate (base v{ref.get('base_version')})"
    if ref.get("version") is None:
        return str(ref["name"])
    return f"{ref['name']} v{ref['version']}"


def prompt_changed(entry: dict) -> tuple[Optional[bool], str]:
    """Whether the replay ran another prompt: by ``(name, version)`` when both
    sides have one, else by the final system prompt text; and how it knew."""
    source, run = entry.get("source") or {}, entry.get("run") or {}
    before, after = _versioned(source.get("prompt")), _versioned(run.get("prompt"))
    if before and after:
        return before != after, "versions"
    texts = source.get("system_prompt"), run.get("system_prompt")
    if None in texts:
        return None, ""
    return texts[0] != texts[1], "text"


def tools_used(run: dict) -> list[str]:
    """Tool names the run called, in order: the calls in ``run.trace`` after
    its last user message (earlier ones belong to the history)."""
    trace = run.get("trace") or []
    users = [i for i, m in enumerate(trace) if isinstance(m, dict) and m.get("role") == "user"]
    names = []
    for item in trace[users[-1] + 1 :] if users else trace:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name"):
            names.append(str(item["name"]))
        for call in item.get("tool_calls") or []:
            if isinstance(call, dict):
                name = (call.get("function") or {}).get("name") or call.get("name")
                if name:
                    names.append(str(name))
    return names


def entry_view(entry: dict) -> dict:
    """What every view shows of one entry."""
    source = entry.get("source") or {}
    run = entry.get("run") if isinstance(entry.get("run"), dict) else None
    changed, compared = prompt_changed(entry)
    word = {True: "changed", False: "unchanged", None: "unknown"}[changed]
    feedback = []
    for item in entry.get("feedback") or []:
        key = (item.get("feedback_id"), item.get("revision"))
        feedback.append(
            {
                **item,
                "assessments": [
                    r
                    for r in entry.get("resolutions") or []
                    if (r.get("feedback_id"), r.get("feedback_revision")) == key
                ],
            }
        )
    return {
        "title": entry.get("trace_name") or entry.get("agent_name"),
        "ids": f"session {entry.get('session_id')} · trace "
        f"{entry.get('source_trace_id')} · run {entry.get('run_id')}",
        "source": source,
        "run": run,
        "status": (run or {}).get("status") or "not replayed",
        "prompt": f"original {describe_prompt(source.get('prompt'))} → replay "
        f"{describe_prompt((run or {}).get('prompt'))}: {word}"
        + (f" ({compared})" if compared else ""),
        "prompt_word": word,
        "tools": tools_used(run or {}),
        "repeats": [r for r in (run or {}).get("repeats") or [] if isinstance(r, dict)],
        "feedback": feedback,
    }


def _header(artifact: dict) -> tuple[str, str]:
    provenance = artifact.get("provenance")
    entries = artifact.get("entries") or []
    first = (
        f"Replay {artifact.get('name')} ({artifact.get('replay_id')}) · created "
        f"{artifact.get('created_at')} · {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
    )
    if not isinstance(provenance, dict):
        return first, "Not replayed yet (no provenance)."
    dirty = {True: "dirty", False: "clean"}.get(provenance.get("dirty"), "dirty unknown")
    return first, (
        f"Commit {provenance.get('git_commit') or '(none)'} ({dirty}) · "
        f"{provenance.get('provider')} {provenance.get('model')}"
    )


# --- Text (``show``)


def cut(value: Any, limit: Optional[int] = TRUNCATE) -> str:
    """*value* as text, at most *limit* characters plus a note of the rest."""
    if value is None:
        return "(none)"
    text = str(value).strip()
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} … [{len(text) - limit} more chars; --full]"


def _block(label: str, value: Any, limit: Optional[int], indent: str = "  ") -> list[str]:
    return [f"{indent}{label}:", textwrap.indent(cut(value, limit), indent + "  ")]


def _text_entry(view: dict, number: str, limit: Optional[int]) -> list[str]:
    run = view["run"]
    lines = ["", f"[{number}] {view['title']} · {view['ids']}"]
    if run is None:
        lines.append("  Not replayed.")
    else:
        error = f" · error: {cut(run.get('error'), limit)}" if run.get("error") else ""
        lines.append(f"  Status: {view['status']}{error}")
        lines.append(f"  Prompt: {view['prompt']}")
        lines.append(f"  Tools: {' → '.join(view['tools']) or 'none'}")
    lines += _block("User message", view["source"].get("user_message"), limit)
    lines += _block("Original reply", view["source"].get("original_reply"), limit)
    if run is not None:
        lines += _block("Replay reply", run.get("reply"), limit)
    repeats = view["repeats"]
    if repeats:
        runs = [run, *repeats]
        complete = sum(1 for r in runs if (r or {}).get("status") == "complete")
        same = [r.get("reply") is not None and r.get("reply") == run.get("reply") for r in repeats]
        lines.append(
            f"  Runs: {len(runs)} (the run and {len(repeats)} repeat(s)) · complete "
            f"{complete} of {len(runs)} · same reply as the run: {sum(same)} of {len(repeats)}"
        )
        for n, (item, identical) in enumerate(zip(repeats, same), 1):
            head = f"    Repeat {n}: {item.get('status')}"
            if item.get("error"):
                head += f" · error: {cut(item.get('error'), limit)}"
            if identical:
                lines.append(f"{head} · same reply as the run")
            else:
                lines += [
                    f"{head} · reply:",
                    textwrap.indent(cut(item.get("reply"), limit), "      "),
                ]
    for item in view["feedback"]:
        lines.append(
            f"  Feedback {item.get('feedback_id')} rev {item.get('revision')} "
            f"[{item.get('author_name')}, {item.get('scope')}]:"
        )
        if item.get("replay_run_id"):
            lines.append(
                f"    (about Evolve replay run {item['replay_run_id']}, not the original reply)"
            )
        lines.append(textwrap.indent(cut(item.get("note"), limit), "    "))
        for r in item["assessments"]:
            lines.append(
                f"    Assessment ({r.get('status')}, {r.get('author')}): "
                f"{cut(r.get('explanation'), limit)}"
            )
            lines.append(f"    Evidence: {cut(r.get('evidence'), limit)}")
            lines.append(f"    Changes: {cut(r.get('changes'), limit)}")
    if not view["feedback"]:
        lines.append("  No feedback frozen.")
    return lines


def summary(artifact: dict, *, full: bool = False) -> list[str]:
    """The file as text lines: provenance, then every entry."""
    limit = None if full else TRUNCATE
    entries = artifact.get("entries") or []
    lines = list(_header(artifact))
    for n, entry in enumerate(entries, 1):
        lines += _text_entry(entry_view(entry), f"{n}/{len(entries)}", limit)
    return lines


# --- HTML (``render``)

_STYLE = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1f2421;--muted:#5f6b66;--line:#dcdfdc;
--ok:#1d7a46;--bad:#b42318;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#141716;--card:#1d211f;--fg:#e6e9e7;
--muted:#9aa5a0;--line:#333a37;--ok:#5cc489;--bad:#f97066;--warn:#e3b341}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);
max-width:1280px;margin:0 auto;padding:24px 16px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 20px;margin:20px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:4px 0;
font:13px/1.45 ui-monospace,monospace}
.meta,small{color:var(--muted)}
.tag{display:inline-block;padding:0 8px;border-radius:10px;border:1px solid var(--line);
font-size:12px;margin-right:6px}
.fixed,.verified,.complete{color:var(--ok)}
.regressed,.unresolved,.failed{color:var(--bad)}
.partial,.blocked,.changed{color:var(--warn)}
summary{cursor:pointer;color:var(--muted)}
h1{font-size:22px}h2{font-size:17px;margin:4px 0}h3{font-size:14px;margin:12px 0 4px}
"""


def _text(value: Any) -> str:
    if value is None:
        return ""
    return escape(
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    )


def _pre(value: Any) -> str:
    return f"<pre>{_text(value)}</pre>"


def _details(label: str, value: Any) -> str:
    return f"<details><summary>{escape(label)}</summary>{_pre(value)}</details>"


def _tag(word: Any) -> str:
    word = escape(str(word))
    return f"<span class='tag {word}'>{word}</span>"


def _html_entry(view: dict) -> str:
    source, run = view["source"], view["run"] or {}
    feedback = (
        "".join(
            f"<div><small>{_text(item.get('author_name'))} · {_text(item.get('scope'))} · "
            f"feedback {_text(item.get('feedback_id'))} rev {_text(item.get('revision'))}</small>"
            f"{_pre(item.get('note'))}"
            + "".join(
                f"<div>{_tag(r.get('status'))}<small>{_text(r.get('author'))} · "
                f"{_text(r.get('created_at'))}</small>{_pre(r.get('explanation'))}"
                f"<small>Evidence</small>{_pre(r.get('evidence'))}"
                f"<small>Changes</small>{_pre(r.get('changes'))}</div>"
                for r in item["assessments"]
            )
            + "</div>"
            for item in view["feedback"]
        )
        or "<p class='meta'>No feedback frozen.</p>"
    )
    error = f"<h3>Error</h3>{_pre(run.get('error'))}" if run.get("error") else ""
    return (
        f"<section><h2>{_text(view['title'])}</h2>"
        f"<p class='meta'>{_text(view['ids'])} · {_tag(view['status'])}</p>"
        f"<h3>User message</h3>{_pre(source.get('user_message'))}"
        "<div class='cols'>"
        f"<div><h3>Original reply</h3>{_pre(source.get('original_reply'))}"
        f"<small>{_text(source.get('timestamp'))}</small></div>"
        f"<div><h3>Replay reply</h3>{_pre(run.get('reply'))}{error}"
        f"<small>{_text(run.get('completed_at'))}</small></div></div>"
        f"<h3>Prompt {_tag(view['prompt_word'])}</h3><p class='meta'>{_text(view['prompt'])}</p>"
        "<div class='cols'>"
        f"<div>{_details('Original final prompt', source.get('system_prompt'))}</div>"
        f"<div>{_details('Replay final prompt', run.get('system_prompt'))}</div></div>"
        f"<h3>Feedback and assessments</h3>{feedback}"
        f"{_details('Preceding history', source.get('history_before'))}"
        f"{_details('Replay trace', run.get('trace'))}"
        f"{_details('Replay tool calls', run.get('tool_calls'))}"
        "</section>"
    )


def render_html(artifact: dict) -> str:
    title = artifact.get("name") or "Evolve Agent replay"
    first, second = _header(artifact)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_text(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{_text(title)}</h1><p class='meta'>{_text(first)}<br>{_text(second)}</p>"
        "<p class='meta'>Each turn is replayed with its original preceding history "
        "and the current code, prompts and model. Tools that ran use current "
        "data; later messages of the conversation are not replayed.</p>"
        + "".join(_html_entry(entry_view(e)) for e in artifact.get("entries") or [])
        + "</body></html>\n"
    )
