"""``agentops evolve`` — the developer side of Evolve Agent.

Commands: find, export, prompt pull, replay, assess, show, render, ask,
publish, revert, install-skill. Only ``ask`` and ``publish`` / ``revert``
with ``--apply`` write to AOS Control (``publish`` also creates a prompt
version per candidate prompt the replay used); the rest read from it or
write local files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import artifact as files
from . import publish as pub
from . import repo, skill, versions, view
from .api import EvolveApi, EvolveApiError, EvolveNotConfiguredError, load_env_file
from .prompts import DEFAULT_LABEL, PROMPT_DIR, PromptError

PROG = "agentops evolve"

# As the TypeScript EVOLVE_ASSESSMENT_STATUSES / EVOLVE_FEEDBACK_STATES.
STATUSES = (
    "fixed",
    "verified",
    "partial",
    "unresolved",
    "regressed",
    "blocked",
    "superseded",
    "wontfix",
)
RESOLVED = ("fixed", "verified", "superseded", "wontfix")
FEEDBACK_STATES = (
    "reply",
    "open",
    "needs_input",
    "partial",
    "fixed",
    "closed",
    "dismissed",
    "hidden",
)
# The frozen feedback object of a replay file.
FROZEN_KEYS = (
    "feedback_id", "revision", "previous_revision", "session_id", "trace_id", "replay_run_id",
    "trace_name", "scope", "source", "note", "author_name", "created_at",
)  # fmt: skip


class Fail(Exception):
    """A user-facing failure: printed without a traceback (exit 2)."""


@dataclass
class Result:
    """What a command prints: *lines* for people, *data* as JSON on stdout."""

    lines: list[str] = field(default_factory=list)
    data: Any = None
    code: int = 0


# --- Selecting turns (find, export)


def default_agents() -> list[str]:
    """The agents of this repo's ``config/agents.yaml`` (or
    ``AGENTS_CONFIG_FILE``); none outside an agent repo."""
    path = Path("config") / os.environ.get("AGENTS_CONFIG_FILE", "agents.yaml")
    if not path.is_file():
        return []
    try:
        import yaml
    except ImportError:
        raise Fail(f"PyYAML is needed to read {path}; or pass --trace-name") from None
    return list((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("agents") or {})


_DURATION = re.compile(r"(\d+)([dhm])")


def parse_time(value: str, flag: str) -> str:
    """``7d`` / ``24h`` / ``30m`` before now as ISO 8601 UTC, or an ISO time as given."""
    match = _DURATION.fullmatch(value.strip())
    if match:
        unit = {"d": "days", "h": "hours", "m": "minutes"}[match[2]]
        moment = datetime.now(timezone.utc) - timedelta(**{unit: int(match[1])})
        return moment.isoformat().replace("+00:00", "Z")
    _moment(value, flag)
    return value


def _moment(value: str, flag: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise Fail(
            f"{flag} {value!r} is neither a duration (7d, 24h, 30m) nor an ISO 8601 time"
        ) from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _csv(values: Optional[list[str]]) -> Optional[str]:
    return ",".join(dict.fromkeys(v for v in values or [] if v)) or None


def turn_filters(args: argparse.Namespace) -> dict:
    """The ``GET /evolve/turns`` query for the selection flags. ``--feedback
    unresolved`` asks for open, partial, fixed and closed: the local rule
    (``still_work``) drops what a commit in HEAD resolved."""
    if args.any_tools and not args.like:
        raise Fail("--any-tools only applies to --like")
    feedback = args.feedback
    if feedback == "unresolved":
        feedback = "open,partial,fixed,closed"
    elif feedback and feedback not in ("any", "none"):
        unknown = set(feedback.split(",")) - set(FEEDBACK_STATES)
        if unknown:
            raise Fail(f"--feedback: unknown state(s) {', '.join(sorted(unknown))}")
    if feedback == "none" and args.vote:
        raise Fail("--feedback none (no note or vote) excludes --vote")
    since = parse_time(args.since, "--since") if args.since else None
    until = parse_time(args.until, "--until") if args.until else None
    if since and until and _moment(since, "--since") > _moment(until, "--until"):
        raise Fail("--since is later than --until")
    # Named turns are never narrowed to this repo's agents.
    named = args.like or args.trace_id or args.feedback_id
    return {
        "traceName": _csv(args.trace_name or ([] if named else default_agents())),
        "sessionId": _csv(args.session),
        "traceId": _csv(args.trace_id),
        "fromTimestamp": since,
        "toTimestamp": until,
        "text": args.text,
        "tool": _csv(args.tool),
        "toolError": "true" if args.tool_error else None,
        "vote": args.vote,
        "feedback": feedback,
        "feedbackId": _csv(args.feedback_id),
        "source": args.source,
        "like": args.like,
        "sameTools": ("false" if args.any_tools else "true") if args.like else None,
    }


def is_resolved(item: dict) -> bool:
    """The current revision's latest assessment resolves it, from a commit
    the local HEAD contains."""
    mine = [
        a
        for a in item.get("assessments") or []
        if a.get("feedback_revision") == item.get("revision")
    ]
    if not mine:
        return False
    latest = max(mine, key=lambda a: a.get("created_at") or "")
    return latest.get("status") in RESOLVED and repo.is_ancestor(latest.get("git_commit") or "")


def still_work(item: dict) -> bool:
    """Open or partly fixed, or fixed/closed by a commit HEAD does not contain."""
    state = item.get("state")
    return state in ("open", "partial") or (state in ("fixed", "closed") and not is_resolved(item))


def excluded(args: argparse.Namespace) -> Callable[[dict], bool]:
    """True for a turn of a session in an ``--exclude`` file: a frozen or
    replayed file (``entries``) or saved ``find`` JSON (``data``)."""
    sessions = set()
    for path in args.exclude:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        sessions |= {t.get("session_id") for t in body.get("entries") or body.get("data") or []}
    sessions.discard(None)
    return lambda turn: turn.get("session_id") in sessions


def counted(turns: list[dict]) -> str:
    items = sum(len(t.get("feedback") or []) for t in turns)
    return f"{len(turns)} turn(s), {items} feedback item(s)"


def select(api: EvolveApi, args: argparse.Namespace, include: str) -> list[dict]:
    """Every turn the flags pick, with *include*, minus ``--exclude`` sessions;
    ``--feedback unresolved`` keeps the turns (and feedback) that are still
    work on this branch."""
    out = excluded(args)
    turns = [t for t in api.all_turns({**turn_filters(args), "include": include}) if not out(t)]
    if args.feedback != "unresolved":
        return turns
    for turn in turns:
        turn["feedback"] = [f for f in turn.get("feedback") or [] if still_work(f)]
    return [t for t in turns if t["feedback"]]


def cmd_find(args: argparse.Namespace) -> Result:
    api = EvolveApi()
    if args.feedback == "unresolved":
        turns = select(api, args, "feedback")
        return Result(data={"data": turns}, lines=[counted(turns) if turns else "No turns match."])
    body = api.turns(
        {**turn_filters(args), "include": "feedback"}, page=args.page, limit=args.limit
    )
    out = excluded(args)
    shown = len(body.get("data") or [])
    body["data"] = [t for t in body.get("data") or [] if not out(t)]
    if body["data"]:
        more = f"; more: --page {args.page + 1}" if shown >= args.limit else ""
        return Result(data=body, lines=[f"{counted(body['data'])} on page {args.page}{more}"])
    hint = (
        ""
        if not args.like or args.any_tools
        else " No turn used the same tools in order; try --any-tools."
    )
    return Result(data=body, lines=[f"No turns match.{hint}"])


def cmd_export(args: argparse.Namespace) -> Result:
    if not (args.trace_id or args.feedback or args.feedback_id):
        raise Fail("pick turns with --feedback, --feedback-id or --trace-id")
    output = Path(args.output)
    files.ensure_new(output)
    api = EvolveApi()
    picked = [t["trace_id"] for t in select(api, args, "feedback")]
    missing = [t for t in dict.fromkeys(args.trace_id) if t not in picked]
    if missing:
        raise Fail(
            f"--trace-id {', '.join(missing)}: not found, not matching the other flags, "
            "in an --exclude file's sessions, or another repository's agent"
        )
    if not picked:
        return Result(["No turns match; nothing exported."])
    # Every feedback item of the picked turns is frozen (a fixed one can regress).
    turns = []
    for start in range(0, len(picked), 100):
        turns += api.all_turns(
            {"traceId": ",".join(picked[start : start + 100]), "include": "feedback,context"}
        )
    entries = []
    for turn in turns:
        if not turn.get("context") or not turn.get("trace_name"):
            raise Fail(f"turn {turn['trace_id']} has no readable model call or agent name")
        entries.append(
            {
                "run_id": None,
                "session_id": turn.get("session_id"),
                "source_trace_id": turn["trace_id"],
                "trace_name": turn["trace_name"],
                "agent_name": turn["trace_name"].split("/", 1)[0],
                "source": turn["context"],
                "feedback": [
                    {k: f.get(k) for k in FROZEN_KEYS} for f in turn.get("feedback") or []
                ],
                "run": None,
                "resolutions": [],
            }
        )
    files.save(
        output,
        {
            "schema_version": files.SCHEMA_VERSION,
            "replay_id": None,
            "name": None,
            "created_at": files.now_iso(),
            "agentops_host": api.host,
            "project_id": api.project_id(),
            "provenance": None,
            "entries": entries,
        },
    )
    count = sum(len(e["feedback"]) for e in entries)
    return Result(
        [
            f"Frozen {len(entries)} turn(s) with {count} feedback revision(s): {output}",
            f"Next: {PROG} replay {output} --env-file .env.evolve --name <name>",
        ]
    )


# --- Local files (prompt pull, replay, assess, show, render)


def cmd_prompt_pull(args: argparse.Namespace) -> Result:
    meta = versions.pull(
        versions.PromptStore(), args.name, label=args.label, output_dir=Path(args.output_dir)
    )
    return Result(
        [
            f"Pulled {args.name} version {meta['base_version']} (label {args.label}): "
            f"{meta['path']}",
            f"Edit it, then: {PROG} replay <frozen.json> --env-file .env.evolve "
            f"--prompt {args.name}={meta['path']}",
        ]
    )


def cmd_replay(args: argparse.Namespace) -> Result:
    from .replay import replay

    prompt_files = {}
    for spec in args.prompt:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise Fail(f"--prompt {spec!r}: expected NAME=PATH")
        if name in prompt_files:
            raise Fail(f"--prompt {name} is given more than once")
        prompt_files[name] = Path(path)
    local_servers = {}
    for spec in args.local_mcp:
        name, sep, command = spec.partition("=")
        if not sep or not name or not command.strip():
            raise Fail(f"--local-mcp {spec!r}: expected NAME=COMMAND")
        if name.lower() in {n.lower() for n in local_servers}:
            raise Fail(f"--local-mcp {name} is given more than once")
        local_servers[name] = command
    done = replay(
        args.file,
        env_file=args.env_file,
        name=args.name,
        output=Path(args.output) if args.output else None,
        repeat=args.repeat,
        concurrency=args.concurrency,
        prompt_files=prompt_files,
        local_servers=local_servers,
    )
    lines = [
        f"Replayed {done['runs']} turn(s), {done['failed']} failed · "
        f"{done['provider']} {done['model']}",
        f"Wrote {done['path']} and its .html comparison",
    ]
    entries = files.read(Path(done["path"]))["entries"]
    lines += [
        f"Failed: trace {e['source_trace_id']}: {e['run']['error']}"
        for e in entries
        if e["run"]["status"] != "complete"
    ]
    if args.repeat > 1:
        repeats = [(e, r) for e in entries for r in e["run"].get("repeats") or []]
        failed = [(e, r) for e, r in repeats if r["status"] != "complete"]
        lines.append(
            f"{args.repeat} runs per turn; {len(repeats) - len(failed)} of {len(repeats)} "
            "repeats complete"
        )
        lines += [f"Failed repeat: trace {e['source_trace_id']}: {r['error']}" for e, r in failed]
    if done["dirty"]:
        lines.append("The tree was dirty: commit, then replay again before publishing.")
    return Result(lines, code=1 if done["failed"] else 0)


def find_entry(artifact: dict, args: argparse.Namespace) -> dict:
    """The one entry ``--run-id``, ``--session`` or ``--trace-id`` picks."""
    flag, key, value = next(
        (f, k, v)
        for f, k, v in (
            ("--run-id", "run_id", args.run_id),
            ("--session", "session_id", args.session),
            ("--trace-id", "source_trace_id", args.trace_id),
        )
        if v
    )
    found = [e for e in artifact.get("entries") or [] if e.get(key) == value]
    if len(found) != 1:
        runs = ", ".join(str(e.get("run_id")) for e in found)
        raise Fail(
            f"no entry with {flag} {value}"
            if not found
            else f"{flag} {value} matches runs {runs}; use --run-id"
        )
    return found[0]


def cmd_assess(args: argparse.Namespace) -> Result:
    """Record (or replace) the assessment of one frozen feedback revision;
    ``changes`` starts with the replay commit, which publish looks for."""
    path = Path(args.artifact)
    artifact = files.read(path)
    entry = find_entry(artifact, args)
    commit = (artifact.get("provenance") or {}).get("git_commit")
    if (entry.get("run") or {}).get("status") != "complete" or not commit:
        raise Fail("a completed replay with a git_commit is required")
    item = next(
        (f for f in entry.get("feedback") or [] if f["feedback_id"] == args.feedback_id), None
    )
    if item is None:
        raise Fail(f"feedback {args.feedback_id} is not frozen in this entry")
    entry["resolutions"] = [
        r for r in entry.get("resolutions") or [] if r["feedback_id"] != args.feedback_id
    ] + [
        {
            "feedback_id": item["feedback_id"],
            "feedback_revision": item["revision"],
            "status": args.status,
            "explanation": args.explanation,
            "evidence": args.evidence,
            "changes": f"{commit[:12]}: {args.changes}",
            "author": args.author,
            "created_at": files.now_iso(),
        }
    ]
    files.save(path, artifact)
    return Result([f"Assessment saved: {path} ({path.with_suffix('.html')})"])


def cmd_show(args: argparse.Namespace) -> Result:
    return Result(view.summary(files.read(Path(args.artifact)), full=args.full))


def cmd_render(args: argparse.Namespace) -> Result:
    path = Path(args.artifact)
    return Result([f"Comparison: {files.write_html(path, files.read(path))}"])


# --- AOS Control writes (ask, publish, revert)


def cmd_ask(args: argparse.Namespace) -> Result:
    api = EvolveApi()
    if args.withdraw:
        return Result(["Withdrawn. The feedback is work again."], api.withdraw(args.withdraw))
    if not (args.feedback_id and args.question and args.author):
        raise Fail("ask needs --feedback-id, --question and --author (or --withdraw <id>)")
    result = api.ask(
        feedback_id=args.feedback_id,
        revision=args.revision,
        question=args.question,
        author=args.author,
    )
    note = (
        "Already asked; the open question is unchanged."
        if result.get("duplicate")
        else ("Asked. The feedback waits (needs_input) until someone answers.")
    )
    return Result([note], result)


def _published_note(existing: dict) -> str:
    created, updated = str(existing.get("created_at"))[:10], str(existing.get("updated_at"))[:10]
    if existing.get("status") == "published":
        return (
            f"Already published on {created}; --apply changes nothing, a changed file is refused."
        )
    return f"Published on {created}, reverted on {updated}; --apply publishes it again."


def cmd_publish(args: argparse.Namespace) -> Result:
    path = Path(args.artifact)
    artifact = files.read(path)
    lines = view.summary(artifact)
    api = EvolveApi()
    replay_id = artifact.get("replay_id")
    existing = api.publication(replay_id) if replay_id else None
    found = pub.problems(
        artifact, head=repo.head(), repo_dirty=repo.is_dirty(), project_id=api.project_id()
    )
    publisher = repo.user_name()
    if publisher:
        lines.append(f"\nPublished by: {publisher} (git config user.name)")
    else:
        found.append("git config user.name is not set: reviewers need to know whom to ask")
    # A replay published before already has its prompt versions.
    store = versions.PromptStore() if existing is None and versions.candidates(artifact) else None
    if existing is not None:
        lines.append("\n" + _published_note(existing))
    if store is not None:
        found += versions.candidate_problems(artifact, store, prompt_dir=Path(args.prompts_dir))
        lines += [
            f"\nPrompt {c['name']}: --apply creates a version from the candidate "
            f"(base v{c['base_version']}), "
            "with no label. A person moves a serving label to it in AOS Control."
            for c in versions.candidates(artifact)
        ]
    if found:
        return Result([*lines, "\nCannot publish:", *(f"- {p}" for p in found)], code=1)
    if not args.apply:
        return Result(
            [*lines, f"\nPreview only; nothing was sent. Apply: {PROG} publish {path} --apply"]
        )
    prompt_versions, reused = (
        versions.push_candidates(artifact, store, prompt_dir=Path(args.prompts_dir))
        if store
        else ([], set())
    )
    lines += [
        f"Prompt {v['name']}: version {v['version']}" + (" reused" if v["name"] in reused else "")
        for v in prompt_versions
    ]
    result = api.publish(artifact, prompt_versions=prompt_versions, published_by=publisher)
    lines.append(
        "\nAlready published (same digest)." if result.get("duplicate") else "\nPublished."
    )
    lines.append(
        "Replay traces are queued; the review page shows them within minutes. "
        "If a run still says its trace is missing, publish the same artifact again."
    )
    return Result(lines, result)


def cmd_revert(args: argparse.Namespace) -> Result:
    target = Path(args.replay)
    replay_id = files.read(target).get("replay_id") if target.suffix == ".json" else args.replay
    if not replay_id:
        raise Fail(f"{target} was never replayed (no replay_id)")
    api = EvolveApi()
    existing = api.publication(replay_id)
    if existing is None or existing.get("status") != "published":
        return Result([f"Replay {replay_id} is not published; nothing to revert."])
    if not args.apply:
        return Result(
            [f"Replay {existing['name']} is published since {existing['created_at'][:10]}.",
             f"Preview only. Withdraw it: {PROG} revert {args.replay} --apply"]
        )  # fmt: skip
    return Result(["Reverted. Reviewers no longer see this replay."], api.revert(replay_id))


def cmd_install_skill(args: argparse.Namespace) -> Result:
    lines = []
    for item in skill.install(
        [Path(t) for t in args.target] or list(skill.DEFAULT_TARGETS), copy=args.copy
    ):
        if item.linked:
            lines.append(
                f"Linked: {item.path} -> {skill.SKILL_SOURCE}"
                + (" (git-ignored locally)" if item.excluded else "")
            )
        else:
            lines.append(f"Copied: {item.path} (run again after upgrading the SDK)")
        if item.was_tracked:
            lines.append(
                f"  {item.path} was committed before: git rm -r --cached {item.path} && git commit"
            )
    return Result(lines)


# --- Arguments


def _selection(p: argparse.ArgumentParser) -> None:
    add = p.add_argument
    add(
        "--trace-name",
        action="append",
        default=[],
        help="agent (also its variants agent/<v>); repeatable; default: config/agents.yaml",
    )
    add("--session", action="append", default=[], help="session id; repeatable")
    add("--trace-id", action="append", default=[], help="a turn; repeatable")
    add("--since", help="7d, 24h, 30m or an ISO 8601 time (server default: 7d)")
    add("--until", help="7d, 24h, 30m or an ISO 8601 time")
    add("--text", help="case-insensitive part of the user message")
    add(
        "--tool", action="append", default=[], help="a tool the turn used; repeatable (all of them)"
    )
    add("--tool-error", action="store_true", help="only turns with a failed tool call")
    add("--vote", choices=("up", "down"))
    add(
        "--feedback",
        help="unresolved (still work on this branch), any, none, or states: "
        + ",".join(FEEDBACK_STATES),
    )
    add("--feedback-id", action="append", default=[], help="repeatable")
    add("--source", help="who wrote the feedback (e.g. user)")
    add("--like", metavar="TRACE_ID", help="turns like this one, most similar first")
    add("--any-tools", action="store_true", help="with --like: do not require the same tools")
    add(
        "--exclude",
        action="append",
        default=[],
        metavar="JSON",
        help="skip the sessions in this frozen file or saved find output; repeatable",
    )


def _env_file(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--env-file",
        metavar="PATH",
        help="read the AGENTOPS_* keys from this dotenv file (only those)",
    )


def _assess_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("artifact")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--run-id")
    which.add_argument("--session")
    which.add_argument("--trace-id", help="the entry's source trace")
    p.add_argument("--feedback-id", required=True)
    p.add_argument("--status", required=True, choices=STATUSES)
    for name in ("--explanation", "--evidence", "--changes", "--author"):
        p.add_argument(name, required=True)


def _replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="a frozen export")
    p.add_argument(
        "--env-file", required=True, help="the agent's env file (needs LANGFUSE_SAMPLE_RATE=0)"
    )
    p.add_argument("--name", help="the replay's name (default: the output folder)")
    p.add_argument("--output", help="new replay.json (default: local/evolve/<date>_<NN>-<name>/)")
    p.add_argument("--repeat", type=int, default=1, help="runs per turn, 1-10")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--prompt",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="replay a candidate prompt",
    )
    p.add_argument(
        "--local-mcp",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="start the MCP server NAME from this checkout for the run ({port}: a free port)",
    )


# (name, help, run, arguments, reads AGENTOPS_* from --env-file)
COMMANDS: list[tuple[str, str, Callable, Callable, bool]] = [
    ("find", "turns with their feedback, as JSON", cmd_find,
     lambda p: (_selection(p), p.add_argument("--limit", type=int, default=50),
                p.add_argument("--page", type=int, default=1)), True),
    ("export", "freeze turns and their feedback for a replay", cmd_export,
     lambda p: (_selection(p), p.add_argument("--output", required=True)), True),
    ("prompt-pull", "write an AOS prompt to a local candidate file", cmd_prompt_pull,
     lambda p: (p.add_argument("name"), p.add_argument("--label", default=DEFAULT_LABEL),
                p.add_argument("--output-dir", default=str(PROMPT_DIR))), True),
    ("replay", "replay a frozen file with this checkout's agent", cmd_replay, _replay_args, False),
    ("assess", "record an assessment of one feedback revision", cmd_assess, _assess_args, False),
    ("show", "a replay as text", cmd_show,
     lambda p: (p.add_argument("artifact"), p.add_argument("--full", action="store_true")), False),
    ("render", "(re)write the HTML comparison", cmd_render,
     lambda p: p.add_argument("artifact"), False),
    ("ask", "ask the reviewers about a feedback item", cmd_ask,
     lambda p: (p.add_argument("--feedback-id"), p.add_argument("--question"),
                p.add_argument("--author"),
                p.add_argument("--revision", help="default: the current one"),
                p.add_argument("--withdraw", metavar="QUESTION_ID")), True),
    ("publish", "check, and with --apply publish, a replay", cmd_publish,
     lambda p: (p.add_argument("artifact"), p.add_argument("--apply", action="store_true"),
                p.add_argument("--prompts-dir", default=str(PROMPT_DIR))), True),
    ("revert", "withdraw a published replay (file or replay id)", cmd_revert,
     lambda p: (p.add_argument("replay"), p.add_argument("--apply", action="store_true")), True),
    ("install-skill", "install the evolve-agent skill here", cmd_install_skill,
     lambda p: (p.add_argument("--target", action="append", default=[]),
                p.add_argument("--copy", action="store_true")), False),
]  # fmt: skip

DESCRIPTION = (
    "Evolve Agent: find reviewed turns, freeze and replay them, assess and "
    "publish the replay. AGENTOPS_HOST, AGENTOPS_PUBLIC_KEY and "
    "AGENTOPS_SECRET_KEY come from the environment or --env-file; .env is never read."
)


def add_subcommands(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="evolve_command", metavar="<command>", required=True)
    for name, help_text, run_command, arguments, reads_env in COMMANDS:
        if name == "prompt-pull":
            prompt = sub.add_parser("prompt", help="candidate AOS prompts").add_subparsers(
                dest="prompt_command", metavar="<command>", required=True
            )
            p = prompt.add_parser("pull", help=help_text)
        else:
            p = sub.add_parser(name, help=help_text)
        arguments(p)
        if reads_env:
            _env_file(p)
        p.set_defaults(run=run_command, reads_env=reads_env)


def run(args: argparse.Namespace) -> int:
    """Run a parsed command; print its result; failures become exit codes."""
    from .local_mcp import LocalMcpError
    from .replay import ReplayRefused

    try:
        if args.reads_env and args.env_file:
            load_env_file(Path(args.env_file))
        result = args.run(args)
    except EvolveApiError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Error: git failed: {(exc.stderr or '').strip()}", file=sys.stderr)
        return 2
    except (
        Fail,
        OSError,
        ValueError,
        EvolveNotConfiguredError,
        PromptError,
        ReplayRefused,
        LocalMcpError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if result.lines:
        print("\n".join(result.lines), file=sys.stderr if result.data is not None else sys.stdout)
    if result.data is not None:
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
    return result.code


def main(argv: Optional[list[str]] = None) -> int:
    """``agentops evolve`` on its own (*argv* without the ``evolve`` word)."""
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    add_subcommands(parser)
    return run(parser.parse_args(argv))
