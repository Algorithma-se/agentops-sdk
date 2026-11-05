"""``agentops evolve replay``: run frozen turns with the agent's current code.

The agent runs through its repo's ``app.evolve_agent`` (foundry-agents-core):
``runtime(prompts)`` yields a ``Runtime`` whose ``run_turn`` replays one turn.
Each entry runs as the root of the trace ``run_id``; its trace events are
captured into ``run.events`` (``capture``), never sent. With ``repeat`` N,
each entry runs N - 1 more times untraced into ``run.repeats`` (status and
reply only). The frozen file is kept; the replay is written to a new file.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional, Union

from .._env import get_host, get_public_key, get_secret_key
from . import artifact as files
from . import capture, local_mcp, repo
from .api import EvolveApi, parse_env_file
from .child.sitecustomize import ENV_FILE_VAR, patch_settings
from .media import Media
from .prompts import LocalCandidate, read_candidate, run_prompt

CHILD_DIR = Path(__file__).parent / "child"

_log = logging.getLogger("agentops")

MAX_REPEAT = 10
WARNING = (
    "Warning: the replay runs the agent's hooks (after_session included) and "
    "its tools as normal, with the credentials in {path}. That file must only "
    "hold read-only or sandbox credentials; never production write credentials."
)
NO_EVENTS = (
    "no trace events were captured; set AGENTOPS_HOST, AGENTOPS_PUBLIC_KEY "
    "and AGENTOPS_SECRET_KEY in the env file (nothing is sent)"
)


class ReplayRefused(RuntimeError):
    """The replay cannot start; nothing has been run or written."""


def load_env_file(path: Path) -> dict[str, str]:
    """Load *path* into ``os.environ``; refuse unless it sets
    ``LANGFUSE_SAMPLE_RATE=0``. Pydantic settings then read *path* instead of
    ``./.env``, and the tracing client gets the file's AgentOps keys."""
    path = Path(path)
    if not path.is_file():
        raise ReplayRefused(f"--env-file {path} does not exist")
    values = parse_env_file(path)
    try:
        zero = float(values.get("LANGFUSE_SAMPLE_RATE", "")) == 0
    except ValueError:
        zero = False
    if not zero:
        raise ReplayRefused(f"{path} must set LANGFUSE_SAMPLE_RATE=0 so the replay sends no traces")
    os.environ.update(values)
    # Never secrets from a vault: with READ_SECRETS=true (from the shell, or
    # `uv run --env-file`), core fills every unset setting from Key Vault
    # with the developer's login, which can be production's.
    os.environ["READ_SECRETS"] = "false"
    for key, get in (
        ("LANGFUSE_HOST", get_host),
        ("LANGFUSE_PUBLIC_KEY", get_public_key),
        ("LANGFUSE_SECRET_KEY", get_secret_key),
    ):
        if key not in values and get():
            os.environ[key] = str(get())
    # Child processes (core's tool server, --local-mcp servers) read the same
    # file instead of their working directory's .env: see child/sitecustomize.
    os.environ[ENV_FILE_VAR] = str(path.resolve())
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(CHILD_DIR), os.environ.get("PYTHONPATH")) if p
    )
    patch_settings(str(path.resolve()))
    return values


def check(entries: list[dict], runtime: Any, candidates: dict[str, LocalCandidate]) -> None:
    """Refuse before anything runs: every agent must be one of the runtime's
    and every candidate must replace the AOS prompt of an agent the replay
    starts (a delegate's or handover agent's prompt is not recorded)."""
    unknown = sorted({e["agent_name"] for e in entries} - set(runtime.agents))
    if unknown:
        raise ReplayRefused(
            "these agents are not in this repository's agents config: "
            + ", ".join(unknown)
            + ". Their feedback belongs to another repository's agents; export "
            "only this repository's agents"
        )
    used = {runtime.agents[e["agent_name"]].aos_prompt for e in entries}
    unused = sorted(set(candidates) - used)
    if unused:
        raise ReplayRefused(
            "; ".join(
                f"--prompt {name}: no replayed agent uses AOS prompt '{name}' "
                "(a delegate's or handover agent's prompt cannot be replayed yet)"
                for name in unused
            )
        )


def redact(text: str) -> str:
    """*text* without URL credentials (``scheme://user:pass@``) and URL query
    strings, which can hold keys: an error is published with the replay."""
    text = re.sub(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@'\"]+@", r"\1<redacted>@", text)
    return re.sub(r"(https?://[^\s?#'\"]+)\?[^\s'\"]*", r"\1?<redacted>", text)


def request_meta(entry: dict) -> Optional[dict]:
    """The recorded ``request_meta``, plus ``{"agent_variant": v}`` for a
    trace named ``<agent_name>/<v>``; None when there is neither."""
    meta = dict(entry["source"].get("request_meta") or {})
    trace_name = entry.get("trace_name") or ""
    prefix = f"{entry['agent_name']}/"
    if trace_name.startswith(prefix) and len(trace_name) > len(prefix):
        meta["agent_variant"] = trace_name[len(prefix) :]
    return meta or None


def new_run() -> dict:
    """An entry's ``run`` before the agent starts (a failure keeps this shape)."""
    return {
        "status": "failed",
        "reply": None,
        "trace": [],
        "system_prompt": None,
        "prompt": None,
        "tool_calls": [],
        "events": [],
        "error": None,
        "started_at": files.now_iso(),
        "completed_at": None,
    }


async def run_entry(
    runtime: Any,
    entry: dict,
    candidates: dict[str, LocalCandidate],
    sink: Optional[capture.TraceCapture] = None,
    media: Optional[Media] = None,
) -> tuple[dict, str]:
    """Replay one entry; return its run record and the model used. With
    *sink*, it runs as the root of the trace ``run_id`` and keeps its events.
    *media* has the images the source refers to (the agent gets them; the
    record keeps their references)."""
    media = media or Media()
    source = media.inline(entry["source"])
    run = new_run()
    result: Any = None

    async def play() -> None:
        nonlocal result
        result = await runtime.run_turn(
            entry["agent_name"],
            source["user_message"],
            copy.deepcopy(source["history_before"]),
            request_meta(entry),
            copy.deepcopy(source.get("user_context")),
            source.get("attachments") or None,
        )

    try:
        if sink is None:
            await play()
        else:
            name = entry.get("trace_name") or entry["agent_name"]
            await capture.traced(sink, entry["run_id"], name, play)
        messages = media.restore(result.messages)
        run.update(
            status="complete",
            reply=result.reply,
            trace=[m for m in messages if m.get("role") != "system"],
            system_prompt=next(
                (m.get("content") for m in reversed(messages) if m.get("role") == "system"),
                None,
            ),
            prompt=run_prompt(result.prompt, candidates),
            tool_calls=result.tool_calls,
        )
        aos_prompt = runtime.agents[entry["agent_name"]].aos_prompt
        if aos_prompt in candidates and not (run["prompt"] or {}).get("candidate"):
            # Not an improvement to show: the old prompt answered.
            run.update(
                status="failed",
                error=f"the agent did not use the candidate prompt '{aos_prompt}'",
            )
    except Exception as exc:
        _log.exception("Replay of trace %s failed", entry.get("source_trace_id"))
        run["error"] = redact(f"{type(exc).__name__}: {exc}")
    if sink is not None:
        run["events"] = media.restore(sink.events(entry["run_id"]))
        if not run["events"] and run["status"] == "complete":
            run.update(status="failed", error=NO_EVENTS)
    run["completed_at"] = files.now_iso()
    return run, (result.model if result is not None else "")


async def _run_all(
    core: Any,
    entries: list[dict],
    candidates: dict[str, LocalCandidate],
    repeat: int,
    concurrency: int,
    media: Media,
) -> dict:
    limit = asyncio.Semaphore(concurrency)
    with capture.capture() as sink:
        async with core.runtime({n: c.text for n, c in candidates.items()}) as runtime:
            check(entries, runtime, candidates)

            async def one(entry: dict, traced: bool) -> tuple[dict, str]:
                async with limit:
                    sunk = sink if traced else None
                    return await run_entry(runtime, entry, candidates, sunk, media)

            # Each run takes its own slot; gather keeps the repeats in order.
            done = await asyncio.gather(
                *(one(e, True) for e in entries),
                *(one(e, False) for e in entries for _ in range(repeat - 1)),
            )
            provider = str(runtime.provider)
    if sink.unrouted:
        _log.warning(
            "%d trace event(s) belonged to no replayed turn and were dropped",
            sink.unrouted,
        )
    for i, entry in enumerate(entries):
        entry["run"] = done[i][0]
        entry["resolutions"] = []
        if repeat > 1:
            start = len(entries) + i * (repeat - 1)
            entry["run"]["repeats"] = [
                {key: run[key] for key in files.REPEAT_KEYS}
                for run, _ in done[start : start + repeat - 1]
            ]
    models = sorted({model for _, model in done[: len(entries)] if model})
    return {"model": ", ".join(models), "provider": provider}


def replay(
    path: Union[str, Path],
    *,
    env_file: Union[str, Path],
    name: Optional[str] = None,
    output: Optional[Path] = None,
    repeat: int = 1,
    concurrency: int = 4,
    prompt_files: Optional[dict[str, Path]] = None,
    local_servers: Optional[dict[str, str]] = None,
    module: str = "app.evolve_agent",
) -> dict:
    """Replay the frozen file *path* into *output* (default
    ``local/evolve/<date>_<NN>-<name|sha7>/replay.json``) with its HTML page.

    Loads *env_file* first and downloads the images the turns refer to
    (``media``). *local_servers* (name -> command) are tool servers started
    from this checkout for the run (``local_mcp``). Raises ``ReplayRefused``,
    ``PromptError``, ``EvolveApiError``, ``LocalMcpError``, ``ValueError`` or
    ``OSError`` before anything runs or is written. Returns
    ``{path, runs, failed, model, provider, dirty}``.
    """
    if not 1 <= repeat <= MAX_REPEAT:
        raise ValueError(f"--repeat must be between 1 and {MAX_REPEAT}")
    load_env_file(Path(env_file))
    print(WARNING.format(path=env_file), file=sys.stderr)
    artifact = files.read(Path(path))
    entries = artifact["entries"]
    candidates = {n: read_candidate(n, Path(p)) for n, p in (prompt_files or {}).items()}
    provenance = repo.snapshot()
    if output:
        output = Path(output)
        label = output.parent.name if output.name == "replay.json" else output.stem
    else:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip("-")
        output = files.default_replay_path(slug or provenance["git_commit"][:7])
        label = output.parent.name
    files.ensure_new(output)

    artifact.update(
        replay_id=uuid.uuid4().hex,
        name=name or artifact.get("name") or label,
        created_at=files.now_iso(),
    )
    for entry in entries:
        entry["run_id"] = uuid.uuid4().hex
    try:
        found = module in sys.modules or importlib.util.find_spec(module) is not None
    except (ModuleNotFoundError, ValueError):
        found = False
    if not found:
        raise ReplayRefused(
            f"{module} is not in this repository: its core/ is older than "
            "foundry-agents-core #361 (Evolve Agent replays); update core/ first"
        )
    media = Media()
    media.load(entries, lambda media_id: EvolveApi().media(media_id))
    if local_servers:  # their logs go next to the replay
        output.parent.mkdir(parents=True, exist_ok=True)
    with local_mcp.started(local_servers or {}, output.parent):
        servers = local_mcp.tool_servers(local_servers or {})
        # Imported here, so the agent's settings see the local servers' URLs.
        core = importlib.import_module(module)
        runtime = asyncio.run(_run_all(core, entries, candidates, repeat, max(1, concurrency), media))
    artifact["provenance"] = {
        **provenance,
        **runtime,
        **({"tool_servers": servers} if servers else {}),
    }
    files.save(output, artifact)
    return {
        "path": output,
        "runs": len(entries),
        "failed": sum(entry["run"]["status"] != "complete" for entry in entries),
        **runtime,
        "dirty": provenance["dirty"],
    }
