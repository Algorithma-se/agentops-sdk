"""``agentops evolve replay`` against a fake ``app.evolve_agent`` (no agent, no
network): the Langfuse v2 client is real, but its ``add_task`` is patched by
the capture, so no event is ever queued or sent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from agentops.evolve import artifact as files
from agentops.evolve import capture, repo
from agentops.evolve.prompts import PromptError
from agentops.evolve.replay import NO_EVENTS, ReplayRefused, replay

SNAPSHOT = {"git_commit": "a" * 40, "dirty": False, "diff": "", "untracked_hashes": {}}
RUN_KEYS = {
    "status",
    "reply",
    "trace",
    "system_prompt",
    "prompt",
    "tool_calls",
    "events",
    "error",
    "started_at",
    "completed_at",
}


@dataclass
class AgentInfo:
    aos_prompt: Optional[str]
    has_tools: bool = False


@dataclass
class TurnResult:
    reply: Optional[str]
    prompt: Optional[dict]
    tool_calls: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    model: str = "gpt-test"


class Runtime:
    def __init__(self, prompts: dict, fail: set):
        self.agents = {"support": AgentInfo("support-system"), "sales": AgentInfo(None)}
        self.provider = "azure"
        self.prompts = prompts
        self.fail = fail
        self.calls: list = []
        self.attachments: list = []

    async def run_turn(
        self, agent_name, user_message, history, request_meta, user_context, attachments=None
    ):
        from langfuse.decorators import observe

        self.calls.append((agent_name, user_message, history, request_meta, user_context))
        self.attachments.append(attachments)

        @observe(name="agent-step")
        async def step(text, images):
            if agent_name in self.fail:
                raise RuntimeError("model down")
            return f"re: {text}"

        reply = await step(user_message, attachments)
        candidate = "support-system" in self.prompts and agent_name == "support"
        system = self.prompts.get("support-system", "You help.")
        return TurnResult(
            reply=reply,
            prompt={"name": "support-system", "version": None if candidate else 7}
            if agent_name == "support"
            else None,
            tool_calls=[{"name": "lookup", "arguments": "{}"}],
            messages=[
                {"role": "system", "content": system},
                *history,
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": reply},
            ],
        )


@pytest.fixture
def core(monkeypatch, tmp_path):
    """A fake ``app.evolve_agent`` under a test-only module name."""
    module = types.ModuleType("fake_core_evolve")
    module.fail = set()
    module.runtimes = []

    @asynccontextmanager
    async def runtime(prompts):
        rt = Runtime(prompts, module.fail)
        module.runtimes.append(rt)
        yield rt

    module.runtime = runtime
    monkeypatch.setitem(sys.modules, "fake_core_evolve", module)
    monkeypatch.chdir(tmp_path)
    # The HTML page is the view's business; only check it is written.
    monkeypatch.setattr(files, "write_html", lambda path, a: path.with_suffix(".html").touch())
    monkeypatch.setattr(repo, "snapshot", lambda: dict(SNAPSHOT))
    return module


@pytest.fixture(autouse=True)
def environ():
    """Restore ``os.environ`` and a fresh Langfuse decorator client."""
    from langfuse.utils.langfuse_singleton import LangfuseSingleton

    saved = dict(os.environ)
    LangfuseSingleton().reset()
    yield
    os.environ.clear()
    os.environ.update(saved)
    LangfuseSingleton().reset()


def _entry(agent_name="support", trace_name=None):
    return {
        "session_id": "s1",
        "source_trace_id": f"t-{agent_name}",
        "trace_name": trace_name or agent_name,
        "agent_name": agent_name,
        "source": {
            "user_message": "Where is my order?",
            "history_before": [{"role": "user", "content": "hi"}],
            "original_reply": "Old",
        },
        "feedback": [],
    }


def _frozen(tmp_path: Path, entries) -> Path:
    path = tmp_path / "frozen.json"
    files.write(path, {"schema_version": 1, "name": None, "entries": entries})
    return path


def _env(tmp_path: Path, rate="0") -> Path:
    path = tmp_path / ".env.evolve"
    path.write_text(
        f"LANGFUSE_SAMPLE_RATE={rate}\n"
        "export AGENTOPS_HOST=http://127.0.0.1:9  # never reached\n"
        "AGENTOPS_PUBLIC_KEY='pk-test'\nAGENTOPS_SECRET_KEY=\"sk-test\"\n"
    )
    return path


def _candidate(tmp_path: Path, name="support-system", meta_name=None) -> Path:
    text = tmp_path / f"{name}.txt"
    text.write_text("You help, kindly.")
    (tmp_path / f"{name}.json").write_text(
        json.dumps({"name": meta_name or name, "label": "production", "base_version": 7})
    )
    return text


def _replay(tmp_path, path, **kwargs):
    before = path.read_text()
    summary = replay(path, env_file=_env(tmp_path), module="fake_core_evolve", **kwargs)
    assert path.read_text() == before  # the frozen export is kept
    assert summary["path"].with_suffix(".html").is_file()
    return summary, json.loads(summary["path"].read_text())


def test_replay_writes_runs_with_captured_events(tmp_path, core, capsys):
    sales = _entry("sales", "sales/nordic")
    sales["source"].update(request_meta={"market": "se"}, user_context={"brand": "Acme"})
    path = _frozen(tmp_path, [_entry(), sales])
    summary, written = _replay(
        tmp_path, path, name="Kinder support", prompt_files={"support-system": _candidate(tmp_path)}
    )
    err = capsys.readouterr().err  # the env file's credentials are named on every run
    assert "read-only or sandbox credentials" in err and str(tmp_path / ".env.evolve") in err

    assert summary["path"].parent.name.endswith("_01-Kinder-support")
    assert summary["path"].name == "replay.json"
    del summary["path"]
    assert summary == {
        "runs": 2,
        "failed": 0,
        "model": "gpt-test",
        "provider": "azure",
        "dirty": False,
    }
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert core.runtimes[0].prompts == {"support-system": "You help, kindly."}
    calls = sorted(core.runtimes[0].calls)
    # The recorded request_meta and user_context are passed back.
    assert calls[0][3:] == ({"market": "se", "agent_variant": "nordic"}, {"brand": "Acme"})
    assert calls[1][3:] == (None, None)

    assert written["replay_id"] and written["name"] == "Kinder support"
    assert written["provenance"] == {**SNAPSHOT, "model": "gpt-test", "provider": "azure"}
    support, sales = written["entries"]
    run = support["run"]
    assert set(run) == RUN_KEYS and support["resolutions"] == []
    assert run["status"] == "complete" and run["reply"] == "re: Where is my order?"
    assert run["system_prompt"] == "You help, kindly."
    assert [m["role"] for m in run["trace"]] == ["user", "user", "assistant"]
    assert run["prompt"]["candidate"] is True and run["prompt"]["base_version"] == 7
    assert run["prompt"]["template_sha256"]
    assert run["tool_calls"] == [{"name": "lookup", "arguments": "{}"}]
    root = [e for e in run["events"] if e["type"] == "trace-create"]
    assert root[0]["body"]["id"] == support["run_id"]
    assert root[0]["body"]["name"] == "support"
    assert any(e["body"].get("name") == "agent-step" for e in run["events"])
    assert {e["body"].get("traceId", support["run_id"]) for e in run["events"]} == {
        support["run_id"]
    }
    assert sales["run"]["prompt"] is None
    assert sales["run"]["events"][0]["body"]["name"] == "sales/nordic"


def test_repeats_are_untraced_and_failures_recorded(tmp_path, core):
    core.fail.add("sales")
    path = _frozen(tmp_path, [_entry(), _entry("sales")])
    summary, written = _replay(tmp_path, path, repeat=3, concurrency=1)

    assert summary["failed"] == 1
    assert written["name"] == summary["path"].parent.name  # <date>_01-aaaaaaa
    assert summary["path"].parent.name.endswith("_01-" + "a" * 7)
    support, sales = written["entries"]
    assert len(core.runtimes[0].calls) == 6
    assert [r["status"] for r in support["run"]["repeats"]] == ["complete"] * 2
    assert set(support["run"]["repeats"][0]) == set(files.REPEAT_KEYS)
    # Repeats stay out of the run's events: one trace, one agent step (the run's).
    events = support["run"]["events"]
    traces = {e["body"].get("traceId", e["body"].get("id")) for e in events}
    assert traces == {support["run_id"]}
    steps = [e for e in events if e["body"].get("name") == "agent-step"]
    assert len({e["body"]["id"] for e in steps}) == 1
    assert sales["run"]["status"] == "failed"
    assert sales["run"]["error"] == "RuntimeError: model down"
    assert sales["run"]["repeats"][0]["error"] == "RuntimeError: model down"


def test_no_events_fails_the_run(tmp_path, core, monkeypatch):
    async def untraced(sink, trace_id, name, run):
        sink.watch(trace_id)
        return await run()

    monkeypatch.setattr(capture, "traced", untraced)
    monkeypatch.setattr(capture.TraceCapture, "add", lambda self, event: None)
    path = _frozen(tmp_path, [_entry()])
    summary, written = _replay(tmp_path, path, output=tmp_path / "out" / "mine.json")
    assert summary["failed"] == 1 and written["name"] == "mine"
    assert written["entries"][0]["run"]["error"] == NO_EVENTS


@pytest.mark.parametrize("rate", ["", "0.5", "x"])
def test_env_file_must_set_sample_rate_zero(tmp_path, core, rate):
    path = _frozen(tmp_path, [_entry()])
    with pytest.raises(ReplayRefused, match="LANGFUSE_SAMPLE_RATE=0"):
        replay(path, env_file=_env(tmp_path, rate), module="fake_core_evolve")
    with pytest.raises(ReplayRefused, match="does not exist"):
        replay(path, env_file=tmp_path / "missing", module="fake_core_evolve")
    assert not core.runtimes


@pytest.mark.parametrize(
    "entries, prompts, match",
    [
        ([_entry("billing")], None, "billing"),
        ([_entry()], {"other-system": "other-system"}, "no replayed agent uses AOS prompt"),
    ],
)
def test_checks_refuse_before_running(tmp_path, core, entries, prompts, match):
    path = _frozen(tmp_path, entries)
    prompt_files = {n: _candidate(tmp_path, f) for n, f in (prompts or {}).items()}
    with pytest.raises(ReplayRefused, match=match):
        _replay(tmp_path, path, prompt_files=prompt_files)
    assert not core.runtimes[0].calls and not (tmp_path / "local").exists()


def test_refuses_an_existing_output_and_bad_options(tmp_path, core):
    (tmp_path / "taken.html").touch()
    with pytest.raises(FileExistsError, match="taken.html exists"):
        _replay(tmp_path, _frozen(tmp_path, [_entry()]), output=tmp_path / "taken.json")
    with pytest.raises(ValueError, match="--repeat"):
        _replay(tmp_path, _frozen(tmp_path, [_entry()]), repeat=11)
    bad = _candidate(tmp_path, "support-system", meta_name="other")
    with pytest.raises(PromptError, match="is for prompt 'other'"):
        _replay(tmp_path, _frozen(tmp_path, [_entry()]), prompt_files={"support-system": bad})
    assert not core.runtimes


def test_a_failed_repeat_keeps_the_main_run_complete(tmp_path, core, monkeypatch):
    run_turn = Runtime.run_turn

    async def flaky(self, *args):
        if self.calls:  # every run after the first (the traced one)
            self.calls.append(args)
            raise RuntimeError("rate limited")
        return await run_turn(self, *args)

    monkeypatch.setattr(Runtime, "run_turn", flaky)
    summary, written = _replay(tmp_path, _frozen(tmp_path, [_entry()]), repeat=2, concurrency=1)
    run = written["entries"][0]["run"]
    assert summary["failed"] == 0 and run["status"] == "complete"
    assert run["repeats"][0]["status"] == "failed"
    assert run["repeats"][0]["error"] == "RuntimeError: rate limited"


def test_the_cli_lists_failed_runs_and_repeats(tmp_path, core, monkeypatch, capsys):
    from agentops.evolve import cli

    monkeypatch.setitem(sys.modules, "app.evolve_agent", core)
    core.fail.add("sales")
    path = _frozen(tmp_path, [_entry(), _entry("sales")])
    argv = ["replay", str(path), "--env-file", str(_env(tmp_path)), "--repeat", "3"]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out.splitlines()
    assert "Failed: trace t-sales: RuntimeError: model down" in out
    assert "3 runs per turn; 2 of 4 repeats complete" in out
    assert out.count("Failed repeat: trace t-sales: RuntimeError: model down") == 2


def test_settings_read_the_env_file_not_dotenv(tmp_path, monkeypatch):
    pydantic_settings = pytest.importorskip("pydantic_settings")
    from agentops.evolve.replay import load_env_file

    # Start unpatched (an earlier replay in this process patched it); restore after.
    init = pydantic_settings.BaseSettings.__init__
    monkeypatch.setattr(
        pydantic_settings.BaseSettings, "__init__", getattr(init, "_evolve_original", init)
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("EVOLVE_PROD_ONLY=prod\nEVOLVE_SHARED=prod\n")

    class Settings(pydantic_settings.BaseSettings):
        model_config = pydantic_settings.SettingsConfigDict(env_file=".env", extra="ignore")
        evolve_prod_only: str = "unset"
        evolve_shared: str = "unset"

    assert Settings().evolve_prod_only == "prod"
    evolve = tmp_path / ".env.evolve"
    evolve.write_text("LANGFUSE_SAMPLE_RATE=0\nEVOLVE_SHARED=sandbox\n")
    load_env_file(evolve)
    settings = Settings()
    assert settings.evolve_prod_only == "unset" and settings.evolve_shared == "sandbox"

    # A child process (core's tool server, a --local-mcp server) too.
    os.environ.pop("EVOLVE_SHARED")
    child = (
        "import pydantic_settings as p\n"
        "class S(p.BaseSettings):\n"
        "    model_config = p.SettingsConfigDict(env_file='.env', extra='ignore')\n"
        "    evolve_prod_only: str = 'unset'\n"
        "    evolve_shared: str = 'unset'\n"
        "s = S(); print(s.evolve_prod_only, s.evolve_shared)"
    )
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=False)
    assert out.stdout.split() == ["unset", "sandbox"], out.stderr


def test_local_mcp_runs_a_server_from_this_checkout_for_the_run(tmp_path, core):
    from agentops.evolve.local_mcp import _answers

    out = tmp_path / "out" / "replay.json"
    # A server that takes 0.5 s to stop, started as the shell's child ("&&
    # true", as Linux's sh does): stopping must wait for it, not the shell.
    server = (
        "import signal, sys, time, http.server as h; "
        "signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.5), sys.exit(0))); "
        "h.HTTPServer(('127.0.0.1', int(sys.argv[1])), h.BaseHTTPRequestHandler)"
        ".serve_forever()"
    )
    command = f'{sys.executable} -c "{server}" {{port}} && true'
    _, written = _replay(
        tmp_path, _frozen(tmp_path, [_entry()]), output=out, local_servers={"shop": command}
    )
    url = os.environ["MCP_URL_SHOP"]
    assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
    assert written["provenance"]["tool_servers"]["shop"] == "local"
    assert core.runtimes[0].calls  # the turn ran while the server was up
    assert not _answers(int(url.split(":")[2].split("/")[0]))  # stopped afterwards
    assert (out.parent / "mcp-shop.log").is_file()


def test_local_mcp_that_does_not_start_stops_the_replay(tmp_path, core):
    from agentops.evolve.local_mcp import LocalMcpError

    out = tmp_path / "out" / "replay.json"
    with pytest.raises(LocalMcpError, match="did not answer"):
        replay(
            _frozen(tmp_path, [_entry()]),
            env_file=_env(tmp_path),
            module="fake_core_evolve",
            output=out,
            local_servers={"shop": "exit 3"},
        )
    assert not core.runtimes and not out.exists()


def test_local_mcp_url_reaches_settings_that_read_it_from_the_env_file(tmp_path, monkeypatch):
    pytest.importorskip("pydantic_settings")
    from agentops.evolve import local_mcp
    from agentops.evolve.replay import load_env_file

    monkeypatch.chdir(tmp_path)
    evolve = tmp_path / ".env.evolve"
    evolve.write_text("LANGFUSE_SAMPLE_RATE=0\nMCP_URL_SHOP=http://localhost:8080/mcp\n")
    load_env_file(evolve)
    # Like a client config: extra keys come from the env file, not the environment.
    child = (
        "import pydantic_settings as p\n"
        "class S(p.BaseSettings):\n"
        "    model_config = p.SettingsConfigDict(env_file='.env', extra='allow')\n"
        "print(S().model_extra['mcp_url_shop'])"
    )
    command = f"{sys.executable} -m http.server {{port}} --bind 127.0.0.1"
    with local_mcp.started({"shop": command}, tmp_path):
        copy = os.environ["AGENTOPS_EVOLVE_ENV_FILE"]
        out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, check=False)
        assert out.stdout.strip() == os.environ["MCP_URL_SHOP"] != "http://localhost:8080/mcp", out.stderr
    assert os.environ["AGENTOPS_EVOLVE_ENV_FILE"] == str(evolve.resolve())
    assert not Path(copy).exists()


def test_images_go_to_the_agent_and_their_references_stay_in_the_record(
    tmp_path, core, monkeypatch
):
    from agentops.evolve.api import EvolveApi

    ref = "@@@langfuseMedia:type=image/png|id=m1|source=base64_data_uri@@@"
    entry = _entry()
    entry["source"]["attachments"] = [{"type": "image_base64", "data": ref}]
    entry["source"]["history_before"] = [{"role": "user", "content": [{"image_url": ref}]}]
    fetched = []
    monkeypatch.setattr(EvolveApi, "media", lambda self, i: fetched.append(i) or b"png")
    _, written = _replay(tmp_path, _frozen(tmp_path, [entry]))

    uri = "data:image/png;base64,cG5n"
    (_, _, history, _, _), = core.runtimes[0].calls
    assert core.runtimes[0].attachments == [[{"type": "image_base64", "data": uri}]]
    assert history == [{"role": "user", "content": [{"image_url": uri}]}]
    assert fetched == ["m1"]  # once, before the run
    run = written["entries"][0]["run"]
    recorded = json.dumps([run["events"], run["trace"]])
    assert ref in recorded and uri not in recorded
