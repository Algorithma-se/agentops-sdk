"""``agentops evolve``: the API client, turn selection, export, assess, the
views, the publish checks, prompt versions and revert. HTTP goes through the
``fake_http`` session; git facts are patched."""

from __future__ import annotations

import json
import types

import pytest
from _http_fakes import FakeResponse

from agentops.evolve import artifact as files
from agentops.evolve import cli, repo, versions, view
from agentops.evolve import publish as pub
from agentops.evolve.api import EvolveApi
from agentops.evolve.prompts import PromptError, candidate_paths, sha256_text

HEAD = "a" * 40


@pytest.fixture(autouse=True)
def configured(monkeypatch, fake_http):
    monkeypatch.setenv("AGENTOPS_HOST", "http://aos.test")
    monkeypatch.setenv("AGENTOPS_PUBLIC_KEY", "pk")
    monkeypatch.setenv("AGENTOPS_SECRET_KEY", "sk")
    monkeypatch.setattr(repo, "head", lambda: HEAD)
    monkeypatch.setattr(repo, "is_dirty", lambda: False)
    monkeypatch.setattr(repo, "user_name", lambda: "Dev")
    monkeypatch.setattr(repo, "is_ancestor", lambda commit: commit == HEAD)
    return fake_http


def ok(body):
    return FakeResponse(200, body)


def item(state, *, assessed=None, **extra):
    """A feedback object as ``include=feedback`` serves it."""
    assessments = (
        [{"feedback_revision": "r1", "status": "fixed", "git_commit": assessed, "created_at": "t"}]
        if assessed
        else []
    )
    return {"feedback_id": "f1", "revision": "r1", "state": state, "note": "too long",
            "assessments": assessments, "thread": [], **extra}  # fmt: skip


def turn(trace_id="t1", feedback=(), context=None):
    return {"trace_id": trace_id, "session_id": "s1", "trace_name": "support/v2",
            "feedback": list(feedback), "context": context}  # fmt: skip


def replayed(**run):
    """A replayed file with one complete run and one frozen feedback."""
    return {
        "schema_version": 1, "replay_id": "rp1", "name": "shorter", "created_at": "t",
        "project_id": "p1", "provenance": {"git_commit": HEAD, "dirty": False,
                                           "provider": "azure", "model": "gpt"},
        "entries": [{
            "run_id": "run1", "session_id": "s1", "source_trace_id": "t1",
            "trace_name": "support", "agent_name": "support",
            "source": {"user_message": "Hi", "original_reply": "<b>long</b>", "history_before": []},
            "feedback": [{"feedback_id": "f1", "revision": "r1", "note": "too long",
                          "author_name": "Rev", "scope": "reply"}],
            "run": {"status": "complete", "reply": "Short", "completed_at": "t",
                    "events": [{"id": "e"}], **run},
            "resolutions": [],
        }],
    }  # fmt: skip


def test_all_turns_reads_until_a_short_page(configured):
    configured.queue(ok({"data": [turn()] * 100}), ok({"data": [turn()]}))
    assert len(EvolveApi().all_turns({"traceName": "support"})) == 101
    assert [c.params["page"] for c in configured.calls] == [1, 2]


def test_find_unresolved_applies_the_head_rule(configured, capsys):
    fixed_here, fixed_elsewhere = item("fixed", assessed=HEAD), item("fixed", assessed="b" * 40)
    configured.queue(ok({"data": [turn("t1", [fixed_here]), turn("t2", [fixed_elsewhere])]}))
    assert cli.main(["find", "--trace-name", "support", "--feedback", "unresolved"]) == 0
    params = configured.calls[0].params
    assert params["feedback"] == "open,partial,fixed,closed"
    assert params["include"] == "feedback" and params["traceName"] == "support"
    printed = capsys.readouterr()
    assert [t["trace_id"] for t in json.loads(printed.out)["data"]] == ["t2"]
    assert printed.err.strip() == "1 turn(s), 1 feedback item(s)"


def test_exclude_skips_the_sessions_of_a_frozen_file(configured, tmp_path, capsys):
    target = tmp_path / "target.json"
    files.save(target, {**replayed(), "entries": [{**replayed()["entries"][0], "run": None}]})
    other = {**turn("t2"), "session_id": "s2"}
    found = tmp_path / "find.json"  # saved find output works too
    found.write_text(json.dumps({"data": [{**turn("t3"), "session_id": "s3"}]}))
    third = {**turn("t3"), "session_id": "s3"}
    configured.queue(ok({"data": [turn("t1"), other, third]}))
    argv = ["find", "--vote", "up", "--exclude", str(target), "--exclude", str(found)]
    assert cli.main(argv) == 0
    assert [t["trace_id"] for t in json.loads(capsys.readouterr().out)["data"]] == ["t2"]


def test_export_freezes_every_feedback_of_the_picked_turns(configured, tmp_path):
    context = {"user_message": "Hi", "history_before": [], "original_reply": "Long"}
    other = item("closed", feedback_id="f2")
    configured.queue(
        ok({"data": [turn("t1", [item("open")])]}),
        ok({"data": [turn("t1", [item("open"), other], context)]}),
        ok({"data": [{"id": "p1"}]}),
    )
    out = tmp_path / "frozen.json"
    assert cli.main(["export", "--feedback", "open", "--output", str(out)]) == 0
    assert configured.calls[1].params["traceId"] == "t1"
    assert configured.calls[1].params["include"] == "feedback,context"
    saved = files.read(out)
    (entry,) = saved["entries"]
    assert saved["project_id"] == "p1" and entry["agent_name"] == "support"
    assert entry["source"] == context and entry["run"] is None
    assert [f["feedback_id"] for f in entry["feedback"]] == ["f1", "f2"]
    assert set(entry["feedback"][0]) == set(cli.FROZEN_KEYS)
    assert out.with_suffix(".html").exists()


def test_assess_then_show_and_render(tmp_path, capsys):
    path = tmp_path / "replay.json"
    files.write(path, replayed())
    args = ["assess", str(path), "--trace-id", "t1", "--feedback-id", "f1", "--status", "fixed",
            "--explanation", "Shorter now", "--evidence", "Short", "--changes", "prompt",
            "--author", "Agent"]  # fmt: skip
    assert cli.main(args) == 0
    (resolution,) = files.read(path)["entries"][0]["resolutions"]
    assert resolution["changes"] == f"{HEAD[:12]}: prompt"
    assert resolution["feedback_revision"] == "r1"
    assert cli.main(["show", str(path)]) == 0
    assert "Assessment (fixed, Agent): Shorter now" in capsys.readouterr().out
    page = (tmp_path / "replay.html").read_text()
    assert "&lt;b&gt;long&lt;/b&gt;" in page and "<b>long</b>" not in page


def test_view_cuts_long_text():
    assert view.cut("x" * 700, 600).endswith("[100 more chars; --full]")
    assert "replay support v8: changed (versions)" in view.entry_view(
        {"source": {"prompt": {"name": "support", "version": 7}},
         "run": {"prompt": {"name": "support", "version": 8}}}
    )["prompt"]  # fmt: skip


def test_publish_problems():
    artifact = replayed(events=[])
    artifact["provenance"]["dirty"] = True
    artifact["entries"][0]["resolutions"] = [{"feedback_id": "f1", "feedback_revision": "r0",
                                              "changes": "no commit"}]  # fmt: skip
    found = pub.problems(artifact, head="c" * 40, repo_dirty=True, project_id="p2")
    assert len(found) == 7
    assert pub.repeat_problems({"repeats": [{"status": "odd"}]}) == [
        "run.repeats[1] must be an object with status, reply, error, started_at, completed_at"
    ]


def test_publish_previews_then_applies(configured, tmp_path, capsys):
    path = tmp_path / "replay.json"
    files.write(path, replayed())
    configured.queue(ok({"data": []}), ok({"data": [{"id": "p1"}]}))
    assert cli.main(["publish", str(path)]) == 0
    assert "Preview only" in capsys.readouterr().out
    assert not configured.posts
    configured.queue(ok({"data": []}), ok({"data": [{"id": "p1"}]}), ok({"duplicate": False}))
    assert cli.main(["publish", str(path), "--apply"]) == 0
    (post,) = configured.posts
    assert post.url.endswith("/evolve/publications") and post.json["published_by"] == "Dev"


def test_revert_needs_apply(configured, tmp_path):
    path = tmp_path / "replay.json"
    files.write(path, replayed())
    published = {"replay_id": "rp1", "name": "shorter", "status": "published", "created_at": "2026"}
    configured.queue(ok({"data": [published]}))
    assert cli.main(["revert", str(path)]) == 0
    assert not configured.posts
    configured.queue(ok({"data": [published]}), ok({**published, "status": "reverted"}))
    assert cli.main(["revert", "rp1", "--apply"]) == 0
    assert configured.posts[0].url.endswith("/evolve/publications/rp1/revert")


def test_default_agents_from_agents_yaml(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    monkeypatch.chdir(tmp_path)
    assert cli.default_agents() == []
    (tmp_path / "config").mkdir()
    (tmp_path / "config/agents.yaml").write_text("agents:\n  support: {}\n  sales: {}\n")
    assert cli.default_agents() == ["support", "sales"]


# --- Selection flags (find, export)


def test_find_flags_map_to_query_params(configured, monkeypatch, capsys):
    monkeypatch.setattr(cli, "default_agents", lambda: ["support"])
    configured.queue(ok({"data": []}), ok({"data": []}), ok({"data": [turn()]}))
    argv = ["find", "--like", "t1", "--limit", "5", "--trace-name", "support",
            "--trace-name", "sales", "--tool", "a", "--tool", "b"]  # fmt: skip
    assert cli.main(argv) == 0
    assert configured.calls[0].params == {
        "like": "t1", "sameTools": "true", "traceName": "support,sales", "tool": "a,b",
        "include": "feedback", "page": 1, "limit": 5,
    }  # fmt: skip
    assert "--any-tools" in capsys.readouterr().err  # nothing used the same tools
    assert cli.main(["find", "--like", "t1", "--any-tools"]) == 0
    assert configured.calls[1].params["sameTools"] == "false"
    assert "traceName" not in configured.calls[1].params  # --like: no default agents
    argv = ["find", "--session", "s1", "--session", "s2", "--since", "2026-10-01T00:00:00Z",
            "--until", "2026-10-02", "--text", "order", "--tool-error", "--vote", "down",
            "--feedback", "open,fixed", "--source", "user", "--page", "2"]  # fmt: skip
    assert cli.main(argv) == 0
    assert configured.calls[2].params == {
        "traceName": "support", "sessionId": "s1,s2", "fromTimestamp": "2026-10-01T00:00:00Z",
        "toTimestamp": "2026-10-02", "text": "order", "toolError": "true", "vote": "down",
        "feedback": "open,fixed", "source": "user", "include": "feedback", "page": 2,
        "limit": 50,
    }  # fmt: skip


@pytest.mark.parametrize("flag", [["--trace-id", "t1"], ["--feedback-id", "f1"]])
def test_named_turns_skip_the_default_agents(configured, monkeypatch, flag):
    monkeypatch.setattr(cli, "default_agents", lambda: ["support"])
    configured.queue(ok({"data": []}))
    assert cli.main(["find", *flag]) == 0
    assert "traceName" not in configured.calls[0].params


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--feedback", "open,nope"], "nope"),
        (["--any-tools"], "--any-tools only applies to --like"),
        (["--feedback", "none", "--vote", "up"], "excludes --vote"),
        (["--since", "last week"], "neither a duration"),
        (["--since", "1d", "--until", "2d"], "--since is later than --until"),
        (["--since", "2026-10-02", "--until", "2026-10-01T00:00:00Z"], "later than"),
        (["--limit", "101"], "limit must be between 1 and 100"),
        (["--page", "0"], "page must be 1 or more"),
    ],
)
def test_find_misuse_fails_before_anything_is_sent(configured, capsys, argv, message):
    assert cli.main(["find", "--trace-name", "support", *argv]) == 2
    assert message in capsys.readouterr().err
    assert configured.calls == []


def test_parse_time_durations_and_iso():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    for value, delta in (("7d", timedelta(days=7)), ("24h", timedelta(hours=24)),
                         ("30m", timedelta(minutes=30))):  # fmt: skip
        moment = datetime.fromisoformat(cli.parse_time(value, "--since").replace("Z", "+00:00"))
        assert abs(now - delta - moment) < timedelta(minutes=1)
    iso = "2026-10-01T08:00:00+02:00"
    assert cli.parse_time(iso, "--since") == iso
    with pytest.raises(cli.Fail, match="neither a duration"):
        cli.parse_time("7w", "--since")


def test_a_server_400_shows_its_detail(configured, capsys):
    configured.queue(FakeResponse(400, {"message": "sameTools needs like"}))
    assert cli.main(["find", "--trace-name", "support"]) == 1
    assert "sameTools needs like" in capsys.readouterr().err


# --- --env-file


def test_env_file_reads_only_the_agentops_keys_and_wins(configured, monkeypatch, tmp_path, capsys):
    import base64
    import os

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env.evolve"
    env_file.write_text(
        "# AOS Control\nexport AGENTOPS_HOST='https://aos.example'\n"
        'AGENTOPS_PUBLIC_KEY="pk-file"\nAGENTOPS_SECRET_KEY=sk-file-secret # comment\n'
        "OPENAI_API_KEY=not-read\n"
    )
    configured.queue(ok({"data": []}))
    assert cli.main(["find", "--trace-name", "s", "--env-file", str(env_file)]) == 0
    (call,) = configured.calls
    assert call.url.startswith("https://aos.example/")
    token = call.headers["Authorization"].removeprefix("Basic ")
    assert base64.b64decode(token).decode() == "pk-file:sk-file-secret"
    assert "OPENAI_API_KEY" not in os.environ
    out = capsys.readouterr()
    assert "sk-file-secret" not in out.out + out.err


def test_an_empty_env_file_value_keeps_the_environment(configured, tmp_path):
    env_file = tmp_path / ".env.evolve"
    env_file.write_text("AGENTOPS_HOST=\nAGENTOPS_SECRET_KEY=sk2\n")
    configured.queue(ok({"data": []}))
    assert cli.main(["find", "--trace-name", "s", "--env-file", str(env_file)]) == 0
    assert configured.calls[0].url.startswith("http://aos.test/")


def test_a_missing_env_file_exits_2_naming_it(configured, tmp_path, capsys):
    missing = tmp_path / "nope.env"
    assert cli.main(["find", "--env-file", str(missing)]) == 2
    assert str(missing) in capsys.readouterr().err
    assert configured.calls == []


@pytest.mark.parametrize(
    "prompts, message",
    [(["support"], "expected NAME=PATH"), (["a=x.txt", "a=y.txt"], "more than once")],
)
def test_replay_refuses_bad_prompt_flags(tmp_path, capsys, prompts, message):
    argv = ["replay", str(tmp_path / "f.json"), "--env-file", str(tmp_path / "e")]
    for spec in prompts:
        argv += ["--prompt", spec]
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


# --- export

# The server's strict artifact schema (web/src/features/public-api/types/
# evolvePublications.ts: EvolveReplayArtifact, EvolveArtifactEntry,
# EvolveResolution); an unknown key is a 400.
ARTIFACT_KEYS = {"schema_version", "replay_id", "name", "created_at", "agentops_host",
                 "project_id", "provenance", "entries"}  # fmt: skip
ENTRY_KEYS = {"run_id", "session_id", "source_trace_id", "trace_name", "agent_name", "source",
              "feedback", "run", "resolutions"}  # fmt: skip
RESOLUTION_KEYS = {"feedback_id", "feedback_revision", "status", "explanation", "evidence",
                   "changes", "author", "created_at"}  # fmt: skip


def test_export_then_assess_write_only_the_server_schema_keys(configured, tmp_path):
    context = {"user_message": "Hi", "history_before": [], "original_reply": "Long"}
    configured.queue(
        ok({"data": [turn("t1", [item("open")])]}),
        ok({"data": [turn("t1", [item("open")], context)]}),
        ok({"data": [{"id": "p1"}]}),
    )
    out = tmp_path / "frozen.json"
    assert cli.main(["export", "--trace-id", "t1", "--output", str(out)]) == 0
    saved = files.read(out)
    assert set(saved) == ARTIFACT_KEYS and set(saved["entries"][0]) == ENTRY_KEYS
    path = tmp_path / "replay.json"
    files.write(path, {**saved, **replayed()})
    _assess(path)
    (resolution,) = files.read(path)["entries"][0]["resolutions"]
    assert set(resolution) == RESOLUTION_KEYS


def test_export_names_trace_ids_that_came_back_missing(configured, tmp_path, capsys):
    configured.queue(ok({"data": [turn("t1")]}))
    out = tmp_path / "frozen.json"
    argv = ["export", "--trace-id", "t1", "--trace-id", "t-other", "--output", str(out)]
    assert cli.main(argv) == 2
    assert "t-other" in capsys.readouterr().err
    assert not out.exists() and len(configured.calls) == 1


def test_export_needs_a_selection(configured, tmp_path, capsys):
    assert cli.main(["export", "--trace-name", "support", "--output", str(tmp_path / "x")]) == 2
    assert "pick turns with" in capsys.readouterr().err
    assert configured.calls == []


# --- assess


def _assess(path, *selector, feedback_id="f1", status="fixed", explanation="Shorter"):
    argv = ["assess", str(path), *(selector or ("--run-id", "run1")), "--feedback-id",
            feedback_id, "--status", status, "--explanation", explanation, "--evidence", "e",
            "--changes", "prompt", "--author", "Agent"]  # fmt: skip
    return cli.main(argv)


def test_assess_replaces_an_earlier_assessment_and_stamps_the_sha(tmp_path):
    path = tmp_path / "replay.json"
    files.write(path, replayed())
    assert _assess(path, status="partial") == 0
    assert _assess(path, explanation="Now fixed") == 0
    (resolution,) = files.read(path)["entries"][0]["resolutions"]
    assert resolution["status"] == "fixed" and resolution["explanation"] == "Now fixed"
    assert resolution["changes"].startswith(HEAD[:12] + ": ")


@pytest.mark.parametrize(
    "run, feedback_id, message",
    [({}, "f9", "f9 is not frozen"), ({"status": "failed"}, "f1", "a completed replay")],
)
def test_assess_refusals(tmp_path, capsys, run, feedback_id, message):
    path = tmp_path / "replay.json"
    files.write(path, replayed(**run))
    assert _assess(path, feedback_id=feedback_id) == 2
    assert message in capsys.readouterr().err
    assert files.read(path)["entries"][0]["resolutions"] == []


def test_assess_picks_the_entry_by_trace_id_or_one_session(tmp_path, capsys):
    artifact = replayed()
    second = {**artifact["entries"][0], "run_id": "run2", "source_trace_id": "t2"}
    artifact["entries"].append(second)
    path = tmp_path / "replay.json"
    files.write(path, artifact)
    assert _assess(path, "--trace-id", "t2") == 0
    first, second = files.read(path)["entries"]
    assert first["resolutions"] == [] and len(second["resolutions"]) == 1
    assert _assess(path, "--session", "s1") == 2
    assert "matches runs run1, run2; use --run-id" in capsys.readouterr().err
    assert _assess(path, "--session", "s9") == 2
    assert "no entry with --session s9" in capsys.readouterr().err


# --- ask, show, view


def test_ask_posts_the_question_and_withdraw_deletes_it(configured, capsys):
    configured.queue(FakeResponse(201, {"id": "q1", "duplicate": False}))
    argv = ["ask", "--feedback-id", "f1", "--question", "10-16 or 10-18?", "--author", "Dev"]
    assert cli.main(argv) == 0
    (post,) = configured.posts
    assert post.url.endswith("/api/public/evolve/questions")
    assert post.json == {"feedback_id": "f1", "question": "10-16 or 10-18?", "author": "Dev"}
    out = capsys.readouterr()
    assert json.loads(out.out)["id"] == "q1" and "Asked." in out.err
    configured.queue(ok({"id": "q1"}))
    assert cli.main(["ask", "--withdraw", "q1"]) == 0
    assert (configured.calls[-1].method, configured.calls[-1].url) == (
        "DELETE",
        "http://aos.test/api/public/evolve/questions/q1",
    )
    assert "Withdrawn." in capsys.readouterr().err


def test_show_counts_repeats_and_marks_identical_replies():
    repeats = [{"status": "complete", "reply": "Short", "error": None},
               {"status": "failed", "reply": None, "error": "boom"}]  # fmt: skip
    text = "\n".join(view.summary(replayed(repeats=repeats)))
    assert (
        "Runs: 3 (the run and 2 repeat(s)) · complete 2 of 3 · same reply as the run: 1 of 2"
        in text
    )
    assert "Repeat 1: complete · same reply as the run" in text
    assert "Repeat 2: failed · error: boom" in text


def test_show_of_a_frozen_export_and_a_failed_run():
    frozen = replayed(status="failed", error="RuntimeError: down", reply=None)
    assert "Status: failed · error: RuntimeError: down" in "\n".join(view.summary(frozen))
    frozen = {**frozen, "provenance": None}
    frozen["entries"][0]["run"] = None
    text = "\n".join(view.summary(frozen))
    assert "Not replayed yet (no provenance)." in text and "Not replayed." in text


def test_prompt_changed_and_describe_prompt():
    entry = {"source": {"prompt": {"name": "s", "version": 7}, "system_prompt": "A"},
             "run": {"prompt": {"name": "s", "version": 7}, "system_prompt": "B"}}  # fmt: skip
    assert view.prompt_changed(entry) == (False, "versions")
    entry["run"]["prompt"] = {"name": "s", "candidate": True, "base_version": 7}
    assert view.prompt_changed(entry) == (True, "text")
    entry["run"]["system_prompt"] = None
    assert view.prompt_changed(entry) == (None, "")
    assert view.describe_prompt(entry["run"]["prompt"]) == "s candidate (base v7)"
    assert view.describe_prompt(None) == "no AOS prompt"
    assert view.describe_prompt({"name": "s", "version": None}) == "s"


def test_tools_used_lists_the_calls_after_the_last_user_message():
    trace = [
        {"role": "user", "content": "old"},
        {"type": "function_call", "name": "history_tool"},
        {"role": "user", "content": "now"},
        {"type": "function_call", "name": "get_order"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_tracking"}}]},
    ]
    assert view.tools_used({"trace": trace}) == ["get_order", "get_tracking"]


# --- Prompt versions and publish


class FakeSdk:
    """The AgentOps client's prompt calls: versions 1..n of one text prompt."""

    def __init__(self, production=2):
        self.texts = ["v1 text", "base text"]
        self.production = production
        self.created = []

    def get_prompt(self, name, *, label, type, cache_ttl_seconds):
        number = len(self.texts) if label == "latest" else self.production
        return types.SimpleNamespace(prompt=self.texts[number - 1], version=number)

    def create_prompt(self, **kwargs):
        self.created.append(kwargs)
        self.texts.append(kwargs["prompt"])
        return types.SimpleNamespace(version=len(self.texts))


CANDIDATE = "Be short."
SHA = sha256_text(CANDIDATE)


def _candidate(prompt_dir, text=CANDIDATE, base_version=2):
    text_path, meta_path = candidate_paths("support", prompt_dir)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text)
    meta = {"name": "support", "label": "production", "base_version": base_version}
    meta_path.write_text(json.dumps(meta))
    return prompt_dir


def _candidate_artifact():
    ref = {"name": "support", "version": None, "candidate": True, "base_version": 2,
           "template_sha256": SHA}  # fmt: skip
    artifact = replayed(prompt=ref)
    artifact["entries"][0]["resolutions"] = [{"feedback_id": "f1", "feedback_revision": "r1",
                                              "changes": f"{HEAD[:12]}: prompt"}]  # fmt: skip
    return artifact


def test_pull_writes_txt_and_json_and_never_overwrites(tmp_path):
    store = versions.PromptStore(FakeSdk())
    meta = versions.pull(store, "support", output_dir=tmp_path)
    assert (tmp_path / "support.txt").read_text() == "base text"
    assert json.loads((tmp_path / "support.json").read_text()) == {
        "name": "support", "label": "production", "base_version": 2,
        "sha256": sha256_text("base text"),
    }  # fmt: skip
    assert meta["path"] == str(tmp_path / "support.txt")
    with pytest.raises(FileExistsError):
        versions.pull(store, "support", output_dir=tmp_path)


def test_a_candidate_edited_after_the_replay_is_refused(tmp_path):
    store = versions.PromptStore(FakeSdk())
    prompt_dir = _candidate(tmp_path, "edited")
    found = versions.candidate_problems(_candidate_artifact(), store, prompt_dir=prompt_dir)
    assert any("changed after the replay" in p for p in found)


def test_a_candidate_whose_json_is_missing_or_stale_or_label_moved_is_refused(tmp_path):
    store = versions.PromptStore(FakeSdk())
    prompt_dir = _candidate(tmp_path, base_version=1)
    found = versions.candidate_problems(_candidate_artifact(), store, prompt_dir=prompt_dir)
    assert any("does not match the replay's base version 2" in p for p in found)
    (tmp_path / "support.json").unlink()
    found = versions.candidate_problems(_candidate_artifact(), store, prompt_dir=prompt_dir)
    assert any("support.json is missing" in p for p in found)
    moved = versions.PromptStore(FakeSdk(production=1))
    prompt_dir = _candidate(tmp_path)
    (found,) = versions.candidate_problems(_candidate_artifact(), moved, prompt_dir=prompt_dir)
    assert "label 'production' now points to version 1" in found


def test_an_edit_between_check_and_push_is_refused_and_creates_nothing(tmp_path):
    sdk = FakeSdk()
    store = versions.PromptStore(sdk)
    prompt_dir = _candidate(tmp_path)
    assert versions.candidate_problems(_candidate_artifact(), store, prompt_dir=prompt_dir) == []
    _candidate(tmp_path, "edited later")
    with pytest.raises(PromptError, match="changed after the replay"):
        versions.push_candidates(_candidate_artifact(), store, prompt_dir=prompt_dir)
    assert sdk.created == []


@pytest.fixture
def publishing(configured, monkeypatch, tmp_path):
    """A replay with a candidate prompt, ready to publish; returns (sdk, argv, path)."""
    sdk = FakeSdk()
    store = versions.PromptStore(sdk)
    monkeypatch.setattr(versions, "PromptStore", lambda: store)
    path = tmp_path / "replay.json"
    files.write(path, _candidate_artifact())
    prompt_dir = _candidate(tmp_path / "prompts")
    return sdk, ["publish", str(path), "--prompts-dir", str(prompt_dir)], path


PUBLISHED = {"replay_id": "rp1", "status": "published", "created_at": "2026-10-21T07:29:58Z",
             "updated_at": "2026-10-22T07:29:58Z"}  # fmt: skip


def test_publish_apply_sends_prompt_versions_outside_the_artifact(configured, publishing, capsys):
    sdk, argv, path = publishing
    configured.queue(ok({"data": []}), ok({"data": [{"id": "p1"}]}), ok({"duplicate": False}))
    assert cli.main([*argv, "--apply"]) == 0
    (post,) = configured.posts
    assert post.json["prompt_versions"] == [{"name": "support", "template_sha256": SHA,
                                             "version": 3}]  # fmt: skip
    assert post.json["published_by"] == "Dev"
    body = {k: v for k, v in post.json.items() if k not in ("prompt_versions", "published_by")}
    assert body == files.read(path)
    out = capsys.readouterr().err
    assert "Prompt support: version 3\n" in out
    assert "publish the same artifact again" in out
    # A retry after a failed POST reuses the version it created.
    configured.queue(ok({"data": []}), ok({"data": [{"id": "p1"}]}), ok({"duplicate": True}))
    assert cli.main([*argv, "--apply"]) == 0
    (created,) = sdk.created  # one unlabelled version, not a second one
    assert created["labels"] == [] and created["prompt"] == CANDIDATE
    assert created["commit_message"] == "Evolve Agent: shorter"
    assert configured.posts[1].json["prompt_versions"] == post.json["prompt_versions"]
    assert "Prompt support: version 3 reused" in capsys.readouterr().err


@pytest.mark.parametrize("status, note", [("published", "Already published on 2026-10-21"),
                                          ("reverted", "--apply publishes it again")])  # fmt: skip
def test_preview_of_a_published_replay_says_so_and_skips_the_prompt_check(
    configured, publishing, capsys, status, note
):
    sdk, argv, _ = publishing
    sdk.production = 3  # a person moved the label after the publish
    configured.queue(ok({"data": [{**PUBLISHED, "status": status}]}), ok({"data": [{"id": "p1"}]}))
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert note in out and "Cannot publish" not in out and "creates a version" not in out
    assert sdk.created == [] and configured.posts == []


def test_apply_on_a_published_replay_creates_no_version(configured, publishing, capsys):
    sdk, argv, _ = publishing
    sdk.production = 3
    configured.queue(ok({"data": [PUBLISHED]}), ok({"data": [{"id": "p1"}]}),
                     ok({**PUBLISHED, "duplicate": True}))  # fmt: skip
    assert cli.main([*argv, "--apply"]) == 0
    assert sdk.created == []
    (post,) = configured.posts
    assert "prompt_versions" not in post.json
    assert "Already published (same digest)." in capsys.readouterr().err


@pytest.mark.parametrize("status", [400, 401, 500])
def test_a_failed_publication_lookup_stops_publish(configured, publishing, capsys, status):
    sdk, argv, _ = publishing
    configured.queue(FakeResponse(status, {"error": "nope"}))
    assert cli.main([*argv, "--apply"]) == 1
    assert f"HTTP {status}" in capsys.readouterr().err
    assert sdk.created == [] and configured.posts == []
