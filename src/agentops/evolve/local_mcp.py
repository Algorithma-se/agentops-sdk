"""``replay --local-mcp NAME=COMMAND``: run a tool server from this checkout.

A replay must test the tools of the commit it records. An MCP server the
agent reaches over ``MCP_URL_<NAME>`` (a deployed one, or a local one started
before a change) runs other code. For each ``NAME=COMMAND`` the replay starts
``COMMAND`` (a shell command; ``{port}`` becomes a free port, also in
``$PORT``), waits until the port answers, points ``MCP_URL_<NAME>`` at
``http://127.0.0.1:<port>/mcp`` and stops the server afterwards. The server
inherits the replay's environment, so it reads only the ``--env-file`` too.

Settings may read ``MCP_URL_<NAME>`` from the env file only (pydantic
settings with ``extra="allow"`` take extra keys from the file, not from the
environment). So while the servers run, every process reads a private copy
of the env file with those URLs set (``_with_urls``), deleted afterwards.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .child.sitecustomize import ENV_FILE_VAR, patch_settings

START_TIMEOUT_S = 60
STOP_TIMEOUT_S = 10


class LocalMcpError(RuntimeError):
    """A local tool server did not start; nothing has been replayed."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _answers(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _group_alive(proc: subprocess.Popen) -> bool:
    proc.poll()  # reap the shell, so only live members count
    try:
        os.killpg(proc.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # macOS: only zombies are left in the group
        return False
    return True


def _stop(proc: subprocess.Popen) -> None:
    """Stop the server's whole process group: the shell may exit before the
    server it started (``/bin/sh`` on Linux does not exec the command)."""
    if not _group_alive(proc):
        return
    os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while _group_alive(proc) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _group_alive(proc):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


@contextmanager
def _with_urls(urls: dict[str, str]) -> Iterator[None]:
    """Point ``ENV_FILE_VAR`` (and this process's settings) at a copy of
    the env file whose ``urls`` keys are replaced; restore and delete after."""
    original = os.environ.get(ENV_FILE_VAR)
    if not urls or not original:
        yield
        return
    folder = Path(tempfile.mkdtemp(prefix="agentops-evolve-"))  # mode 0700
    try:
        kept = [
            line
            for line in Path(original).read_text(encoding="utf-8").splitlines()
            if line.strip().removeprefix("export ").partition("=")[0].strip() not in urls
        ]
        copy = folder / ".env.evolve"
        copy.write_text("\n".join([*kept, *(f"{k}={v}" for k, v in urls.items())]) + "\n")
        os.environ[ENV_FILE_VAR] = str(copy)
        patch_settings(str(copy))
        yield
    finally:
        os.environ[ENV_FILE_VAR] = original
        patch_settings(original)
        shutil.rmtree(folder, ignore_errors=True)


def tool_servers(local: dict[str, str]) -> dict[str, str]:
    """Which server each ``MCP_URL_<NAME>`` reached: ``local`` or its URL
    (without a query string, which may hold a key). For the provenance."""
    found = {}
    for key, url in os.environ.items():
        if key.startswith("MCP_URL_") and url:
            name = key[len("MCP_URL_") :].lower()
            parts = urlsplit(url)
            found[name] = f"{parts.scheme}://{parts.netloc}{parts.path}"
    return {**found, **{name.lower(): "local" for name in local}}


@contextmanager
def started(servers: dict[str, str], log_dir: Path) -> Iterator[None]:
    """Run each ``name -> command`` while the block runs; see the module.
    A SIGTERM (a harness timeout) exits through the cleanup below, so the
    servers and the env-file copy with secrets do not stay behind."""
    procs: list[subprocess.Popen] = []
    logs = []
    urls = {}
    on_term = None
    if servers and threading.current_thread() is threading.main_thread():
        on_term = signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        for name, command in servers.items():
            port = _free_port()
            log = (log_dir / f"mcp-{name.lower()}.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                command.replace("{port}", str(port)),
                shell=True,
                env={**os.environ, "PORT": str(port)},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append(proc)
            deadline = time.monotonic() + START_TIMEOUT_S
            while not _answers(port):
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise LocalMcpError(
                        f"--local-mcp {name}: the server did not answer on port {port} "
                        f"within {START_TIMEOUT_S} s; see {log.name}"
                    )
                time.sleep(0.5)
            urls[f"MCP_URL_{name.upper()}"] = f"http://127.0.0.1:{port}/mcp"
        os.environ.update(urls)
        with _with_urls(urls):
            yield
    finally:
        for proc in procs:
            _stop(proc)
        for log in logs:
            log.close()
        if on_term is not None:
            signal.signal(signal.SIGTERM, on_term)
