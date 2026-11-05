"""Make every process of a replay read only its ``--env-file``.

``agentops evolve replay`` puts this folder first on ``PYTHONPATH`` and the
env file's path in ``AGENTOPS_EVOLVE_ENV_FILE``. Python imports
``sitecustomize`` at start, so each child process (core's tool server, a
``--local-mcp`` server) reads that file instead of the ``.env`` in its
working directory, as the replay process itself does. Without it a child
takes every key the env file leaves out from that ``.env``.
"""

import os

ENV_FILE_VAR = "AGENTOPS_EVOLVE_ENV_FILE"


def patch_settings(path: str) -> None:
    """Pydantic settings read *path* unless a caller passes ``_env_file``."""
    try:
        import pydantic_settings  # type: ignore[import-not-found]
    except ImportError:
        return
    base = pydantic_settings.BaseSettings
    if getattr(base.__init__, "_evolve_env_file", None) == path:
        return
    original = getattr(base.__init__, "_evolve_original", base.__init__)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("_env_file", path)
        original(self, *args, **kwargs)

    __init__._evolve_env_file = path  # type: ignore[attr-defined]
    __init__._evolve_original = original  # type: ignore[attr-defined]
    base.__init__ = __init__  # type: ignore[method-assign]


if os.environ.get(ENV_FILE_VAR):
    patch_settings(os.environ[ENV_FILE_VAR])
