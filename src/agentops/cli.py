"""The ``agentops`` console script. Today it has one command group: ``evolve``."""

from __future__ import annotations

import argparse
from typing import Optional

from .evolve import cli as evolve_cli


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agentops", description="AgentOps SDK tools.")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    evolve = sub.add_parser(
        "evolve",
        help="Evolve Agent: find, export, replay, assess, publish",
        description=evolve_cli.DESCRIPTION,
    )
    evolve_cli.add_subcommands(evolve)
    return evolve_cli.run(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
