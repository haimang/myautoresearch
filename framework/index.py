#!/usr/bin/env python3
"""Unified framework entrypoint."""

from __future__ import annotations

import importlib
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

COMMANDS = {
    "analyze": "framework.facade.analyze_cli",
    "sweep": "framework.facade.sweep_cli",
    "branch": "framework.facade.branch_cli",
    "promote": "framework.facade.promote_cli",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python framework/index.py <command> [args...]")
        print(f"Commands: {', '.join(sorted(COMMANDS))}")
        raise SystemExit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown framework command: {command}")
        print(f"Commands: {', '.join(sorted(COMMANDS))}")
        raise SystemExit(1)

    module = importlib.import_module(module_name)
    sys.argv = ["framework/index.py", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
