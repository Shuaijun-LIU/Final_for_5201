#!/usr/bin/env python3
"""
Thin wrapper kept for backward-compatible usage:

    python3 path2/RL_trajectory_planner/plan.py ...

Implementation has been split into multiple modules under `path2/RL_trajectory_planner/`.
"""

import os
import sys

# Ensure this directory is on sys.path so sibling modules can be imported when running as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from cli import main  # noqa: E402


if __name__ == "__main__":
    main()


