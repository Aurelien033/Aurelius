# ruff: noqa: F403, F405
"""Aurelius terminal shell surface.

This module is a thin re-export facade: the implementation now lives in the
cohesive :mod:`src.ui.shell` sub-modules (models, state, interactions, skills,
workflow, snapshot, commands, render, capabilities) and every historical import
path under ``src.ui.aurelius_shell`` keeps working unchanged.

The shell itself stays thin and explicit: it delegates thread, approval,
checkpoint, subagent, background-job, channel-routing, and tool-call work to
:class:`src.model.interface_framework.AureliusInterfaceFramework` and only
maintains the minimal shell session state needed to render and persist a
terminal-first interaction surface.
"""

from __future__ import annotations

from src.ui.shell.capabilities import *
from src.ui.shell.commands import *
from src.ui.shell.interactions import *
from src.ui.shell.models import *
from src.ui.shell.render import *
from src.ui.shell.shell import *
from src.ui.shell.skills import *
from src.ui.shell.snapshot import *
from src.ui.shell.state import *
from src.ui.shell.util import *
from src.ui.shell.workflow import *

__all__ = [
    "AureliusShell",
    "AureliusShellError",
    "MessageEnvelope",
    "SkillRecord",
    "Workstream",
    "WorkflowRun",
    "WorkflowStep",
]
