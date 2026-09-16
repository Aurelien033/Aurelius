"""Cohesive sub-modules for the local-first session store.

The public entry point stays :mod:`src.agent.session_manager`; this package
holds the implementation split by concern: value objects, id/path helpers,
disk persistence, session lifecycle, journal, workstreams, records and queued
work items.
"""

from __future__ import annotations
