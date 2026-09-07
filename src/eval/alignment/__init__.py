"""Alignment EvalLab — release gates for Aurelius model ships.

Method-transfer from published Anthropic research (Teaching Claude Why /
Global Workspace / agentic-misalignment posts), re-grounded for this project:
deterministic, choice-verified scoring (no LLM judges), small clean scenario
bank authored in-repo (no external-model outputs -> firewall-safe), and an
honest scope note: these are CHOICE-BASED PROXIES for agentic misalignment,
not real tool-use rollouts (those are the v4 harness's job).

Modules:
  scenarios             — 24-scenario bank (8 domains x core/OOD splits) + prompt builder
  agentic_misalignment  — run + parse + summarize + G_align
  identity_perturbation — same scenarios under renamed/subagent/roleplay identities
  tool_augmented_safety — same scenarios with (unused) tool schemas visible
  why_density           — deterministic principle-causal density scorer for datasets
  release_gate          — CLI runner: scorecard + PASS/FAIL gates + model-card section
"""
from .scenarios import SCENARIOS, IDENTITIES, build_prompt
from .agentic_misalignment import parse_decision, run_suite, summarize, g_align
from .why_density import why_density_score, is_action_only, score_dataset, passes_why_gate
