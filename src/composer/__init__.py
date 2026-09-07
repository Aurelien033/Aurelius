"""
Aurelius Composer — autonomous coding agent inspired by Cursor Composer 2.5.

Components:
  - CodebaseIndexer: repository understanding via AST + ripgrep
  - ContextAssembler: token-budgeted code context selection
  - DiffEngine: precise unified-diff generation and application
  - MultiFileEditOrchestrator: dependency-aware edit planning
  - EditVerifier: targeted verification gates
  - CheckpointRollback: file-scoped rollback checkpoints
  - TerminalSandbox: safe command execution
  - ComposerAgent: the main orchestration loop
"""

from .checkpoint_rollback import Checkpoint, CheckpointFile, CheckpointRollback
from .codebase_indexer import CodebaseIndexer, FileEntry, SearchHit, SymbolDef
from .composer_agent import (
    AgentAction,
    AgentConfig,
    AgentResult,
    AgentTurn,
    ComposerAgent,
)
from .context_assembler import (
    AssembledContext,
    ContextAssembler,
    ContextBudget,
    ContextItem,
    ContextRequest,
)
from .diff_engine import (
    ApplyResult,
    DiffApplier,
    DiffGenerator,
    DiffParser,
    EditHunk,
    FileEdit,
    MultiFileEdit,
)
from .edit_verifier import EditVerifier, VerificationCheck, VerificationResult
from .multifile_orchestrator import EditPlan, EditTarget, MultiFileEditOrchestrator
from .terminal_sandbox import SandboxConfig, SandboxResult, TerminalSandbox

__all__ = [
    # Codebase
    "CodebaseIndexer",
    "FileEntry",
    "SearchHit",
    "SymbolDef",
    # Context
    "ContextBudget",
    "ContextRequest",
    "ContextItem",
    "AssembledContext",
    "ContextAssembler",
    # Diff
    "EditHunk",
    "FileEdit",
    "MultiFileEdit",
    "DiffGenerator",
    "DiffApplier",
    "DiffParser",
    "ApplyResult",
    # Orchestration
    "EditTarget",
    "EditPlan",
    "MultiFileEditOrchestrator",
    # Verification
    "VerificationCheck",
    "VerificationResult",
    "EditVerifier",
    # Rollback
    "CheckpointFile",
    "Checkpoint",
    "CheckpointRollback",
    # Sandbox
    "SandboxConfig",
    "SandboxResult",
    "TerminalSandbox",
    # Agent
    "AgentAction",
    "AgentConfig",
    "AgentResult",
    "AgentTurn",
    "ComposerAgent",
]
