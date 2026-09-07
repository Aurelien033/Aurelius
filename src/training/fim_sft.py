"""
src/training/fim_sft.py — Fill-in-the-Middle training for tool arguments and code.

Converts SFT examples to FIM (prefix + suffix -> middle) format. For tool
calls: prefix = partial call, suffix = expected output structure, target =
missing arguments. For code: standard FIM with <FIM_HOLE> markers.

Reference: "Fill-in-the-Middle for Structured Outputs" (Aurelius, 2026)
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class FIMSFTConfig:
    """Configuration for FIM-converted SFT training.

    Attributes:
        fim_ratio: Fraction of training examples to convert to FIM format.
        fim_tool_ratio: Fraction of tool-call examples to convert to FIM.
        fim_code_ratio: Fraction of code examples to convert to FIM.
        fim_prefix_token: Token or string marking FIM prefix region.
        fim_suffix_token: Token or string marking FIM suffix region.
        fim_middle_token: Token or string marking FIM middle region.
        strategy: FIM transformation strategy ('psm', 'sps', 'psm+random').
    """

    fim_ratio: float = 0.3
    fim_tool_ratio: float = 0.4
    fim_code_ratio: float = 0.3
    fim_prefix_token: str = "<FIM_PREFIX>"
    fim_suffix_token: str = "<FIM_SUFFIX>"
    fim_middle_token: str = "<FIM_MIDDLE>"
    strategy: str = "psm"  # prefix-suffix-middle


@dataclass
class FIMExample:
    """A FIM-converted training example."""

    original_prompt: str
    original_completion: str
    fim_text: str
    fim_type: str  # 'tool_fim', 'code_fim', 'standard'
    prefix_span: str = ""
    suffix_span: str = ""
    middle_span: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class FIMConverter:
    """Converts SFT examples to Fill-in-the-Middle format.

    Supports three transformation strategies:
    - PSM: prefix -> suffix -> middle (standard FIM)
    - SPS: suffix -> prefix -> middle
    - Mixed: random PSM/SPS per example
    """

    def __init__(self, config: FIMSFTConfig | None = None) -> None:
        self.config = config or FIMSFTConfig()
        self._rng = random.Random(42)

    def convert_tool_call(
        self,
        prompt: str,
        tool_call: dict[str, Any],
    ) -> FIMExample:
        """Convert a tool-call example to FIM format.

        The tool call is transformed so that the model predicts missing
        arguments given the partial call (prefix) and the expected output
        structure (suffix).

        Args:
            prompt: User/system prompt.
            tool_call: Dict with 'name' and 'arguments' keys.

        Returns:
            FIMExample with FIM-formatted training text.
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", tool_call.get("parameters", {}))

        # Split arguments: keep some visible (prefix), hide some (middle)
        if not args:
            return self._standard_fim(prompt, json.dumps(tool_call), "tool_fim")

        arg_items = list(args.items())
        n_visible = max(1, len(arg_items) // 2)
        visible = dict(arg_items[:n_visible])
        hidden = dict(arg_items[n_visible:])

        prefix_span = f'{name}({", ".join(f"{k}={v}" for k, v in visible.items())}, '
        middle_span = ", ".join(f"{k}={v}" for k, v in hidden.items())
        suffix_span = ")"

        fim_text = self._apply_strategy(
            prefix=prompt + "\n" + self.config.fim_prefix_token + prefix_span,
            suffix=self.config.fim_suffix_token + suffix_span,
            middle=self.config.fim_middle_token + middle_span,
        )

        return FIMExample(
            original_prompt=prompt,
            original_completion=json.dumps(tool_call),
            fim_text=fim_text,
            fim_type="tool_fim",
            prefix_span=prefix_span,
            suffix_span=suffix_span,
            middle_span=middle_span,
        )

    def convert_code(
        self,
        prompt: str,
        code: str,
    ) -> FIMExample:
        """Convert a code generation example to FIM format.

        Splits the code body at a random point. The model predicts the
        middle (hidden) section given prefix (code before hole) and
        suffix (code after hole).

        Args:
            prompt: User prompt describing the code task.
            code: Full code solution.

        Returns:
            FIMExample with FIM-formatted training text.
        """
        lines = code.split("\n")
        if len(lines) < 4:
            return self._standard_fim(prompt, code, "code_fim")

        # Find a reasonable split point (end of a statement)
        split_idx = self._rng.randint(1, len(lines) - 2)
        prefix = "\n".join(lines[:split_idx])
        middle = lines[split_idx]
        suffix = "\n".join(lines[split_idx + 1:])

        fim_text = self._apply_strategy(
            prefix=prompt + "\n" + self.config.fim_prefix_token + prefix,
            suffix=self.config.fim_suffix_token + suffix,
            middle=self.config.fim_middle_token + middle,
        )

        return FIMExample(
            original_prompt=prompt,
            original_completion=code,
            fim_text=fim_text,
            fim_type="code_fim",
            prefix_span=prefix,
            suffix_span=suffix,
            middle_span=middle,
            metadata={"code_language": "unknown"},
        )

    def convert_standard(
        self,
        prompt: str,
        completion: str,
    ) -> FIMExample:
        """Convert a standard text example to FIM format.

        Splits the completion at a random sentence boundary.
        """
        return self._standard_fim(prompt, completion, "standard")

    def _standard_fim(
        self,
        prompt: str,
        completion: str,
        fim_type: str,
    ) -> FIMExample:
        """Generic FIM transformation for any text."""
        sentences = completion.replace("! ", ". ").replace("? ", ". ").split(". ")
        if len(sentences) < 3:
            # Too short for FIM; return as-is
            text = prompt + "\n" + completion
            return FIMExample(
                original_prompt=prompt,
                original_completion=completion,
                fim_text=text,
                fim_type=fim_type + "_skip_too_short",
            )

        split = len(sentences) // 2
        prefix = ". ".join(sentences[:split])
        middle = sentences[split] if split < len(sentences) else ""
        suffix = ". ".join(sentences[split + 1:])

        fim_text = self._apply_strategy(
            prefix=prompt + "\n" + self.config.fim_prefix_token + prefix,
            suffix=self.config.fim_suffix_token + ("." + suffix if suffix else ""),
            middle=self.config.fim_middle_token + middle,
        )

        return FIMExample(
            original_prompt=prompt,
            original_completion=completion,
            fim_text=fim_text,
            fim_type=fim_type,
            prefix_span=prefix,
            suffix_span=suffix,
            middle_span=middle,
        )

    def _apply_strategy(
        self,
        prefix: str,
        suffix: str,
        middle: str,
    ) -> str:
        """Apply the configured FIM strategy."""
        if self.config.strategy == "psm":
            return f"{prefix}{suffix}{middle}"
        elif self.config.strategy == "sps":
            return f"{suffix}{prefix}{middle}"
        else:  # random
            if self._rng.random() < 0.5:
                return f"{prefix}{suffix}{middle}"
            return f"{suffix}{prefix}{middle}"

    def should_convert(
        self,
        example_type: str,
    ) -> bool:
        """Determine whether to convert an example based on type and ratio."""
        if example_type == "tool":
            return self._rng.random() < self.config.fim_tool_ratio
        elif example_type == "code":
            return self._rng.random() < self.config.fim_code_ratio
        else:
            return self._rng.random() < self.config.fim_ratio


class FIMMixedDataset:
    """Wraps a dataset, converting a configurable fraction to FIM format.

    Produces a mixed stream of standard + FIM examples.
    """

    def __init__(
        self,
        converter: FIMConverter,
        examples: list[dict[str, Any]],
    ) -> None:
        self.converter = converter
        self.examples = examples
        self._converted: list[FIMExample] = []
        self._standard: list[FIMExample] = []

    def prepare(self) -> list[FIMExample]:
        """Convert the configured fraction of examples to FIM format.

        Returns:
            List of FIMExample objects (mixed FIM + standard).
        """
        self._converted = []
        self._standard = []

        for ex in self.examples:
            prompt = ex.get("prompt", "")
            completion = ex.get("completion", ex.get("response", ""))
            ex_type = ex.get("type", "standard")

            if self.converter.should_convert(ex_type):
                if ex_type == "tool":
                    tool_call = ex.get("tool_call", json.loads(completion) if completion else {})
                    self._converted.append(self.converter.convert_tool_call(prompt, tool_call))
                elif ex_type == "code":
                    self._converted.append(self.converter.convert_code(prompt, completion))
                else:
                    self._converted.append(self.converter.convert_standard(prompt, completion))
            else:
                self._standard.append(FIMExample(
                    original_prompt=prompt,
                    original_completion=completion,
                    fim_text=prompt + "\n" + completion,
                    fim_type="standard",
                ))

        combined = self._converted + self._standard
        random.shuffle(combined)
        logger.info(
            "FIMMixedDataset: %d FIM + %d standard = %d total (FIM ratio=%.2f)",
            len(self._converted), len(self._standard), len(combined),
            len(self._converted) / max(len(combined), 1),
        )
        return combined


# Registry entry
FIM_REGISTRY: dict[str, type] = {
    "fim_converter": FIMConverter,
    "fim_mixed_dataset": FIMMixedDataset,
}
