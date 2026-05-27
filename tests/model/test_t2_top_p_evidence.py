"""
Evidence test for T2: Top-p Nucleus Sampling Correctness (C-17, C-18, NEW-05)

Proves:
1. _apply_top_p_filter matches HuggingFace TopPLogitsWarper behavior.
2. generate() and generate_stream() source calls _apply_top_p_filter (no inline duplication).
3. Old inline cumulative_probs pattern is absent from both methods.
"""
from __future__ import annotations

import inspect
import re

import torch
import pytest

from src.model.transformer import _apply_top_p_filter, AureliusTransformer


def _huggingface_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """HuggingFace TopPLogitsWarper reference implementation."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


class TestTopPFilterCorrectness:
    """Verify _apply_top_p_filter matches HuggingFace behavior."""

    def test_top_p_boundary_behavior(self):
        """At cumulative boundary, should keep tokens where cumprob <= top_p."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)
        top_p = 0.9

        ours = _apply_top_p_filter(logits, top_p)
        ref = _huggingface_top_p(logits, top_p)

        ours_count = (ours != float("-inf")).sum().item()
        ref_count = (ref != float("-inf")).sum().item()

        assert ours_count == ref_count, (
            f"Our filter kept {ours_count} tokens, HuggingFace kept {ref_count}"
        )

    def test_top_p_sweep(self):
        """Sweep top_p from 0.1 to 1.0, verify token counts match."""
        torch.manual_seed(42)
        logits = torch.randn(1, 100)

        for top_p in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            ours = _apply_top_p_filter(logits, top_p)
            ref = _huggingface_top_p(logits, top_p)

            ours_count = (ours != float("-inf")).sum().item()
            ref_count = (ref != float("-inf")).sum().item()

            assert ours_count == ref_count, (
                f"top_p={top_p}: our filter kept {ours_count} tokens, "
                f"HuggingFace kept {ref_count}"
            )

    def test_top_p_never_masks_all(self):
        """At least one token must always survive."""
        torch.manual_seed(0)
        logits = torch.randn(8, 100)

        for top_p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = _apply_top_p_filter(logits, top_p)
            for batch_idx in range(8):
                alive = (result[batch_idx] != float("-inf")).sum().item()
                assert alive >= 1, f"top_p={top_p} batch {batch_idx}: all tokens masked"

    def test_top_p_1_keeps_all(self):
        """top_p=1.0 should keep all tokens (no filtering)."""
        logits = torch.randn(1, 50)
        result = _apply_top_p_filter(logits, 1.0)
        torch.testing.assert_close(result, logits)

    def test_top_p_invalid_values(self):
        """top_p=0 or negative should raise; top_p>1 is passthrough."""
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError):
            _apply_top_p_filter(logits, 0.0)
        with pytest.raises(ValueError):
            _apply_top_p_filter(logits, -0.5)
        # >1.0 is passthrough
        result = _apply_top_p_filter(logits, 1.5)
        torch.testing.assert_close(result, logits)


class TestGenerateUsesSharedFilter:
    """Static source analysis: generate() and generate_stream() must call _apply_top_p_filter."""

    OLD_INLINE_PATTERN = re.compile(r"cumulative_probs\s*<=\s*\(1\.0\s*-\s*top_p\)")
    SHARED_CALL_PATTERN = re.compile(r"_apply_top_p_filter")

    def test_generate_no_inline_top_p(self):
        """generate() must not contain the old inline cumulative_probs pattern."""
        source = inspect.getsource(AureliusTransformer.generate)
        assert not self.OLD_INLINE_PATTERN.search(source), (
            "generate() still contains inline top-p logic (cumulative_probs <= (1.0 - top_p)). "
            "Replace with call to _apply_top_p_filter()."
        )

    def test_generate_calls_shared_filter(self):
        """generate() must call _apply_top_p_filter."""
        source = inspect.getsource(AureliusTransformer.generate)
        assert self.SHARED_CALL_PATTERN.search(source), (
            "generate() does not call _apply_top_p_filter(). "
            "Both generate() and generate_stream() must share the same implementation."
        )

    def test_generate_stream_no_inline_top_p(self):
        """generate_stream() must not contain the old inline cumulative_probs pattern."""
        source = inspect.getsource(AureliusTransformer.generate_stream)
        assert not self.OLD_INLINE_PATTERN.search(source), (
            "generate_stream() still contains inline top-p logic. "
            "Replace with call to _apply_top_p_filter()."
        )

    def test_generate_stream_calls_shared_filter(self):
        """generate_stream() must call _apply_top_p_filter."""
        source = inspect.getsource(AureliusTransformer.generate_stream)
        assert self.SHARED_CALL_PATTERN.search(source), (
            "generate_stream() does not call _apply_top_p_filter(). "
            "Both generate() and generate_stream() must share the same implementation."
        )
