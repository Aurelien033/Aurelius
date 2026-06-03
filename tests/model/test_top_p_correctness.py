"""Regression tests for NEW-05 + C-17 + C-18: Top-p nucleus sampling correctness.

NEW-05: Off-by-one at boundary — >= vs > means one fewer token kept at exact mass.
C-17: generate() top-p should use _apply_top_p_filter.
C-18: generate_stream() top-p should use _apply_top_p_filter.

Reference: HuggingFace TopPLogitsWarper keeps the first token to exceed top_p.
"""
from __future__ import annotations

import pytest
import torch


class TestApplyTopPFilter:
    """Test _apply_top_p_filter boundary behavior against HuggingFace reference."""

    def _huggingface_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Reference implementation matching HuggingFace TopPLogitsWarper."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def test_top_p_1_keeps_all(self):
        """top_p=1.0 should keep all tokens."""
        from src.model.transformer import _apply_top_p_filter
        logits = torch.randn(1, 100)
        result = _apply_top_p_filter(logits, 1.0)
        assert torch.equal(result, logits)

    def test_top_p_matches_huggingface_at_boundary(self):
        """At exact cumulative boundary, must keep same tokens as HuggingFace."""
        from src.model.transformer import _apply_top_p_filter
        # Construct logits where cumulative hits exactly top_p
        # 4 tokens: probs [0.60, 0.30, 0.05, 0.05]
        # We need logits whose softmax gives these probs
        logits = torch.tensor([[2.0, 1.0, -1.0, -1.0]])
        top_p = 0.9
        
        ours = _apply_top_p_filter(logits, top_p)
        ref = self._huggingface_top_p(logits, top_p)
        
        # Count how many tokens survive (not -inf)
        ours_alive = (ours != float("-inf")).sum().item()
        ref_alive = (ref != float("-inf")).sum().item()
        
        assert ours_alive == ref_alive, (
            f"Our filter keeps {ours_alive} tokens, HuggingFace keeps {ref_alive}. "
            f"Off-by-one at boundary."
        )
        torch.testing.assert_close(ours, ref)

    def test_top_p_sweep_matches_huggingface(self):
        """Sweep top_p from 0.1 to 1.0 — must match HuggingFace on every value."""
        from src.model.transformer import _apply_top_p_filter
        torch.manual_seed(42)
        logits = torch.randn(1, 50)
        
        for top_p_x10 in range(1, 11):
            top_p = top_p_x10 / 10.0
            ours = _apply_top_p_filter(logits, top_p)
            ref = self._huggingface_top_p(logits, top_p)
            
            ours_alive = (ours != float("-inf")).sum().item()
            ref_alive = (ref != float("-inf")).sum().item()
            
            assert ours_alive == ref_alive, (
                f"top_p={top_p}: our filter keeps {ours_alive} tokens, "
                f"HuggingFace keeps {ref_alive}"
            )

    def test_top_p_never_masks_all(self):
        """At least one token must always survive."""
        from src.model.transformer import _apply_top_p_filter
        torch.manual_seed(0)
        logits = torch.randn(8, 100)
        
        for top_p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = _apply_top_p_filter(logits, top_p)
            for batch_idx in range(8):
                alive = (result[batch_idx] != float("-inf")).sum().item()
                assert alive >= 1, f"top_p={top_p} batch {batch_idx}: all tokens masked"

    def test_top_p_invalid_raises(self):
        """top_p outside (0, 1] should raise ValueError (0 and negatives only; >1 is no-op)."""
        from src.model.transformer import _apply_top_p_filter
        logits = torch.randn(1, 10)
        with pytest.raises(ValueError):
            _apply_top_p_filter(logits, 0.0)
        with pytest.raises(ValueError):
            _apply_top_p_filter(logits, -0.5)
        # top_p > 1.0 is a passthrough (no filtering), not an error
        result = _apply_top_p_filter(logits, 1.5)
        torch.testing.assert_close(result, logits)

    def test_top_p_batch_consistency(self):
        """Batch result must equal per-row result."""
        from src.model.transformer import _apply_top_p_filter
        torch.manual_seed(42)
        logits = torch.randn(4, 30)
        
        batch_result = _apply_top_p_filter(logits, 0.9)
        for i in range(4):
            single = _apply_top_p_filter(logits[i:i+1], 0.9)
            torch.testing.assert_close(batch_result[i:i+1], single)
