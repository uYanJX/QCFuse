"""CUDA correctness test for QCFuse mean attention aggregation."""

import math
import unittest

try:
    import torch
except ImportError:  # Allow CPU-only source checks without the runtime stack.
    torch = None


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "Test requires PyTorch and CUDA",
)
class TestTritonAttentionScoreMean(unittest.TestCase):
    def test_query_head_layer_mean_matches_torch(self):
        from sglang.srt.utils.triton_attention_score import (
            compute_att_full_softmax_importance,
        )

        torch.manual_seed(7)
        layers = 2
        seq_q = 5
        seq_k = 11
        heads_q = 4
        heads_k = 2
        head_dim = 32
        target_start = 2
        target_len = 5
        q_start = 6

        q = torch.randn(
            layers,
            seq_q,
            heads_q,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )
        k = torch.randn(
            layers,
            seq_k,
            heads_k,
            head_dim,
            device="cuda",
            dtype=torch.float16,
        )

        q_heads = q.permute(0, 2, 1, 3).float()
        k_heads = (
            k.permute(0, 2, 1, 3)
            .repeat_interleave(heads_q // heads_k, dim=1)
            .float()
        )
        logits = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        logits *= 1.0 / math.sqrt(head_dim)
        q_positions = q_start + torch.arange(seq_q, device="cuda")
        k_positions = torch.arange(seq_k, device="cuda")
        causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        logits.masked_fill_(
            ~causal_mask.view(1, 1, seq_q, seq_k),
            float("-inf"),
        )
        target_probs = torch.softmax(logits, dim=-1)[
            ..., target_start : target_start + target_len
        ]
        expected = target_probs.mean(dim=(0, 1, 2))

        actual = compute_att_full_softmax_importance(
            q,
            k,
            target_start=target_start,
            target_len=target_len,
            q_start=q_start,
            causal=True,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
