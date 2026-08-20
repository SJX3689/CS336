"""Unit tests for the core Transformer modules.

The file uses only ``unittest`` so it can run in the lightweight project
environment, while remaining discoverable by pytest if pytest is installed.
"""

import unittest

import torch
from torch import nn

from LLM_basics import RoPEEmbedding as TopLevelRoPEEmbedding
from LLM_basics.Modules import (
    FFN,
    FeedForward,
    Linear,
    MoE,
    MultiHeadAttention,
    RMSNorm,
    RoPEEmbedding,
    SwiGLU,
    scaled_dot_product_attention,
    stable_softmax,
)
from LLM_basics.Modules.linear import Linear as LowercaseLinear
from LLM_basics.Modules.rope import RoPEEmbedding as ModuleRoPEEmbedding


class LinearAndNormalizationTests(unittest.TestCase):
    def test_linear_is_pytorch_linear(self) -> None:
        self.assertIs(Linear, nn.Linear)
        self.assertIs(LowercaseLinear, nn.Linear)
        layer = Linear(3, 5)
        self.assertIsInstance(layer, nn.Linear)

    def test_rms_norm_matches_reference_and_backpropagates(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        norm = RMSNorm(4, eps=1e-6, dtype=torch.float64)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))

        actual = norm(x)
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected * norm.weight
        torch.testing.assert_close(actual, expected)

        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rms_norm_rejects_wrong_feature_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "last dimension"):
            RMSNorm(4)(torch.randn(2, 3))


class FeedForwardTests(unittest.TestCase):
    def test_swiglu_shape_aliases_and_gradients(self) -> None:
        self.assertIs(FFN, FeedForward)
        self.assertIs(SwiGLU, FeedForward)
        layer = FeedForward(8, 13, dropout=0.0)
        self.assertIsInstance(layer.up, nn.Linear)
        self.assertIsInstance(layer.gate, nn.Linear)
        self.assertIsInstance(layer.down, nn.Linear)

        x = torch.randn(2, 5, 8, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        output.sum().backward()
        self.assertIsNotNone(x.grad)


class AttentionPrimitiveTests(unittest.TestCase):
    def test_stable_softmax_handles_large_and_fully_masked_rows(self) -> None:
        logits = torch.tensor([[10_000.0, 9_999.0], [-torch.inf, -torch.inf]])
        probabilities = stable_softmax(logits)
        torch.testing.assert_close(
            probabilities[0], torch.softmax(logits[0], dim=-1)
        )
        torch.testing.assert_close(probabilities[1], torch.zeros(2))
        torch.testing.assert_close(
            stable_softmax(torch.tensor([torch.inf, 0.0])),
            torch.tensor([1.0, 0.0]),
        )

    def test_causal_attention_and_allow_masks(self) -> None:
        query = torch.ones(2, 1)
        key = torch.ones(2, 1)
        value = torch.tensor([[1.0], [3.0]])
        causal = scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        torch.testing.assert_close(causal, torch.tensor([[1.0], [2.0]]))

        allow_mask = torch.tensor([[True, False], [False, False]])
        masked = scaled_dot_product_attention(query, key, value, allow_mask)
        torch.testing.assert_close(masked, torch.tensor([[1.0], [0.0]]))

        integer_allow_mask = torch.tensor([[1, 0], [1, 1]])
        masked = scaled_dot_product_attention(query, key, value, integer_allow_mask)
        torch.testing.assert_close(masked, torch.tensor([[1.0], [2.0]]))

    def test_additive_float_mask(self) -> None:
        query = torch.ones(1, 1)
        key = torch.ones(2, 1)
        value = torch.tensor([[2.0], [10.0]])
        mask = torch.tensor([[0.0, -torch.inf]])
        result = scaled_dot_product_attention(query, key, value, mask)
        torch.testing.assert_close(result, torch.tensor([[2.0]]))

    def test_every_float_mask_is_additive_including_all_zeros(self) -> None:
        query = torch.ones(2, 1)
        key = torch.ones(2, 1)
        value = torch.tensor([[1.0], [3.0]])

        unmasked = scaled_dot_product_attention(query, key, value)
        zero_masked = scaled_dot_product_attention(
            query, key, value, torch.zeros(2, 2)
        )
        torch.testing.assert_close(zero_masked, unmasked)

        # Even a 0/1-valued float tensor is additive; dtype, not values,
        # determines mask semantics.
        binary_valued_float = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        result = scaled_dot_product_attention(
            query, key, value, binary_valued_float
        )
        expected_first = torch.softmax(torch.tensor([1.0, 0.0]), dim=0) @ value[:, 0]
        torch.testing.assert_close(result[0, 0], expected_first)
        torch.testing.assert_close(result[1], unmasked[1])

    def test_invalid_shapes_have_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence lengths"):
            scaled_dot_product_attention(
                torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 4)
            )


class MultiHeadAttentionTests(unittest.TestCase):
    def test_public_rope_api_is_used_by_attention(self) -> None:
        self.assertIs(ModuleRoPEEmbedding, RoPEEmbedding)
        self.assertIs(TopLevelRoPEEmbedding, RoPEEmbedding)

        rope = RoPEEmbedding(
            theta=10_000.0,
            d_k=4,
            max_seq_len=2,
            device="cpu",
        )
        x = torch.tensor(
            [[[1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]]]
        )
        positions = torch.tensor([0, 1, 4])
        rotated = rope(x, positions)

        torch.testing.assert_close(rotated[:, 0], x[:, 0])
        torch.testing.assert_close(
            rotated.unflatten(-1, (2, 2)).square().sum(-1),
            x.unflatten(-1, (2, 2)).square().sum(-1),
        )
        self.assertGreaterEqual(rope.cached_sequence_length, 5)

        attention = MultiHeadAttention(8, 2, use_rope=True)
        self.assertIsInstance(attention.rope, RoPEEmbedding)

    def test_output_is_causal_and_accepts_padding_mask(self) -> None:
        torch.manual_seed(1)
        layer = MultiHeadAttention(8, 2, bias=False).eval()
        x = torch.randn(2, 5, 8)
        changed_future = x.clone()
        changed_future[:, 3:] += 100.0

        original = layer(x)
        changed = layer(changed_future)
        torch.testing.assert_close(original[:, :3], changed[:, :3])

        # A fully masked batch item produces zero attention (and, with no
        # projection bias, therefore a zero layer output).
        padding_mask = torch.tensor(
            [[False, False, False, False, False], [True, True, True, True, True]]
        )
        masked = layer(x, mask=padding_mask)
        torch.testing.assert_close(masked[0], torch.zeros_like(masked[0]))

    def test_cached_decode_matches_full_decode_and_extends_rope_cache(self) -> None:
        torch.manual_seed(2)
        layer = MultiHeadAttention(
            8, 2, use_rope=True, max_seq_len=2, dropout=0.0, bias=False
        ).eval()
        x = torch.randn(2, 6, 8)

        full_output = layer(x)
        first_output, cache = layer(x[:, :4], use_cache=True)
        second_output, cache = layer(
            x[:, 4:], past_key_value=cache, use_cache=True
        )
        cached_output = torch.cat((first_output, second_output), dim=1)

        torch.testing.assert_close(full_output, cached_output, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache[0].shape, (2, 2, 6, 4))
        self.assertGreaterEqual(layer.rope.cached_sequence_length, 6)

    def test_validation_rejects_bad_model_and_cache_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            MultiHeadAttention(7, 2)

        layer = MultiHeadAttention(8, 2)
        bad_cache = (torch.randn(1, 3, 2, 4), torch.randn(1, 3, 2, 4))
        with self.assertRaisesRegex(ValueError, "cached key"):
            layer(torch.randn(1, 1, 8), past_key_value=bad_cache)


class ExtensionPointTests(unittest.TestCase):
    def test_moe_cannot_be_silently_used(self) -> None:
        layer = MoE(8, num_experts=4, top_k=2)
        self.assertFalse(layer.is_implemented)
        self.assertEqual(layer.config["num_experts"], 4)
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            layer(torch.randn(2, 3, 8))


if __name__ == "__main__":
    unittest.main()
