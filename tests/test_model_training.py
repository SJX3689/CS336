"""CPU tests for the model and training utilities (stdlib unittest only)."""

from __future__ import annotations

import io
import unittest

import torch
from torch.nn import functional as F

from LLM_basics import (
    AdamW,
    TransformerConfig,
    TransformerLM,
    WarmupCosineScheduler,
    cosine_annealing_lr,
    cross_entropy,
    gradient_clip,
    load_checkpoint,
    save_checkpoint,
)
from main import evaluate, prepare_batch, train, train_step


def tiny_model(*, use_moe: bool = False) -> TransformerLM:
    torch.manual_seed(7)
    return TransformerLM(
        TransformerConfig(
            vocab_size=19,
            context_length=8,
            d_model=16,
            num_layers=2,
            num_heads=4,
            d_ff=32,
            dropout=0.0,
            use_moe=use_moe,
        )
    )


class ModelTests(unittest.TestCase):
    def test_forward_backward_and_causality(self) -> None:
        model = tiny_model().eval()
        tokens_a = torch.tensor([[1, 2, 3, 4]])
        tokens_b = torch.tensor([[1, 2, 8, 9]])
        logits_a = model(tokens_a)
        logits_b = model(tokens_b)

        self.assertEqual(tuple(logits_a.shape), (1, 4, 19))
        torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2])

        loss = cross_entropy(logits_a, torch.tensor([[2, 3, 4, 5]]))
        loss.backward()
        self.assertIsNotNone(model.token_embeddings.weight.grad)

    def test_model_forwards_key_padding_mask_to_every_block(self) -> None:
        model = tiny_model().eval()
        tokens_a = torch.tensor([[1, 2, 3, 4]])
        tokens_b = torch.tensor([[1, 9, 3, 4]])
        # Position 1 differs between the inputs but is unavailable as a key.
        # Consequently, later positions must be independent of that token.
        attention_mask = torch.tensor([[True, False, True, True]])
        masked_a = model(tokens_a, attention_mask=attention_mask)
        masked_b = model(tokens_b, attention_mask=attention_mask)
        torch.testing.assert_close(masked_a[:, 2:], masked_b[:, 2:])

        # Without the padding mask, causal attention can observe position 1.
        unmasked_a = model(tokens_a)
        unmasked_b = model(tokens_b)
        self.assertFalse(torch.allclose(unmasked_a[:, 2:], unmasked_b[:, 2:]))

    def test_config_and_token_position_validation(self) -> None:
        config = tiny_model().config
        with self.assertRaisesRegex(TypeError, "positional config"):
            TransformerLM(config, d_model=config.d_model)

        model = TransformerLM(config)
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(TypeError, "integer indices"):
            model(tokens, token_positions=torch.arange(3, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            model(tokens, token_positions=torch.tensor(0))
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            model(tokens, token_positions=torch.arange(3).repeat(3, 1))
        with self.assertRaisesRegex(ValueError, "negative"):
            model(tokens, token_positions=torch.tensor([0, -1, 2]))

    def test_generate_restores_training_mode(self) -> None:
        model = tiny_model().train()
        generated = model.generate(
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=3,
            temperature=0,
        )
        self.assertEqual(tuple(generated.shape), (1, 6))
        self.assertTrue(model.training)

    def test_moe_extension_is_explicitly_unimplemented(self) -> None:
        model = tiny_model(use_moe=True)
        with self.assertRaises(NotImplementedError):
            model(torch.tensor([[1, 2]]))


class LossOptimizerTests(unittest.TestCase):
    def test_cross_entropy_matches_torch(self) -> None:
        torch.manual_seed(3)
        logits = torch.randn(2, 3, 5, requires_grad=True)
        labels = torch.randint(0, 5, (2, 3))
        actual = cross_entropy(logits, labels)
        expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))
        torch.testing.assert_close(actual, expected)

    def test_adamw_matches_torch_for_dense_parameters(self) -> None:
        ours = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        reference = torch.nn.Parameter(ours.detach().clone())
        ours_optimizer = AdamW([ours], lr=1e-2, weight_decay=0.1)
        reference_optimizer = torch.optim.AdamW(
            [reference], lr=1e-2, weight_decay=0.1
        )
        for gradient in (torch.tensor([0.3, -0.4]), torch.tensor([0.1, 0.2])):
            ours.grad = gradient.clone()
            reference.grad = gradient.clone()
            ours_optimizer.step()
            reference_optimizer.step()
        torch.testing.assert_close(ours, reference)

    def test_adamw_rejects_invalid_eps_and_complex_parameters(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        for eps in (0.0, -1e-8, float("nan")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "positive"):
                AdamW([parameter], eps=eps)

        complex_parameter = torch.nn.Parameter(
            torch.ones(1, dtype=torch.complex64)
        )
        with self.assertRaisesRegex(TypeError, "complex parameters"):
            AdamW([complex_parameter])

    def test_gradient_clip_and_schedule_boundaries(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        norm = gradient_clip([parameter], 1.0)
        torch.testing.assert_close(norm, torch.tensor(5.0))
        self.assertLessEqual(float(parameter.grad.norm()), 1.000001)

        self.assertEqual(cosine_annealing_lr(0, 1.0, 0.1, 2, 10), 0.0)
        self.assertEqual(cosine_annealing_lr(2, 1.0, 0.1, 2, 10), 1.0)
        self.assertEqual(cosine_annealing_lr(10, 1.0, 0.1, 2, 10), 0.1)
        self.assertEqual(cosine_annealing_lr(11, 1.0, 0.1, 2, 10), 0.1)

    def test_checkpoint_round_trip(self) -> None:
        model = torch.nn.Linear(3, 2)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=1, total_steps=4, min_lr=1e-5
        )
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        checkpoint = io.BytesIO()
        save_checkpoint(model, optimizer, 12, checkpoint, scheduler=scheduler)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
        checkpoint.seek(0)
        iteration = load_checkpoint(
            checkpoint, model, optimizer, scheduler=scheduler
        )
        self.assertEqual(iteration, 12)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name])


class TrainingLoopTests(unittest.TestCase):
    def test_batch_shift_step_train_and_evaluate(self) -> None:
        model = tiny_model()
        tokens = torch.randint(0, model.vocab_size, (2, 6))
        inputs, labels = prepare_batch(tokens)
        self.assertEqual(tuple(inputs.shape), (2, 5))
        torch.testing.assert_close(inputs[:, 1:], labels[:, :-1])

        optimizer = AdamW(model.parameters(), lr=1e-3)
        before = model.token_embeddings.weight.detach().clone()
        loss = train_step(model, tokens, optimizer, max_grad_norm=1.0)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.token_embeddings.weight))

        state = train(model, [tokens, tokens], optimizer, max_steps=2)
        self.assertEqual(state.step, 2)
        self.assertIsNotNone(state.last_loss)
        validation_loss = evaluate(model, [tokens])
        self.assertGreater(validation_loss, 0.0)

    def test_empty_and_exhausted_iterables_do_not_inflate_epoch(self) -> None:
        model = tiny_model()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        empty_state = train(model, [], optimizer, epochs=3)
        self.assertEqual((empty_state.step, empty_state.epoch), (0, 0))

        tokens = torch.randint(0, model.vocab_size, (2, 4))
        one_shot_batches = iter([tokens])
        one_shot_state = train(model, one_shot_batches, optimizer, epochs=3)
        self.assertEqual((one_shot_state.step, one_shot_state.epoch), (1, 1))

    def test_explicit_device_must_match_model_before_training(self) -> None:
        model = tiny_model()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, model.vocab_size, (2, 4))
        first_parameter = next(model.parameters())

        with self.assertRaisesRegex(ValueError, "does not match model device"):
            train_step(model, tokens, optimizer, device="meta")
        self.assertEqual(first_parameter.device.type, "cpu")

        # The higher-level loop must not move a model behind an already-created
        # optimizer's back either; it delegates the same explicit check.
        with self.assertRaisesRegex(ValueError, "does not match model device"):
            train(model, [tokens], optimizer, device="meta")
        self.assertIs(optimizer.param_groups[0]["params"][0], first_parameter)
        self.assertEqual(first_parameter.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
