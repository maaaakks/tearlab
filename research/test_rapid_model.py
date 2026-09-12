"""Behavioral contracts for the rapid experiment's shared representation."""
import pytest
import torch
from torch.nn import functional as F

from research.rapid_model import RapidQ


def sample_history(batch=3):
    generator = torch.Generator().manual_seed(214)
    history = torch.rand(batch, 2, 140, generator=generator)
    for start, count, width in ((4, 4, 4), (20, 32, 3), (116, 8, 3)):
        slots = history[..., start:start + count * width].reshape(batch, 2, count, width)
        slots[..., -1] = torch.randint(2, (batch, 2, count), generator=generator)
    return history


def test_shapes_and_finite_outputs_with_empty_object_pools():
    torch.manual_seed(11)
    for heads, auxiliary in ((3, True), (1, False)):
        model = RapidQ(heads=heads, auxiliary=auxiliary)
        history = torch.zeros(2, 2, 140)
        previous = torch.tensor([45, 0])
        q = model(history, previous)
        assert q.shape == (2, heads, 45)
        assert torch.isfinite(q).all()
        if auxiliary:
            prediction = model.auxiliary_predictions(history, previous, torch.tensor([[1, 2, 3], [4, 5, 6]]))
            assert prediction.shape == (2, 3, 4)
            assert torch.isfinite(prediction).all()
            assert model.auxiliary_predictions(history, previous, torch.empty(2, 0, dtype=torch.long)).shape == (2, 0, 4)
        else:
            with pytest.raises(ValueError, match='disabled'):
                model.auxiliary_predictions(history, previous, torch.zeros(2, 1, dtype=torch.long))


def test_independent_per_frame_slot_permutations_preserve_predictions():
    torch.manual_seed(12)
    model = RapidQ()
    history = sample_history()
    permuted = history.clone()
    for start, count, width in ((4, 4, 4), (20, 32, 3), (116, 8, 3)):
        for frame in range(2):
            raw = history[:, frame, start:start + count * width].reshape(3, count, width)
            permuted[:, frame, start:start + count * width] = raw[:, torch.randperm(count)].flatten(1)
    previous = torch.tensor([45, 4, 8])
    actions = torch.tensor([[1, 2], [3, 4], [5, 6]])
    torch.testing.assert_close(model(history, previous), model(permuted, previous))
    torch.testing.assert_close(model.auxiliary_predictions(history, previous, actions),
                               model.auxiliary_predictions(permuted, previous, actions))


def test_absent_slot_contents_do_not_change_q_values():
    torch.manual_seed(13)
    model = RapidQ()
    history = sample_history()
    changed = history.clone()
    for start, count, width in ((4, 4, 4), (20, 32, 3), (116, 8, 3)):
        slots = changed[..., start:start + count * width].reshape(3, 2, count, width)
        slots[..., :-1] = torch.where(slots[..., -1:] == 0, 123., slots[..., :-1])
    previous = torch.tensor([45, 4, 8])
    torch.testing.assert_close(model(history, previous), model(changed, previous))


def test_previous_command_and_future_commands_affect_their_causal_outputs():
    torch.manual_seed(14)
    model = RapidQ()
    history = sample_history()
    previous = torch.tensor([45, 4, 8])
    assert not torch.allclose(model(history, previous), model(history, torch.tensor([1, 2, 3])))
    actions = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    changed = actions.clone()
    changed[:, 1:] = 44
    original = model.auxiliary_predictions(history, previous, actions)
    altered = model.auxiliary_predictions(history, previous, changed)
    torch.testing.assert_close(original[:, 0], altered[:, 0])
    assert not torch.allclose(original[:, 1:], altered[:, 1:])


def test_q_and_auxiliary_losses_train_shared_encoder_and_their_own_heads():
    torch.manual_seed(15)
    model = RapidQ()
    history = sample_history()
    previous = torch.tensor([45, 4, 8])
    actions = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    q_loss = model(history, previous).square().mean()
    q_loss.backward()
    for parameter in (model.enemy_encoder[0].weight, model.projectile_encoder[0].weight,
                      model.trunk[0].weight, model.q_heads[0].advantage.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.dynamics.weight_hh.grad is None
    model.zero_grad(set_to_none=True)
    prediction = model.auxiliary_predictions(history, previous, actions)
    loss = prediction[..., :3].square().mean() + F.binary_cross_entropy_with_logits(
        prediction[..., 3], torch.zeros_like(prediction[..., 3]))
    loss.backward()
    for parameter in (model.enemy_encoder[0].weight, model.projectile_encoder[0].weight,
                      model.trunk[0].weight, model.dynamics.weight_ih,
                      model.dynamics.weight_hh, model.consequences.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.q_heads[0].value.weight.grad is None
