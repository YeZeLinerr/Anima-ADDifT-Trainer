import torch
from torch.utils.checkpoint import checkpoint


def test_bidirectional_slider_learns_nonzero_direction():
    """ADDifT prediction matching must learn a non-zero signed direction."""
    base_target_prediction = torch.ones(1)
    base_reference_prediction = torch.zeros(1)
    training_scale = 0.25
    lora_direction = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([lora_direction], lr=0.2)

    for _ in range(600):
        optimizer.zero_grad()
        positive_prediction = base_reference_prediction + training_scale * lora_direction
        negative_prediction = base_target_prediction - training_scale * lora_direction
        loss = 0.5 * (
            torch.nn.functional.mse_loss(positive_prediction, base_target_prediction)
            + torch.nn.functional.mse_loss(negative_prediction, base_reference_prediction)
        )
        loss.backward()
        optimizer.step()

    # Training at ±0.25 learns a full-strength reference -> target direction.
    assert torch.allclose(lora_direction.detach(), torch.full_like(lora_direction, 4.0), atol=1e-3)
    assert loss.item() < 1e-5


def test_shared_noise_preserves_pair_direction_for_rectified_flow():
    target_latent = torch.randn(2, 4, 1, 8, 8)
    reference_latent = torch.randn_like(target_latent)
    noise = torch.randn_like(target_latent)

    target_flow = noise - target_latent
    reference_flow = noise - reference_latent

    assert torch.allclose(
        target_flow - reference_flow,
        reference_latent - target_latent,
        atol=1e-6,
        rtol=1e-6,
    )


def test_paired_timestep_range_uses_ui_scale_and_shared_noise():
    target_latent = torch.ones(4, 1, 1, 1, 1)
    reference_latent = torch.zeros_like(target_latent)
    noise = torch.full_like(target_latent, 2.0)
    timesteps_ui = torch.tensor([500.0, 600.0, 800.0, 999.0])
    timesteps = timesteps_ui / 1000.0
    expanded = timesteps.view(-1, 1, 1, 1, 1)

    target_noisy = (1.0 - expanded) * target_latent + expanded * noise
    reference_noisy = (1.0 - expanded) * reference_latent + expanded * noise

    assert torch.all((timesteps >= 0.5) & (timesteps <= 1.0))
    assert torch.allclose(
        target_noisy - reference_noisy,
        (1.0 - expanded) * (target_latent - reference_latent),
        atol=1e-6,
        rtol=1e-6,
    )


def test_mask_area_normalization_preserves_small_region_loss_strength():
    loss = torch.ones(1, 4, 1, 10, 10)
    mask = torch.zeros(1, 1, 1, 10, 10)
    mask[..., :1, :1] = 1.0

    diluted = (loss * mask).mean()
    normalized_mask = mask / mask.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    normalized = (loss * normalized_mask).mean()

    assert torch.allclose(diluted, torch.tensor(0.01))
    assert torch.allclose(normalized, torch.tensor(1.0))


def test_signed_multiplier_survives_checkpoint_recompute_until_backward():
    multiplier = {"value": -1.0}
    weight = torch.nn.Parameter(torch.tensor(1.0))
    x = torch.tensor(2.0, requires_grad=True)

    def signed_layer(value):
        return value * weight * multiplier["value"]

    output = checkpoint(signed_layer, x, use_reentrant=False)
    output.backward()
    multiplier["value"] = 1.0  # equivalent to the post-backward restoration hook

    assert torch.allclose(weight.grad, torch.tensor(-2.0))
