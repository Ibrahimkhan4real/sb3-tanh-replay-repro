"""Float32 gSDE replay probe against a pinned, unmodified SB3 checkout."""

import json
import math
import subprocess
import sys
from pathlib import Path

import torch as th

PIN = "7cfb4dd6055e74b5caa4ed4d6777492209946e26"
UPSTREAM = Path(__file__).resolve().parent / ".upstream"
assert subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() == PIN
assert not subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
sys.path.insert(0, str(UPSTREAM))
import stable_baselines3 as sb3  # noqa: E402
from stable_baselines3.common.distributions import StateDependentNoiseDistribution  # noqa: E402

assert Path(sb3.__file__).resolve().is_relative_to(UPSTREAM)
th.set_num_threads(1)


def clipped_loss(log_prob: th.Tensor, old_log_prob: th.Tensor) -> th.Tensor:
    """PPO clipped surrogate loss for one sample with advantage +1."""
    ratio = (log_prob - old_log_prob).exp()
    return -th.minimum(ratio, ratio.clamp(0.8, 1.2)).mean()


def finite_difference(mean: float, sample: float, variance: float) -> float:
    """Independent scalar finite difference at fixed collected sample and old mean."""
    def loss(new_mean: float) -> float:
        ratio = math.exp(-((sample - new_mean) ** 2 - (sample - mean) ** 2) / (2 * variance))
        return -min(ratio, max(0.8, min(1.2, ratio)))

    step = 1e-5
    return (loss(mean + step) - loss(mean - step)) / (2 * step)


def probe(location: float) -> dict:
    """Collect old likelihoods, then independently evaluate the same fixed samples."""
    th.manual_seed(71)
    mean = th.tensor([[location]], dtype=th.float32, requires_grad=True)
    log_std = th.tensor([[math.log(0.5)]], dtype=th.float32)
    features = th.ones((1, 1), dtype=th.float32)
    dist = StateDependentNoiseDistribution(1, squash_output=True)
    dist.sample_weights(log_std)
    dist.proba_distribution(mean, log_std, features)
    with th.no_grad():
        # Upstream gSDE reuses its exploration matrix, so this is the actual sample.
        latent = mean + dist.get_noise(features)
        action = dist.sample()
        assert th.equal(action, th.tanh(latent))
        old_native = dist.log_prob(action)
        old_retained = (dist.distribution.log_prob(latent) - dist.bijector.log_prob_correction(latent)).sum(1)

    dist.proba_distribution(mean, log_std, features)
    native_lp = dist.log_prob(action)
    retained_lp = (dist.distribution.log_prob(latent) - dist.bijector.log_prob_correction(latent)).sum(1)
    native_grad = th.autograd.grad(clipped_loss(native_lp, old_native), mean, retain_graph=True)[0].item()
    retained_grad = th.autograd.grad(clipped_loss(retained_lp, old_retained), mean)[0].item()
    variance = dist.distribution.variance.detach().item()
    reconstructed = dist.bijector.inverse(action).item()
    analytic_grad = -(latent.item() - location) / variance
    assert math.isclose(retained_grad, analytic_grad, rel_tol=1e-5, abs_tol=1e-5)
    for sample, gradient in [(reconstructed, native_grad), (latent.item(), retained_grad)]:
        assert math.isclose(finite_difference(location, sample, variance), gradient, rel_tol=1e-5, abs_tol=1e-5)
    return dict(
        mean=location, latent=latent.item(), action=action.item(), reconstructed_latent=reconstructed,
        native_ratio=(native_lp - old_native).exp().item(),
        retained_ratio=(retained_lp - old_retained).exp().item(),
        native_loss_mean_gradient=native_grad, retained_loss_mean_gradient=retained_grad,
        analytic_loss_mean_gradient=analytic_grad,
    )


if __name__ == "__main__":
    rows = [probe(0.0), probe(12.0)]
    assert abs(rows[0]["native_loss_mean_gradient"] - rows[0]["retained_loss_mean_gradient"]) < 1e-5
    assert rows[1]["action"] == 1.0
    assert rows[1]["native_loss_mean_gradient"] > 0 > rows[1]["retained_loss_mean_gradient"]
    assert all(r["native_ratio"] == r["retained_ratio"] == 1 for r in rows)
    print(json.dumps(dict(
        upstream_commit=PIN, sb3_version=sb3.__version__, torch_version=th.__version__, python=sys.version,
        dtype="float32", device="cpu", seed=71, checks="passed", cases=rows,
    ), indent=2))
