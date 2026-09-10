"""Deterministic float32 probe of upstream gSDE squashed-action replay."""
import json
import math
from pathlib import Path
import subprocess
import sys

import torch as th

PIN = "7cfb4dd6055e74b5caa4ed4d6777492209946e26"
ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / ".upstream"
assert subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() == PIN
assert not subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
sys.path.insert(0, str(UPSTREAM))
import stable_baselines3 as sb3
from stable_baselines3.common.distributions import StateDependentNoiseDistribution

assert Path(sb3.__file__).resolve().is_relative_to(UPSTREAM)
th.set_num_threads(1)


def probe(location):
    """Compare native and retained-sample PPO gradients at unchanged weights."""
    th.manual_seed(71)
    mean = th.tensor([[location]], dtype=th.float32, requires_grad=True)
    log_std = th.tensor([[math.log(0.5)]], dtype=th.float32)
    features = th.ones((1, 1), dtype=th.float32)
    dist = StateDependentNoiseDistribution(1, squash_output=True)
    dist.sample_weights(log_std)
    dist.proba_distribution(mean, log_std, features)
    # Exactly the latent used by upstream sample(), without changing its code.
    with th.no_grad():
        latent = mean + dist.get_noise(features)
        action = dist.sample()
        assert th.equal(action, th.tanh(latent))
    native_lp = dist.log_prob(action)
    # Hold the collected latent fixed. The transform term is parameter independent.
    retained_lp = (dist.distribution.log_prob(latent)
                   - dist.bijector.log_prob_correction(latent)).sum(dim=1)
    # Each route has its OWN rollout denominator. No hybrid ratio is used.
    native_ratio = (native_lp - native_lp.detach()).exp()
    retained_ratio = (retained_lp - retained_lp.detach()).exp()
    # PPO with advantage +1 and clip_range .2, inside the clipping interval.
    def loss(ratio):
        return -th.minimum(ratio, ratio.clamp(0.8, 1.2)).mean()
    native_grad = th.autograd.grad(loss(native_ratio), mean, retain_graph=True)[0].item()
    retained_grad = th.autograd.grad(loss(retained_ratio), mean)[0].item()
    analytic_grad = (-(latent - mean.detach()) / dist.distribution.variance.detach()).item()
    assert math.isclose(retained_grad, analytic_grad, rel_tol=1e-5, abs_tol=1e-5)
    return dict(mean=location, latent=latent.item(), action=action.item(),
                reconstructed_latent=dist.bijector.inverse(action).item(),
                native_ratio=native_ratio.item(), retained_ratio=retained_ratio.item(),
                native_loss_mean_gradient=native_grad,
                retained_loss_mean_gradient=retained_grad,
                analytic_loss_mean_gradient=analytic_grad)


if __name__ == "__main__":
    rows = [probe(0.0), probe(12.0)]
    assert abs(rows[0]["native_loss_mean_gradient"] - rows[0]["retained_loss_mean_gradient"]) < 1e-5
    assert rows[1]["action"] == 1.0
    assert abs(rows[1]["native_loss_mean_gradient"] - rows[1]["retained_loss_mean_gradient"]) > 1
    assert all(r["native_ratio"] == r["retained_ratio"] == 1 for r in rows)
    print(json.dumps(dict(upstream_commit=PIN, sb3_version=sb3.__version__,
                         torch_version=th.__version__, python=sys.version,
                         dtype="float32", device="cpu", seed=71,
                         checks="passed", cases=rows), indent=2))
