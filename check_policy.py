"""Verify separate native ActorCriticPolicy.forward/evaluate_actions calls."""

import json
import math

from reproduce import PIN, clipped_loss, th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


def check_policy(location: float) -> dict:
    """Check a deliberately configured policy, without training or changing SB3."""
    model = PPO(
        "MlpPolicy", "Pendulum-v1", use_sde=True,
        policy_kwargs=dict(squash_output=True, log_std_init=-2.0),
        n_steps=2, batch_size=2, seed=71, device="cpu",
    )
    try:
        policy = model.policy
        with th.no_grad():
            policy.action_net.weight.zero_()
            policy.action_net.bias.fill_(location)
        obs = obs_as_tensor(model.env.reset(), policy.device)
        policy.reset_noise(1)
        with th.no_grad():
            dist = policy.get_distribution(obs)
            latent = dist.distribution.mean + dist.get_noise(dist._latent_sde)
            action, _, old_native = policy(obs)
            assert th.equal(th.tanh(latent), action)
            old_retained = (dist.distribution.log_prob(latent) - dist.bijector.log_prob_correction(latent)).sum(1)
        _, native_lp, _ = policy.evaluate_actions(obs, action)
        dist = policy.get_distribution(obs)
        retained_lp = (dist.distribution.log_prob(latent) - dist.bijector.log_prob_correction(latent)).sum(1)
        native_grad = th.autograd.grad(clipped_loss(native_lp, old_native), policy.action_net.bias)[0].item()
        retained_grad = th.autograd.grad(clipped_loss(retained_lp, old_retained), policy.action_net.bias)[0].item()
        analytic = (-(latent - dist.distribution.mean.detach()) / dist.distribution.variance.detach()).item()
        assert math.isclose(retained_grad, analytic, abs_tol=1e-5, rel_tol=1e-5)
        native_ratio = (native_lp - old_native).exp().item()
        retained_ratio = (retained_lp - old_retained).exp().item()
        assert native_ratio == retained_ratio == 1.0
        if location == 0.0:
            assert math.isclose(native_grad, retained_grad, abs_tol=1e-5, rel_tol=1e-5)
        else:
            assert action.item() == 1.0
            assert abs(native_grad - retained_grad) > 1.0
        return dict(
            mean=location, latent=latent.item(), action=action.item(),
            native_ratio=native_ratio, retained_ratio=retained_ratio,
            native_bias_gradient=native_grad, retained_bias_gradient=retained_grad,
            analytic_bias_gradient=analytic,
        )
    finally:
        model.env.close()


if __name__ == "__main__":
    print(json.dumps(dict(upstream_commit=PIN, checks="passed", cases=[check_policy(0.0), check_policy(12.0)]), indent=2))
