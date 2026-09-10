# Draft: [Bug] gSDE+tanh action reconstruction can reverse PPO replay gradients despite a unit initial ratio

Status: prepared for author review; not submitted. No maintainer endorsement.

## Bug

With `use_sde=True` and `squash_output=True`, float32 tanh saturation can make
`StateDependentNoiseDistribution.log_prob(action)` evaluate the Gaussian density
at a reconstructed latent different from the sample that produced the action.
In the example below, the PPO surrogate mean gradient reverses sign relative to
the retained-sample reference, even though both routes have initial ratio 1.

The native collection and replay likelihoods are mutually consistent at the
reconstructed sample. The discrepancy is relative to the original sampled latent;
this report is not claiming that native rollout and replay log probabilities
necessarily disagree before an update.

## To reproduce

No custom environment, training run, checkpoint, or research fork is required.
The mean is deliberately set to 12 to expose saturation; this is a deterministic
correctness example, not evidence of typical training prevalence.

Create an empty directory, save the code below there as `reproduce.py`, then run:

```bash
git clone https://github.com/DLR-RM/stable-baselines3.git .upstream
git -C .upstream checkout 7cfb4dd6055e74b5caa4ed4d6777492209946e26
python -m pip install -e .upstream
python reproduce.py
```

Use a virtual environment for dependency installation. Validation used Python
3.12.2 and PyTorch 2.11.0+cpu. The script checks its import source and clean pin.

```python
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
```

## Observed result

| mean | sampled u | stored action | inverse(action) | native loss mean gradient | retained reference |
|---|---:|---:|---:|---:|---:|
| 0 | 0.783660 | 0.654803 | 0.783660 | -3.134629 | -3.134629 |
| 12 | 12.783661 | 1.0 | 8.317766 | +14.728875 | -3.134631 |

Both ratios are exactly 1. Each route uses its own old likelihood. The retained
result matches the analytic gradient `-(u-mean)/variance`. The control agrees
within 1e-5. No exception or NaN is expected in this small example.

## Expected behavior and proposed direction

For the sampled-latent PPO objective, score the same retained pre-tanh sample
throughout collection and optimisation. Changing inverse-tanh epsilon cannot
recover information lost from the bounded action. A unit-ratio check alone does
not test this gradient property.

Would maintainers consider retaining the original gSDE sample through the rollout
buffer and exposing it to update-time likelihood evaluation? I would appreciate
feedback on the preferred buffer/policy API before proposing implementation.
The numerator and denominator must both use the retained representation.

This reference does not model probability mass over rounded action cells, does
not change sampling, and does not establish a universal stability or return gain.
The proposed contribution would be limited to replay correctness; research
interventions and hyperparameter changes would remain separate.

Related context: #1593 reports PPO+gSDE NaNs, but this script does not establish
that its reported failures share this cause. #2249 changes non-gSDE Gaussian
means and appears to address a different concern.

## System info

Imported SB3 source: pinned upstream checkout, not a patched installed package.

```text
- OS: Linux-6.8.0-134-generic-x86_64-with-glibc2.39 # 134-Ubuntu SMP PREEMPT_DYNAMIC Fri Jun 26 18:43:11 UTC 2026
- Python: 3.12.2
- Stable-Baselines3: 2.9.2a0
- PyTorch: 2.11.0+cpu
- GPU Enabled: False
- Numpy: 1.26.4
- Cloudpickle: 3.1.0
- Gymnasium: 1.2.0
- OpenAI Gym: 0.26.2

```

## AI assistance disclosure

OpenAI Codex generated the reproducer and this draft and ran it against the
unmodified upstream checkout. This draft requires the author's technical review
before submission. No fully generated PR is being proposed, and no maintainer
has initiated or approved a patch.

## Author checklist before submission

- [ ] Personally review and rerun the reproducer; verify the reference and scope.
- [ ] Recheck related issues and read the relevant SB3 documentation.
- [ ] Complete the upstream issue form truthfully, including assistant disclosure.

The executable example uses no custom Gym environment and the code is included
above. Leave unchecked requirements that the author has not yet completed.
