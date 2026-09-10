# PPO replay with gSDE and tanh squashing

This example comes from my investigation into numerical instability in PPO with
gSDE. It isolates a problem in how SB3 evaluates actions after tanh squashing.

The idea is simple: PPO samples a Gaussian value, applies `tanh`, and stores the
bounded action. During an update, SB3 reconstructs the Gaussian value from that
action. In float32, tanh can round to exactly 1 or -1, so the original value is
lost. The reconstructed value can then give a different gradient, even when the
old and new likelihoods give a ratio of exactly one.

The example runs against unmodified SB3. The proposed fix is to keep the original
sample and use it when evaluating the action during an update. I have raised
this in [SB3 issue #2285](https://github.com/DLR-RM/stable-baselines3/issues/2285).

## What the example shows

In the saturated case below, the sampled value is **12.783661**. Tanh rounds it to
**1.0**, and SB3's inverse returns **8.317766**. That changes the loss gradient
from **-3.134631** to **+14.728875**: it points in the opposite direction.

| Case | Original sample | Stored action | Reconstructed sample | SB3 loss gradient | Gradient using original sample |
|---|---:|---:|---:|---:|---:|
| Unsaturated, mean 0 | 0.783660 | 0.654803 | 0.783660 | -3.134629 | -3.134629 |
| Saturated, mean 12 | 12.783661 | 1.0 | 8.317766 | +14.728875 | -3.134631 |

Both initial ratios are exactly one. The unsaturated case gives the same gradient
to numerical precision, which provides a control for the comparison.

[reproduce.py](reproduce.py) samples an action using SB3's gSDE distribution,
records the old likelihoods, and evaluates them again with the parameters
unchanged. Each version uses its own old likelihood. The loss is PPO's clipped
surrogate with advantage +1 and clip range 0.2; the gradients above are with
respect to the Gaussian mean.

The gradient using the original sample agrees with the analytic expression
`-(u - mean) / variance`. The script also checks both gradients using finite
differences. The full output is in [result.json](result.json).

## Checking the policy methods

[check_policy.py](check_policy.py) checks the same problem through SB3's policy
methods. It creates a PPO policy with gSDE and squashing, sets its mean to a
constant, and uses the built-in Pendulum-v1 environment for the observation and
action spaces. It gets the old likelihood from `ActorCriticPolicy.forward` and
then calls `evaluate_actions` separately.

At mean 12, the gradient of the mean-head bias is **59.8440** with reconstruction
and **0.294376** with the original sample. The latter agrees with the analytic
calculation. Both ratios are one, and the mean-zero control agrees across both
methods. See [policy_result.json](policy_result.json) for the output.

This check covers the policy methods; it does not run the rollout buffer or
`PPO.train()`.

## Run it

The setup below was tested in a fresh environment on Linux x86_64, using Python
3.12.2 and CPU-only PyTorch 2.11.0. From a clone of this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
git clone https://github.com/DLR-RM/stable-baselines3.git .upstream
git -C .upstream checkout 7cfb4dd6055e74b5caa4ed4d6777492209946e26
python -m pip install -e .upstream
python -m pip check
python reproduce.py
python check_policy.py
```

If `.upstream` already exists at this revision, skip the clone. Both scripts check
that they are importing the pinned, unmodified checkout. They require Git and use
assertions, so run Python without `-O`. Each takes a few seconds once the
dependencies are installed.

[requirements.txt](requirements.txt) pins the direct dependencies.
[environment.json](environment.json) records the full package list, and
[system_info.txt](system_info.txt) records the test environment. The installation
was tested without access to the host's Python packages. GPU execution and other
operating systems have not been tested.

## What this does and does not establish

The mean of 12 is chosen deliberately to expose saturation. The examples show
that reconstructing the sample can change the PPO gradient. They do not tell us
how often this happens during training, reproduce a training crash, or establish
that retaining samples improves return or prevents every numerical failure.

The reference evaluates the original Gaussian sample. It is not a calculation of
probability mass over the rounded action's floating-point cell. Both versions use
SB3's existing Jacobian correction.

The next question is how to retain samples through rollout collection and policy
updates without breaking existing interfaces. No patch is included here.
[UPSTREAM_NOTES.md](UPSTREAM_NOTES.md) lists the relevant code and compatibility
questions; [ISSUE_DRAFT.md](ISSUE_DRAFT.md) contains the submitted report.

## License and AI assistance

This repository uses the [MIT license](LICENSE). The SB3 checkout is downloaded
separately and retains its own license.

OpenAI Codex generated the scripts and report and ran the checks. I authorized
publication, and this assistance is disclosed in the upstream issue. SB3's
contribution rules
require disclosure and do not accept fully LLM-generated PRs unless initiated by
a maintainer.
