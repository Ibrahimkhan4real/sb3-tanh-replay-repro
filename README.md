# SB3 gSDE+tanh replay reproducer

A small float32 correctness probe for discussion with Stable-Baselines3 maintainers.
This repository is separate from the research codebase and imports no research code.
No issue or PR has been submitted.

## Run

Python 3.12 was used for validation. From this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
# Skip the clone if .upstream already exists.
git clone https://github.com/DLR-RM/stable-baselines3.git .upstream
git -C .upstream checkout 7cfb4dd6055e74b5caa4ed4d6777492209946e26
python -m pip install -e .upstream
python reproduce.py
```

For the exact tested PyTorch version, install `torch==2.11.0` before installing
SB3. The committed `system_info.txt` records the validation environment; this is
not a fully locked dependency environment. The probe verifies the upstream commit,
clean checkout, and imported module location. It does not patch upstream, install
anything automatically, or launch training. Runtime was about one second on the
validation host. `result.json` is the captured output, not an input to the probe.

## Result

| Case | sampled latent | stored action | reconstructed latent | native loss gradient | retained loss gradient |
|---|---:|---:|---:|---:|---:|
| Unsaturated, mean 0 | 0.783660 | 0.654803 | 0.783660 | -3.134629 | -3.134629 |
| Saturated, mean 12 | 12.783661 | 1.0 | 8.317766 | +14.728875 | -3.134631 |

Both old/new ratios are exactly 1 in both cases. Each likelihood route uses its
own detached old log probability: there is no mixed-likelihood denominator.
The loss is the PPO clipped surrogate with advantage +1 and clip range 0.2.
The retained mean gradient agrees with the independent analytic expression
`-(u - mean) / variance`. The assertion checks include the unsaturated negative
control, saturation, ratio identity, and retained-gradient correctness.

## What this establishes

The existing `StateDependentNoiseDistribution` reconstructs the sample from its
bounded action for likelihood evaluation. In float32, a saturated tanh action
cannot identify its original latent. For an unchanged policy, native replay can
therefore have a unit ratio while differentiating at a different latent and even
reversing the mean-gradient direction relative to retained-sample replay.

This uses native distribution sampling and native `log_prob`; the retained branch
is an explicit reference. It does not run the full PPO trainer or show natural
exposure frequency. Mean 12 is deliberately selected to expose saturation. There
is no claim that this reproduces every NaN report, guarantees a training failure,
or proves improved return. The analytic reference is for the existing Gaussian
latent objective, not floating-point action-cell probability mass. The stable
Jacobian formula is not under test here: both branches use SB3's existing correction.

## Proposed contribution

See `ISSUE_DRAFT.md` for a report mapped to the upstream issue form, and
`UPSTREAM_NOTES.md` for scope and acceptance requirements. The proposed repair is
to retain sampled latents through rollout collection and minibatch likelihood
evaluation, consistently on both sides of the ratio. Buffer/API design requires
maintainer discussion; no upstream implementation is bundled here.

## AI assistance

OpenAI Codex generated this reproducer and draft, inspected the pinned upstream
source, and executed the validation. The author must review and own the report
before submission. SB3 requires disclosure and does not accept fully LLM-generated
PRs unless initiated by a maintainer. This repository is preparation for discussion,
not a claim of maintainer approval.
