## Bug

With `use_sde=True` and `squash_output=True`, float32 tanh saturation can make
gSDE likelihood evaluation score a different Gaussian sample from the one that
produced the action. A small example gives opposite PPO surrogate mean gradients
for reconstruction and retained-sample evaluation, although both initial ratios
are one.

This is a deliberately saturated correctness example, not a training-crash
report. It uses unmodified SB3 at commit
`7cfb4dd6055e74b5caa4ed4d6777492209946e26` (2.9.2a0).

## To reproduce

The [reproducer repository](https://github.com/Ibrahimkhan4real/sb3-tanh-replay-repro)
contains [the distribution example](https://github.com/Ibrahimkhan4real/sb3-tanh-replay-repro/blob/main/reproduce.py)
and [a native policy check](https://github.com/Ibrahimkhan4real/sb3-tanh-replay-repro/blob/main/check_policy.py).
On Linux with Python 3.12:

```bash
git clone https://github.com/Ibrahimkhan4real/sb3-tanh-replay-repro.git
cd sb3-tanh-replay-repro
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

The old likelihoods are collected under `no_grad` and independently reevaluated
at unchanged weights. Each route uses its own old likelihood. Both scripts
include an unsaturated control. The distribution example also checks the gradients
against scalar finite differences and the analytic retained-sample mean gradient.

## Relevant output

For advantage +1 and clip range 0.2:

| mean | sampled latent | stored action | reconstructed latent | native loss mean gradient | retained reference |
|---|---:|---:|---:|---:|---:|
| 0 | 0.783660 | 0.654803 | 0.783660 | -3.134629 | -3.134629 |
| 12 | 12.783661 | 1.0 | 8.317766 | +14.728875 | -3.134631 |

Both ratios equal one. The retained result matches `-(u - mean) / variance`.
No NaN or exception is expected in this example.

The second script obtains the old native likelihood from
`ActorCriticPolicy.forward` and the new one from `evaluate_actions` on a PPO
policy with a deliberately constant mean head. At mean 12, the native bias
gradient is 59.8440 versus 0.294376 for the retained reference and analytic result.
The unsaturated control agrees. This uses built-in Pendulum-v1 for its spaces;
it does not run the rollout buffer, `PPO.train()`, or a custom environment.

## Expected behavior / design question

For the sampled-latent PPO objective, the update should score the same pre-tanh
sample that was collected. The native old and new likelihoods are consistent at
the reconstructed sample; the discrepancy is relative to the original sample.
A unit initial ratio therefore does not detect this difference.

Would you consider retaining the sampled latent through collection and replay?
Which buffer/policy interface would you prefer? Both numerator and denominator
would need to use the retained representation. No patch is proposed yet.

These checks do not establish natural exposure frequency, a general stability
guarantee, or a return improvement. The reference is not a rounded-action-cell
mass objective. Existing issue #1593 is related NaN context, but the example does
not establish the cause of that report. PR #2249 addresses a different mean-bound
change.

## System info

Validated in a fresh venv with no system site-packages: Linux x86_64, Python
3.12.2, PyTorch 2.11.0+cpu, NumPy 1.26.4, Gymnasium 1.2.0, Cloudpickle 3.1.0,
SB3 2.9.2a0 installed editable from the clean pinned checkout. No GPU or OpenAI
Gym was used. Full versions and captured outputs are in the repository.

## Checklist and assistance disclosure

The checks below were performed with OpenAI Codex. Codex generated the scripts
and this report, inspected the relevant upstream code/documentation, and ran the
validation. The repository owner authorized publication. No independent human
re-execution or maintainer approval is claimed. This is a request for discussion,
not a fully generated PR.

- [x] This report does not concern a custom Gym environment.
- [x] Searched related issues; no matching retained-latent repair was found. Related reports are identified above.
- [x] Reviewed the relevant PPO documentation, contribution guide, and implementation.
- [x] Provided executable examples verified against the pinned upstream source.
- [x] Used fenced code blocks for reproduction commands.
