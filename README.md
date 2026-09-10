# SB3 gSDE+tanh replay reproducer

A float32 correctness example against unmodified Stable-Baselines3. Under
saturation, reconstructing the Gaussian sample from a bounded action can change
the PPO surrogate gradient, even when the initial likelihood ratio is one.

Upstream discussion: [SB3 issue #2285](https://github.com/DLR-RM/stable-baselines3/issues/2285).

## Run

Tested on Linux x86_64 with Python 3.12.2 and CPU-only PyTorch 2.11.0. From a
fresh clone of this repository:

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

Skip cloning `.upstream` if it already exists at the indicated revision. The
scripts require Git, verify the clean upstream checkout and import location, and
never patch SB3 or run training. They use assertions, so run without Python's
`-O` flag. Each finishes in a few seconds after dependencies are installed.

Direct dependencies are pinned in `requirements.txt`; `environment.json` records
all resolved package versions. `system_info.txt` describes the isolated validation
environment. The documented installation was tested without access to the host's
site-packages. GPU and other operating systems have not been validated.

## Distribution example

`reproduce.py` obtains an action from native gSDE sampling, collects the old
likelihoods under `no_grad`, and separately reevaluates each fixed sample.
Each route uses its own old likelihood, so no hybrid denominator is involved.

| Case | sampled latent | stored action | reconstructed latent | native loss gradient | retained loss gradient |
|---|---:|---:|---:|---:|---:|
| Unsaturated, mean 0 | 0.783660 | 0.654803 | 0.783660 | -3.134629 | -3.134629 |
| Saturated, mean 12 | 12.783661 | 1.0 | 8.317766 | +14.728875 | -3.134631 |

Both initial ratios are exactly one. Gradients are with respect to the Gaussian
mean for a PPO clipped surrogate with advantage +1 and clip range 0.2.
The retained gradient matches `-(u - mean) / variance`. Independent scalar finite
differences check both native and retained gradients. Saved output: `result.json`.

## Native policy check

`check_policy.py` creates an upstream PPO policy with gSDE and squashing, using
Gymnasium's built-in Pendulum-v1 for its observation/action spaces. It deliberately
sets the mean head to a constant, collects the old native likelihood through
`ActorCriticPolicy.forward`, then calls `evaluate_actions` independently.

At mean 12, the native mean-head bias gradient is **59.8440**, versus **0.294376**
for the retained reference and analytic calculation. Both ratios are one.
At mean zero, the gradients agree. Saved output: `policy_result.json`.
This check exercises policy methods, not the rollout buffer or `PPO.train()`.

## Scope

Mean 12 is deliberately selected to expose float32 saturation. These examples
establish a gradient discrepancy relative to the original sampled-latent objective.
They do not establish natural exposure frequency, a training crash, improved
return, or a universal stability guarantee. The retained reference is not the
probability mass of a rounded action cell. Both branches use SB3's existing
Jacobian correction; this repository does not propose a Jacobian change.

The proposed discussion is whether sampled latents should be retained through
collection and update-time likelihood evaluation. Buffer/API design should be
agreed with maintainers before a patch. See `ISSUE_DRAFT.md` for the proposed
report and `UPSTREAM_NOTES.md` for source pointers and compatibility concerns.

## License and assistance

This repository is MIT licensed; see `LICENSE`. The separately downloaded SB3
checkout retains its own license and is excluded from this repository.

OpenAI Codex generated the scripts and report and executed the checks. Publication
was authorized by the repository owner. This is not a claim of independent human
re-execution or maintainer approval. SB3 requires public disclosure of assistant
use and does not accept fully LLM-generated PRs unless initiated by a maintainer.
No upstream implementation patch is included.
