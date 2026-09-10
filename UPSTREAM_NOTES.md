# Upstream assessment (2026-09-10)

Pinned upstream: `7cfb4dd6055e74b5caa4ed4d6777492209946e26`, September 9, 2026.

- `stable_baselines3/common/distributions.py`: gSDE `sample` constructs a Gaussian
  sample then squashes it; `log_prob` reconstructs through `bijector.inverse`.
- `stable_baselines3/common/policies.py`: `evaluate_actions` takes observations
  and bounded actions; it has no retained-sample argument.
- `stable_baselines3/common/on_policy_algorithm.py`: rollout collection stores
  actions and old likelihoods, not pre-tanh samples.
- Historical issue https://github.com/DLR-RM/stable-baselines3/issues/1593 reports
  gSDE NaNs. It is related context, not a confirmed instance of this mechanism.
- https://github.com/DLR-RM/stable-baselines3/pull/2249 concerns bounding the mean
  of non-gSDE Gaussian policies, not retained-latent replay.

## Suggested implementation discussion

Preserve the exact sampled latent through collection and minibatch shuffling,
and score the same fixed sample for both rollout and updated likelihoods. Ask
maintainers whether they prefer an optional buffer field, a specialised buffer,
or another representation. Avoid silently changing what `actions` means for
custom policies, callbacks, or downstream algorithms. Review A2C and contrib
algorithms sharing these interfaces. Keep entropy behavior explicit and separate
from the actor-ratio correction. Do not import the research fork or add CELL,
KL guards, reward changes, and tuning in the same change.

Validation for an eventual patch should cover a failing-then-passing saturated
replay-gradient regression, unsaturated parity, vector and dictionary observations,
buffer alignment, ordinary unsquashed PPO behavior, and save/load. The present
script is a diagnostic that asserts the CURRENT discrepancy; it is not yet a
regression test that should pass after the fix. Once behavior is changed, the
regression should assert agreement with the retained reference.

## Submission policy

Source: https://github.com/DLR-RM/stable-baselines3/blob/7cfb4dd6055e74b5caa4ed4d6777492209946e26/CONTRIBUTING.md

The upstream guide requires an issue before a PR and public disclosure of code
assistant use. It explicitly excludes fully LLM-generated PRs unless initiated
by a maintainer. Human review is required before publishing this generated draft;
do not describe it as independently human-authored. No maintainer approval exists.
Future PR requirements include regression tests, type/style checks, documentation
where needed, and a changelog entry. No GitHub posting was performed.
