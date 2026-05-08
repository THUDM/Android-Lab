# Clean Inference Strategy Layout

This branch starts from upstream `main` and keeps AndroidLab's native text-only
CoT generate path unchanged.

## Strategy 1: CoT-Generate

Config task class:

```yaml
task:
  class: CoTGenerate_AutoTest
```

`CoTGenerate_AutoTest` is only a semantic alias of the upstream
`TextOnlyMobileTask_AutoTest`. It uses the original AndroidLab XML prompt with
`Analysis` and `Operation` examples.

## Strategy 2: Rank-then-Verify

Config task class:

```yaml
task:
  class: RankThenVerify_AutoTest
```

All non-CoT candidate-based strategies are parameterizations of this one class:

| Variant | Parameterization |
|---|---|
| No-CoT Generate | `rtv_k: 1`; score candidate labels once and execute the top candidate |
| Rank-then-Verify | `1 < rtv_k < candidate_count`; rank top-k by label logits, then independently verify each finalist with Yes/No logits |
| Full Verify | `rtv_k: "max"`; verify every candidate with Yes/No logits without rank pruning |

Main parameters live under `agent.args` so the upstream `TaskConfig` does not
need extra fields:

| Parameter | Meaning |
|---|---|
| `rtv_k` | Number of candidates entering verification; `1` and `"max"` are special cases |
| `rtv_refine_mode` | `"verify"`; each selected candidate is scored by an independent Yes/No prompt |
| `rtv_verify_workers` | Number of concurrent Yes/No verify requests; important for `rtv_k: "max"` |
| `rtv_max_candidates` | Maximum candidate actions listed in prompts; capped at 26 for single-token A-Z logit scoring |
| `rtv_xml_max_chars` | XML truncation guard for long screens |
| `rtv_include_control_actions` | Include Back/Home/Enter/Swipe candidates |
| `rtv_include_type_action` | Include a generic Type candidate for focused text fields |

Rank prompts ask for one uppercase candidate letter only. Verification prompts
ask one candidate at a time for exact `Yes` or `No`; the selected action is the
candidate with the highest exact `Yes` logit. Both steps request next-token
logits (`return_prefill_logits`) from the OpenAI-compatible vLLM server.
Fillable candidates are selected first, then a separate fill prompt replaces
`<TEXT_TO_FILL>` or `<MESSAGE_TO_FILL>`.

## Example configs

Qwen3.5-4B:

- `configs/qwen35-4b-cot-generate-local.yaml`
- `configs/qwen35-4b-rtv-k1-local.yaml`
- `configs/qwen35-4b-rtv-k5-local.yaml`
- `configs/qwen35-4b-rtv-kmax-local.yaml`

Gemma4-E4B-it:

- `configs/gemma4-e4b-it-cot-generate-local.yaml`
- `configs/gemma4-e4b-it-rtv-k1-local.yaml`
- `configs/gemma4-e4b-it-rtv-k3-local.yaml`
- `configs/gemma4-e4b-it-rtv-kmax-local.yaml`

## Smoke script

```bash
bash scripts/run_clean_strategy_smoke_20260507.sh
```

The script defaults to `clock_10` and runs all eight example configs against
the already deployed local OpenAI-compatible router at `127.0.0.1:18000`.
