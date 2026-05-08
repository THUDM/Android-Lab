# Next Agent Benchmark Guide

This repository is the clean AndroidLab baseline workspace:

```bash
cd /mnt/c/Working/Android-Lab-UpstreamBaseline
```

The goal is to test two inference families only:

- `CoT-Generate`: native AndroidLab text-only generate prompt, unchanged.
- `Rank-then-Verify`: candidate ranking by one-token letter logits, then per-candidate Yes/No verify logits. `k=1` is No-CoT Generate; `k=max` is Full Verify.

## Environment

Run benchmark commands from WSL. The known working Python is:

```bash
/mnt/c/Working/Android-Lab/.wsl-venv/bin/python
```

The configs call a local OpenAI-compatible router:

```text
http://127.0.0.1:18000/v1
```

If the tunnel is missing, recreate it from WSL:

```bash
ssh -p <SSH_PORT> -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -f -N -L 18000:127.0.0.1:18000 <USER>@<REMOTE_HOST>
```

Quick checks:

```bash
curl -s http://127.0.0.1:18000/v1/models
docker ps --filter ancestor=android_eval:latest
```

Stop stale emulator containers before reruns:

```bash
docker ps --filter ancestor=android_eval:latest -q | xargs -r docker stop
```

## Configs

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

Important RTV args:

- `rtv_k`: `1`, middle k, or `"max"`.
- `rtv_verify_workers`: concurrent Yes/No verify requests. Default is `8`.
- `rtv_max_candidates`: capped at `26` because letter scoring uses single uppercase A-Z tokens.
- `top_logprobs`: default `20`.

## Smoke Script

Use the single maintained smoke entrypoint:

```bash
bash scripts/run_strategy_smoke.sh --task clock_10
```

Useful smaller runs:

```bash
# Fast qwen check: native generate, k=1, k=5
bash scripts/run_strategy_smoke.sh \
  --task clock_10 \
  --models qwen \
  --strategies cot,k1,kmid \
  --timeout 900

# Full 8-combination check on one task
bash scripts/run_strategy_smoke.sh \
  --task clock_10 \
  --models qwen,gemma \
  --strategies cot,k1,kmid,kmax \
  --timeout 1200

# Print commands only
bash scripts/run_strategy_smoke.sh --task clock_10 --dry-run
```

The script creates fresh run names using a timestamp suffix, so AndroidLab should not skip tasks due to existing result directories. It writes logs under:

```text
logs/evaluation/smoke_<model>_<strategy>_<task>_<suffix>.log
```

Each run is individually timed out, and the script stops `android_eval:latest` containers after each run.

## Direct Commands

Manual one-off run:

```bash
/mnt/c/Working/Android-Lab/.wsl-venv/bin/python eval.py \
  -n manual_qwen_k5_clock10_$(date +%Y%m%d_%H%M%S) \
  -c configs/qwen35-4b-rtv-k5-local.yaml \
  --task_id clock_10 \
  -p 1
```

Do not reuse an old `-n` unless you intentionally want AndroidLab to skip already-run task ids.

## Server Monitoring

The remote server is:

```bash
ssh -p <SSH_PORT> <USER>@<REMOTE_HOST>
```

Current vLLM processes are expected on ports:

- Qwen: `18100`, `18101`, `18102`, `18103`
- Gemma: `18104`, `18105`, `18106`, `18107`
- Router: `18000`

vLLM was launched with:

```text
--enable-prefix-caching --enable-chunked-prefill
```

The useful INFO logs are:

```text
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_qwen35-4b__18100.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_qwen35-4b__18101.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_qwen35-4b__18102.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_qwen35-4b__18103.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_gemma4-e4b-it__18104.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_gemma4-e4b-it__18105.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_gemma4-e4b-it__18106.log
/tmp/androidlab_qwen_gemma_4plus4_logs/vllm_gemma4-e4b-it__18107.log
```

Monitor prefix cache and queue status:

```bash
ssh -p <SSH_PORT> <USER>@<REMOTE_HOST> \
  "tail -f /tmp/androidlab_qwen_gemma_4plus4_logs/vllm_qwen35-4b__18100.log"
```

Look for lines like:

```text
Avg prompt throughput: ...
Avg generation throughput: ...
Running: ... reqs, Waiting: ... reqs
GPU KV cache usage: ...%
Prefix cache hit rate: ...%
```

For exact counters, `/metrics` is also available on the vLLM ports:

```bash
ssh -p <SSH_PORT> <USER>@<REMOTE_HOST> \
  "curl -s http://127.0.0.1:18100/metrics | grep -E 'prefix_cache|prompt_tokens_cached|kv_cache_usage'"
```

## Known Results From Latest Smoke

On `clock_10`:

- `qwen kmax`: completed in 1 round with `finish(message="The earliest open alarm is 7:30 AM.")`.
- `gemma CoT`: completed in 2 rounds.
- `gemma k1`: completed in 6 rounds.
- `gemma k3`: completed in 9 rounds.
- `gemma kmax`: reached max rounds, looping between alarm elements.

Some earlier qwen retry logs were skipped because their run names already existed. Use `run_strategy_smoke.sh` with a fresh suffix for reliable retesting.

## Implementation Pointers

- `evaluation/auto_test.py`: exposes `CoTGenerate_AutoTest` and `RankThenVerify_AutoTest`.
- `evaluation/rank_then_verify.py`: candidate enumeration, rank prompt, Yes/No verify prompt, fill prompt.
- `agent/model.py`: shared one-token prefill logits logic:
  - `score_letters(...)` for rank.
  - `score_yes_no(...)` for verify.

Rank uses exact uppercase candidate labels only. Verify uses exact `Yes` and `No` tokens only.
