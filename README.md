# Qwen3.6 NVFP4 on Blackwell with vLLM

Docker Compose setup for serving [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) with [vLLM](https://github.com/vllm-project/vllm), targeting NVIDIA Blackwell hardware. Exposes an OpenAI-compatible API on port 8000 with the model's full 262,144-token context, fp8 KV cache, and MTP speculative decoding enabled.

## Requirements

- An NVIDIA Blackwell GPU — NVFP4 relies on Blackwell's native FP4 tensor cores. The defaults here assume a ~96 GB card; see [Tuning](#tuning) for smaller GPUs.
- A recent NVIDIA driver, Docker, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- A Hugging Face token for the model download.

## Quick start

```bash
cp .env.example .env   # then fill in your HF token (or export HF_TOKEN instead)
./start.sh
```

First boot downloads the model weights and warms up the engine, which can take several minutes; the healthcheck allows up to 10 minutes. Watch progress with `docker compose logs -f`.

Once healthy, test it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Model weights are cached in `${HOME}/.cache/huggingface` on the host (bind-mounted into the container), so they survive container rebuilds and are shared with any local Hugging Face tooling.

## What's configured

| Setting | Value | Why |
| --- | --- | --- |
| Context length | 262,144 tokens | The model's full native context |
| KV cache | fp8 | Roughly doubles KV capacity vs fp16 |
| Speculative decoding | MTP, 2 draft tokens | Uses the model's bundled multi-token-prediction head |
| Reasoning parser | `qwen3` | Exposes thinking via the API's `reasoning` field |
| Tool-call parser | `qwen3_xml` | Matches the XML-style tool calls this model emits (`hermes` silently fails to parse them) |
| Quantization | auto-detected | No `--quantization` flag: vLLM detects compressed-tensors and picks the fast cute-DSL W4A4 kernel; forcing a backend can cost ~2.5x decode throughput |

## Swapping models

The compose file targets Qwen3.6-27B, but any Qwen3.6 NVFP4 checkpoint works the same way. Change the model (the first entry under `command:`, passed positionally) and `--served-model-name` in `docker-compose.yml`, then `docker compose up -d`.

Verified with [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) (MoE, 3B active parameters): with only those two values changed, vLLM resolves the MoE architecture, loads the bundled MTP head, and picks the NVFP4 MoE fast path (`FLASHINFER_CUTLASS` backend). First boot reached healthy in ~7 minutes including the cold weight download, within the healthcheck's 10-minute allowance. See [Benchmarks](#benchmarks) for how it performs.

## Benchmarks

All figures below are vLLM v0.26.0 with this repo's compose files as-is, MTP speculative decoding enabled, on two Blackwell machines:

| Machine | GPU | Memory | Compose file | `--gpu-memory-utilization` | `--max-num-batched-tokens` |
| --- | --- | --- | --- | --- | --- |
| **RTX PRO 6000** | RTX PRO 6000 Blackwell Workstation (sm120) | 96 GB dedicated | [docker-compose.yml](docker-compose.yml) | 0.85 | 32768 |
| **DGX Spark** | GB10 Grace Blackwell (sm121) | 121 GB unified | [docker-compose.spark.yml](docker-compose.spark.yml) | 0.78 | 2048 |

Both take the native NVFP4 path — `FlashInferCutlassNvFp4LinearKernel` for dense GEMMs, the `FLASHINFER_CUTLASS` backend for MoE — including the GB10 on the stock upstream image, with no Marlin fallback.

### Chat decode throughput

Greedy chat completions generating 1024 tokens, decode rate timed from the first streamed token to the last so prefill is excluded. Single-stream is the mean of 3 runs; the aggregate is one batch of 8 concurrent requests. This is what [bench.py](bench.py) measures:

| Machine | Model | Single-stream decode | 8 concurrent, aggregate | MTP acceptance | KV cache capacity |
| --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 | [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 113 tok/s | 775 tok/s | 67% | 1.51M tokens |
| RTX PRO 6000 | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 281 tok/s | 1,546 tok/s | 69% | 4.34M tokens |
| DGX Spark | [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 22 tok/s | 139 tok/s | 70% | 2.00M tokens |
| DGX Spark | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 76 tok/s | 316 tok/s | 67% | 5.70M tokens |

Reproduce against a running server with [bench.py](bench.py) (no dependencies beyond the standard library):

```bash
./bench.py                          # defaults to qwen3.6-27b
./bench.py --model qwen3.6-35b-a3b
```

The MoE's 3B active parameters make it 2.5x faster per stream than the 27B dense model on the RTX PRO 6000, and 3.5x faster on the Spark, while its smaller KV footprint nearly triples cache capacity at the same 262k context. The Spark is 3.7–5.3x slower per stream than the RTX PRO 6000 — LPDDR5X bandwidth (~273 GB/s vs ~1.8 TB/s) is the decode limiter — but holds a *larger* KV cache despite the lower utilization fraction, since the GB10 has more total memory. Speculative-decode acceptance is prompt-dependent; expect a few points of variance either way.

### Standard serving benchmark

The table above uses one fixed prompt. For load-shaped numbers, `vllm bench serve` against the same servers: the `random` dataset over `/v1/completions`, `--ignore-eos` so every request emits exactly 1024 output tokens, and `--request-rate inf` so all requests are queued at once and the server is never idle. The client runs inside the serving container, so no network sits between it and the server:

```bash
docker compose exec vllm vllm bench serve \
  --backend openai --base-url http://localhost:8000 \
  --model unsloth/Qwen3.6-27B-NVFP4 --served-model-name qwen3.6-27b \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 128 --max-concurrency 64 --request-rate inf \
  --ignore-eos --seed 0
```

Request counts scale with concurrency so every run stays long enough to be steady-state: 8/32/64/128 prompts at c1/c8/c32/c64 for 1k inputs, and 4/16/32/64 for 8k inputs.

Because every request is submitted up front, TTFT at c32/c64 is dominated by queueing behind other prefills rather than by prefill latency itself — read it as a saturation measure, not as the latency a single interactive user would see. At 8k/c64 the server has 512k tokens of prompt to chew through before the last request can emit anything, and that, not any per-request cost, is what the number reports.

`vllm bench serve` does not pin sampling parameters, so requests inherit the checkpoint's defaults (`temperature 1.0`, `top_k 20`, `top_p 0.95`). That makes MTP acceptance vary run to run, and single-stream throughput tracks acceptance almost perfectly (r ≥ 0.98 across repeats). The **c1 column is therefore the median of 4 runs**; spread there reaches 22–28% on the worst cells. Every other column is a single run, where batching averages the noise out and the effects discussed below are far larger than it.

#### 1,024-token prompts

**Output token throughput (tok/s)** — higher is better

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 116 | 743 | 1,826 | 2,019 |
| RTX PRO 6000 · 35B-A3B | 256 | 1,049 | 2,391 | 3,406 |
| DGX Spark · 27B | 21 | 125 | 230 | 316 |
| DGX Spark · 35B-A3B | 71 | 235 | 433 | 592 |

**Median TTFT (s) / median TPOT (ms)** — lower is better

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 0.10 / 8.5 | 0.14 / 9.8 | 1.74 / 14.3 | 1.14 / 24.0 |
| RTX PRO 6000 · 35B-A3B | 0.06 / 3.7 | 0.08 / 6.2 | 0.46 / 9.5 | 0.30 / 14.1 |
| DGX Spark · 27B | 0.53 / 45.9 | 0.74 / 57.5 | 2.78 / 99.9 | 2.54 / 155.8 |
| DGX Spark · 35B-A3B | 0.22 / 12.9 | 0.35 / 28.9 | 0.92 / 58.3 | 1.28 / 89.4 |

#### 8,192-token prompts

**Output token throughput (tok/s)**

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 109 | 503 | 735 | 745 |
| RTX PRO 6000 · 35B-A3B | 260 | 927 | 1,566 | 1,955 |
| DGX Spark · 27B | 21 | 91 | 138 | 146 |
| DGX Spark · 35B-A3B | 74 | 176 | 271 | 331 |

**Median TTFT (s) / median TPOT (ms)**

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 0.73 / 8.5 | 2.27 / 11.3 | 17.57 / 22.2 | 30.10 / 44.2 |
| RTX PRO 6000 · 35B-A3B | 0.20 / 3.7 | 0.58 / 6.4 | 4.34 / 11.5 | 7.36 / 19.1 |
| DGX Spark · 27B | 3.64 / 44.3 | 8.82 / 69.4 | 63.30 / 146.2 | 129.60 / 246.7 |
| DGX Spark · 35B-A3B | 1.36 / 12.0 | 3.46 / 31.8 | 24.49 / 70.9 | 50.93 / 117.2 |

Reading these:

- **The 27B dense model is prefill-bound at 8k.** On the RTX PRO 6000 its throughput plateaus at ~740 tok/s from c32 onward — going from 32 to 64 concurrent buys 1% more throughput and doubles TPOT. The MoE keeps scaling to c64.
- **The MoE is the right choice for the Spark.** At 8k it delivers ~2x the throughput of the 27B dense model and reaches the first token ~2.5x sooner at every concurrency, because 3B active parameters prefill about 2.7x faster on a bandwidth-limited part (6.0k vs 2.3k tok/s).
- **Prefill chunk size is the largest configuration effect measured on either machine.** Cutting `--max-num-batched-tokens` from 32768 to 2048 raised the Spark's 8k throughput by 27–63% and more than halved its TTFT, at no cost in memory — see [Tuning](#tuning). The same change is not worth making on the RTX PRO 6000.
- **MTP acceptance holds up under load**: 62–80% across the matrix, with no systematic decay as concurrency rises, even though the `random` dataset feeds the model incoherent prompts.

Both machines ran the identical matrix, each against its own compose file as shipped. Every run in these tables completed all requests with zero failures.

## Tuning

- `--gpu-memory-utilization 0.85` is a concurrency ceiling, not just a KV-cache dial. The flag only decides how much memory is *left over* for the KV cache after a profiling pass estimates peak activation — it does not cap allocation. That profile under-measures the Qwen3.6 GDN linear-attention path (`chunk_gated_delta_rule`), whose scratch buffers scale with total batched tokens. At 0.92 on a 96 GB card the engine dies outright once a 32-way batch fills `--max-num-batched-tokens`: `torch.OutOfMemoryError` on a 372 MiB allocation with 285 MiB free, taking the container down with it. 0.85 costs ~11% of KV capacity (1.71M → 1.51M tokens, still 5.8x the full context) and no measurable throughput. Raise it only if you also lower `--max-num-seqs`.
- On unified-memory machines (DGX Spark / GB10) use [docker-compose.spark.yml](docker-compose.spark.yml) instead — select it with `COMPOSE_FILE=docker-compose.spark.yml` in `.env`. The GPU shares the 121 GB with the OS, so utilization is capped at 0.78: 0.85 measured to transiently starve the host below 5 GiB during KV-cache allocation, and 0.92 livelocks the machine hard enough to need a power cycle (disable swap so an overrun OOM-kills the engine instead of thrashing).
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights.
- **`--max-num-batched-tokens` should be sized to how fast your GPU prefills, not to how much memory it has.** A chunked-prefill step blocks every decoding request until it finishes, so the worst-case decode stall is `max-num-batched-tokens ÷ prefill rate` — visible directly as p99 inter-token latency. The RTX PRO 6000 prefills the 27B at ~11k tok/s, so its 32768-token chunk costs a ~3 s stall; the GB10 prefilled at ~1.3k tok/s under that setting, so the *same value* cost it ~25 s (measured p99 ITL 28.2 s at 8k/c64). Hence 2048 on the Spark and 32768 on the RTX.

  On the Spark the smaller chunk also makes prefill itself faster, which is the bigger effect: an 8k prompt prefills in 3.64 s at 2048 versus 6.23 s at 32768, a 1.7x speedup. A 1k prompt — which fits in one chunk under either setting — is unchanged at 0.53 s, confirming the difference comes from chunk size rather than from anything else in the config. Sweeping 32768/8192/4096/2048/1024 put the knee at 2048; 1024 gains a further 7% at c64 but gives up 16% at c32.

  Two caveats. This was tuned on the **27B dense** model, which is what the compose file ships; the 35B-A3B MoE prefills ~2.7x faster and sees roughly no benefit — at 8k it measured −14/−5/+4% at c8/c32/c64 — so revisit the value if you swap models. And on the RTX PRO 6000 the sweep found **no dominant value** — 32768 is 23% faster at c8 while 8192 is 11% faster at c64 — so it keeps the larger chunk.

- `--max-num-seqs 64` is sized for a workstation serving a handful of concurrent clients; raise it for heavier batch serving. Note that memory headroom, not this setting, is what actually bounds usable concurrency — see the utilization note above.

## License

[MIT](LICENSE)
