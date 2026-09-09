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
| Prefix caching | off (pinned) | vLLM disables it by default for hybrid models — Qwen3.6 is 48 linear-attention + 16 full-attention layers — and keeps it opt-in while the feature matures. Pinned explicitly with `--no-enable-prefix-caching` so a future default flip can't silently change behaviour. **Multi-turn chat therefore re-prefills the whole conversation each turn**; see [Tuning](#tuning) before enabling it |
| Speculative decoding | MTP, 2 draft tokens | Uses the model's bundled multi-token-prediction head |
| Reasoning parser | `qwen3` | Exposes thinking via the API's `reasoning` field |
| Tool-call parser | `qwen3_xml` | Matches the XML-style tool calls this model emits (`hermes` silently fails to parse them) |
| Quantization | auto-detected | No `--quantization` flag: vLLM detects compressed-tensors and picks the fast cute-DSL W4A4 kernel; forcing a backend can cost ~2.5x decode throughput |

## Swapping models

The compose file targets Qwen3.6-27B, but any Qwen3.6 NVFP4 checkpoint works the same way. Change the model (the first entry under `command:`, passed positionally) and `--served-model-name` in `docker-compose.yml`, then `docker compose up -d`.

Verified with [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) (MoE, 3B active parameters): with only those two values changed, vLLM resolves the MoE architecture, loads the bundled MTP head, and picks the NVFP4 MoE fast path (`FLASHINFER_CUTLASS` backend). First boot reached healthy in ~7 minutes including the cold weight download, within the healthcheck's 10-minute allowance. See [Benchmarks](#benchmarks) for how it performs.

## Two-Spark cluster

[run_cluster.sh](run_cluster.sh) joins two DGX Sparks into a Ray cluster and serves one model across both GPUs with tensor parallelism (TP=2), NCCL riding RDMA (RoCE) over the dedicated 200 GbE link between them. A single GB10 already runs these models comfortably; what the second Spark buys is headroom — twice the aggregate memory bandwidth and twice the KV-cache memory — at the price of putting every tensor-parallel all-reduce on the wire.

```bash
./run_cluster.sh head                # on the head Spark
./run_cluster.sh worker              # on the other Spark (head IP from .env, or pass it)
./run_cluster.sh serve 27b           # back on the head; or: serve 35b-a3b
```

`status` reports tmux/container/Ray/API state on any node; `stop` tears down that node's half. Everything long-running lives in detached tmux sessions (`ray-node` holds the Ray container on each node, `vllm-serve` holds the engine on the head), so an SSH drop doesn't take the cluster down; engine output is mirrored to `~/vllm-cluster-serve.log`. Set `CLUSTER_HEAD_IP` in `.env` (see `.env.example`) to the head's IP *on the 200G link*; `CLUSTER_IF` and `CLUSTER_HCA` default to the Spark's 200G netdev and its RoCE device. The image ships without Ray, so each node pip-installs it at container start (~1 min, needs internet). Once healthy, the API is on port 8000 of the head node, same as the single-node compose.

The serve profile reuses the single-Spark tuning unchanged — fp8 KV cache, utilization 0.78 (a per-node fraction; the host-starvation ceiling it protects doesn't move by adding a machine), `--max-num-batched-tokens 2048` — and keeps MTP speculative decoding on.

One caveat if you pin your own image: multi-node needs **vLLM v0.27.0 or later**. v0.26.0's shared-memory message queue — which the engine uses to drive cross-node workers — can lose a reader wakeup notification, parking the engine and both workers forever on queues that have data; the engine then dies minutes later with "RPC call to sample_tokens timed out". v0.27.0 bounds the park time so a lost wakeup recovers within ~5 s. Single-node deployments don't exercise this path at risk.

## 32 GB cards (RTX 5090)

[docker-compose.rtx5090.yml](docker-compose.rtx5090.yml) serves [kelnei/Qwen3.8-27B-NVFP4](https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4) on a single RTX 5090. Select it with `COMPOSE_FILE=docker-compose.rtx5090.yml` in `.env`. It targets 1–8 concurrent requests, not wide serving, and it is a **different model** from the rest of this repo — the Qwen3.8 checkpoint, whose numbers are not comparable to the Qwen3.6 tables below.

**This config assumes a headless machine with the card dedicated to the model.** That is a precondition, not a detail. The KV pool below is sized to within ~1.9 GiB of the card's total capacity, measured under load; a desktop session, a browser, or any other CUDA process sharing the GPU takes that headroom and the engine dies on the first concurrent burst rather than degrading gracefully. On a shared card none of these numbers hold and the pool has to be re-derived with the rest of the load subtracted from the budget.

At 32 GB the KV cache is whatever survives the weights, and there is not much slack:

| | |
| --- | --- |
| Total (headless, dedicated) | 32,607 MiB |
| Weights + non-torch | 22.37–22.55 GiB |
| CUDA graphs | 0.14 GiB |
| Peak activation | 2.0 GiB |
| **KV cache** | **5.75 GiB → 153,382 tokens** |

Two departures from the other configs make that fit, both about the same underlying problem — the profiling pass under-measures the gated-DeltaNet path, so the fraction-based sizing is unsafe here in a way it is not on a 96 GB card:

- **`--kv-cache-memory` is set explicitly and `--gpu-memory-utilization` is parked at 0.98**, deliberately non-binding. Sizing the pool from the fraction, or from the startup line that suggests a `--kv-cache-memory` value "to fully utilize gpu memory", produces a server that boots clean and then dies on the first concurrent burst: at the suggested 6.03 GiB it hit `torch.OutOfMemoryError` on a 272 MiB allocation with 269 MiB free, took `EngineDeadError`, and 500'd 8 of 16 requests at a 31,702 MiB peak. The 5.75 GiB above was sized from a load test instead and peaks at 30,706 MiB with no failures.
- **`--max-num-batched-tokens 2048`, for memory rather than latency.** The GDN chunked scratch buffer scales at ~0.22 GiB per 1k tokens as profiled and ~0.38 GiB per 1k under load, straight out of the KV pool; 32768 does not boot at all (`Available KV cache memory: -2.96 GiB`). Nothing is lost by going small — prefill on this card is compute-bound at ~9.5k tok/s and chunk size is neutral from 2048 to 8192, while 16384 is 16–26% *worse* at 16k–30k prompts. Raising 2048 → 8192 would buy no prefill speed and cost ~47k KV tokens.

One counterintuitive result: a *larger* `--max-model-len` yields **more** usable KV tokens from identical bytes, because it changes how the hybrid allocator pads attention and mamba pages to a common size — 32768 gives 120,149 tokens, 65536 gives 144,179, and the 131072 shipped here gives 153,382. It is isolated to that flag; `--max-num-seqs` 8 and 16 give the same count. The trade-off is at the top end: one request at the full 131k context consumes most of the pool (max concurrency 1.17x), so long-context and concurrent use are mutually exclusive here.

Prefix caching is **on** in this file, unlike the others in this repo. Note the hybrid allocator forces a 1600-token KV block and vLLM never reuses the last matched block, so a shared prefix below 3,200 tokens caches nothing at all — see [Tuning](#tuning) for the measured hit rates.

Measured on this config, greedy, MTP k=2 at 54.9% acceptance:

| | c1 | c8 |
| --- | --- | --- |
| Chat decode | 99.5 tok/s | 721 tok/s |
| 8k prompts, output throughput | — | 365.6 tok/s |
| TTFT, 1k / 8k prompt | 135 ms / 903 ms | median 2.91 s at 8k |

Those were measured on v0.27.1. Re-run on v0.29.0 (2026-09-09, same bench, same-day v0.27.1 control of 100.1 / 664 tok/s), chat decode moved to **112.2 tok/s at c1 and 893 tok/s at c8**, acceptance 60.1%: the new default V2 model runner captures FULL decode CUDA graphs where v0.27.1 fell back to PIECEWISE with MTP on FlashInfer. The KV pool is unchanged at 153,382 tokens, but graph capture takes 0.33 GiB instead of 0.14 and idle usage is ~1.4 GiB higher, so the headroom under a 16-request burst of unique 13k-token prompts is now ~1.3 GiB (peak 31,274 MiB, no failures). The 8k-prompt and TTFT rows were not re-measured.

Image input works on this config as shipped (the checkpoint is a VL model), verified end-to-end against the served endpoint.

## Benchmarks

All figures below were measured on vLLM v0.26.0 (the cluster on a v0.27 pre-release nightly) with this repo's config as-is, MTP speculative decoding enabled, on three Blackwell setups; the repo now pins v0.29.0 (the RTX 5090 row above is the only one re-verified on it so far):

| Machine | GPU | Memory | Config | `--gpu-memory-utilization` | `--max-num-batched-tokens` |
| --- | --- | --- | --- | --- | --- |
| **RTX PRO 6000** | RTX PRO 6000 Blackwell Workstation (sm120) | 96 GB dedicated | [docker-compose.yml](docker-compose.yml) | 0.85 | 32768 |
| **DGX Spark** | GB10 Grace Blackwell (sm121) | 121 GB unified | [docker-compose.spark.yml](docker-compose.spark.yml) | 0.78 | 2048 |
| **2x DGX Spark** | 2x GB10, TP=2 over 200 GbE (RoCE) | 2x 121 GB unified | [run_cluster.sh](run_cluster.sh) | 0.78 per node | 2048 |

All of them take the native NVFP4 path — `FlashInferCutlassNvFp4LinearKernel` for dense GEMMs, the `FLASHINFER_CUTLASS` backend for MoE — including the GB10s on stock upstream images, with no Marlin fallback.

### Chat decode throughput

Greedy chat completions generating 1024 tokens, decode rate timed from the first streamed token to the last so prefill is excluded. Single-stream is the mean of 3 runs; the aggregate is one batch of 8 concurrent requests. This is what [bench.py](bench.py) measures:

| Machine | Model | Single-stream decode | 8 concurrent, aggregate | MTP acceptance | KV cache capacity |
| --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 | [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 113 tok/s | 775 tok/s | 67% | 1.51M tokens |
| RTX PRO 6000 | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 281 tok/s | 1,546 tok/s | 69% | 4.34M tokens |
| DGX Spark | [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 22 tok/s | 139 tok/s | 70% | 2.00M tokens |
| DGX Spark | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 76 tok/s | 316 tok/s | 67% | 5.70M tokens |
| 2x DGX Spark | [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 28 tok/s | 169 tok/s | 70% | 4.48M tokens |
| 2x DGX Spark | [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 64 tok/s | 311 tok/s | 68% | 13.26M tokens |

Reproduce against a running server with [bench.py](bench.py) (no dependencies beyond the standard library):

```bash
./bench.py                          # defaults to qwen3.6-27b
./bench.py --model qwen3.6-35b-a3b
```

The MoE's 3B active parameters make it 2.5x faster per stream than the 27B dense model on the RTX PRO 6000, and 3.5x faster on the Spark, while its smaller KV footprint nearly triples cache capacity at the same 262k context. The Spark is 3.7–5.3x slower per stream than the RTX PRO 6000 — LPDDR5X bandwidth (~273 GB/s vs ~1.8 TB/s) is the decode limiter — but holds a *larger* KV cache despite the lower utilization fraction, since the GB10 has more total memory. Speculative-decode acceptance is prompt-dependent; expect a few points of variance either way.

The cluster rows show what cross-node tensor parallelism does and doesn't buy. The bandwidth-bound dense 27B gains from splitting each layer across two memory systems: +27% single-stream, +21% at 8 concurrent. The MoE *loses* single-stream speed (64 vs 76 tok/s): with only 3B active parameters there is little decode work to split, so the per-layer all-reduce crossing the 200 GbE link (~25 GB/s vs ~273 GB/s local) dominates. What the cluster buys both models unambiguously is KV capacity — 2.2–2.3x, to 13.26M tokens on the MoE.

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
| 2x DGX Spark · 27B | 28 | 148 | 315 | 420 |
| 2x DGX Spark · 35B-A3B | 55 | 269 | 479 | 634 |

**Median TTFT (s) / median TPOT (ms)** — lower is better

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 0.10 / 8.5 | 0.14 / 9.8 | 1.74 / 14.3 | 1.14 / 24.0 |
| RTX PRO 6000 · 35B-A3B | 0.06 / 3.7 | 0.08 / 6.2 | 0.46 / 9.5 | 0.30 / 14.1 |
| DGX Spark · 27B | 0.53 / 45.9 | 0.74 / 57.5 | 2.78 / 99.9 | 2.54 / 155.8 |
| DGX Spark · 35B-A3B | 0.22 / 12.9 | 0.35 / 28.9 | 0.92 / 58.3 | 1.28 / 89.4 |
| 2x DGX Spark · 27B | 0.43 / 34.1 | 0.63 / 43.8 | 1.65 / 89.6 | 2.49 / 115.7 |
| 2x DGX Spark · 35B-A3B | 0.18 / 16.9 | 0.28 / 25.7 | 1.18 / 51.9 | 1.76 / 75.4 |

#### 8,192-token prompts

**Output token throughput (tok/s)**

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 109 | 503 | 735 | 745 |
| RTX PRO 6000 · 35B-A3B | 260 | 927 | 1,566 | 1,955 |
| DGX Spark · 27B | 21 | 91 | 138 | 146 |
| DGX Spark · 35B-A3B | 74 | 176 | 271 | 331 |
| 2x DGX Spark · 27B | 27 | 111 | 161 | 178 |
| 2x DGX Spark · 35B-A3B | 58 | 211 | 299 | 355 |

**Median TTFT (s) / median TPOT (ms)**

| Config | c1 | c8 | c32 | c64 |
| --- | --- | --- | --- | --- |
| RTX PRO 6000 · 27B | 0.73 / 8.5 | 2.27 / 11.3 | 17.57 / 22.2 | 30.10 / 44.2 |
| RTX PRO 6000 · 35B-A3B | 0.20 / 3.7 | 0.58 / 6.4 | 4.34 / 11.5 | 7.36 / 19.1 |
| DGX Spark · 27B | 3.64 / 44.3 | 8.82 / 69.4 | 63.30 / 146.2 | 129.60 / 246.7 |
| DGX Spark · 35B-A3B | 1.36 / 12.0 | 3.46 / 31.8 | 24.49 / 70.9 | 50.93 / 117.2 |
| 2x DGX Spark · 27B | 2.82 / 32.0 | 6.43 / 58.6 | 48.38 / 121.9 | 98.88 / 178.8 |
| 2x DGX Spark · 35B-A3B | 1.05 / 15.5 | 1.90 / 31.1 | 22.97 / 58.7 | 42.21 / 95.2 |

Reading these:

- **The 27B dense model is prefill-bound at 8k.** On the RTX PRO 6000 its throughput plateaus at ~740 tok/s from c32 onward — going from 32 to 64 concurrent buys 1% more throughput and doubles TPOT. The MoE keeps scaling to c64.
- **The MoE is the right choice for the Spark.** At 8k it delivers ~2x the throughput of the 27B dense model and reaches the first token ~2.5x sooner at every concurrency, because 3B active parameters prefill about 2.7x faster on a bandwidth-limited part (6.0k vs 2.3k tok/s).
- **Prefill chunk size is the largest configuration effect measured on either machine.** Cutting `--max-num-batched-tokens` from 32768 to 2048 raised the Spark's 8k throughput by 27–63% and more than halved its TTFT, at no cost in memory — see [Tuning](#tuning). The same change is not worth making on the RTX PRO 6000.
- **MTP acceptance holds up under load**: 62–80% across the matrix, with no systematic decay as concurrency rises, even though the `random` dataset feeds the model incoherent prompts.
- **Cluster scaling is model-dependent, and batching pays back what single-stream gives up.** The dense 27B gains everywhere: +33% at 1k/c64 (420 vs 316 tok/s) and +22% at 8k/c64 over one Spark. The MoE loses ~22% at c1 — the all-reduce latency cost discussed under [chat decode](#chat-decode-throughput) — but the wire cost amortizes across a batch: +14–20% at c8 and still ahead at c32/c64. TTFT also drops nearly across the board (an 8k prompt prefills ~1.3x faster on both models); the MoE's 1k c32/c64 cells are the one exception, where the queue drains fast enough that the all-reduce cost shows up in TTFT instead.

All three configurations ran the identical matrix, each against its config as shipped. Every run in these tables completed all requests with zero failures.

## Tuning

- `--gpu-memory-utilization 0.85` is a concurrency ceiling, not just a KV-cache dial. The flag only decides how much memory is *left over* for the KV cache after a profiling pass estimates peak activation — it does not cap allocation. That profile under-measures the Qwen3.6 GDN linear-attention path (`chunk_gated_delta_rule`), whose scratch buffers scale with total batched tokens. At 0.92 on a 96 GB card the engine dies outright once a 32-way batch fills `--max-num-batched-tokens`: `torch.OutOfMemoryError` on a 372 MiB allocation with 285 MiB free, taking the container down with it. 0.85 costs ~11% of KV capacity (1.71M → 1.51M tokens, still 5.8x the full context) and no measurable throughput. Raise it only if you also lower `--max-num-seqs`.
- On unified-memory machines (DGX Spark / GB10) use [docker-compose.spark.yml](docker-compose.spark.yml) instead — select it with `COMPOSE_FILE=docker-compose.spark.yml` in `.env`. The GPU shares the 121 GB with the OS, so utilization is capped at 0.78: 0.85 measured to transiently starve the host below 5 GiB during KV-cache allocation, and 0.92 livelocks the machine hard enough to need a power cycle (disable swap so an overrun OOM-kills the engine instead of thrashing).
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights. On a 32 GB card that is not enough on its own; see [32 GB cards](#32-gb-cards-rtx-5090) for a config where the KV pool is set in bytes and `--max-num-batched-tokens` is a memory dial rather than a latency one.
- **A smaller KV dtype than fp8 is not available on these models, and would not buy much quality-adjusted capacity if it were.** `--kv-cache-dtype nvfp4` is rejected by every attention backend in v0.27.1 at this architecture's `head_size=256` (FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT), so the engine refuses to start. The `turboquant_*` 4-bit variants do run, and do add capacity — measured per 2 GiB of pool on the 27B: bf16 27,989 tokens, fp8 47,981 (1.71x), `turboquant_k8v4` 61,067 (2.18x), `turboquant_4bit_nc` 74,638 (2.67x) — but they cost decode throughput. Note the ratios fall short of the nominal ones (fp8 gives 1.71x, not 2x) because 48 of the 64 layers are gated-DeltaNet recurrent state held in fp32; it shares the same pool and does not quantize, so the dial only moves the 16 full-attention layers.

  On fidelity the three quantized dtypes are indistinguishable. With the attention backend held constant, each differs from bf16 on the same ~5.7% of teacher-forced top-1 positions, with the same ~0.015 mean top-5 KL and ~0.0017 median — and fp8 versus `turboquant_4bit_nc` differs by that same ~5.7%. That number is the size of the near-tie population, not a quality ladder: it saturates at the first perturbation and does not grow from 8 bits to 4. Resolving a real difference would need an end-to-end eval, not logprob agreement.

  **If you A/B this flag yourself, pin the backend.** The KV dtype silently selects it — bf16 picks FLASH_ATTN, fp8 picks FLASHINFER — so the naive comparison swaps kernels at the same time. Only TRITON_ATTN runs both dtypes correctly here; forcing FLASHINFER with a bf16 KV cache produces incoherent output on this model (it lost a 14k-token document entirely and hallucinated about it), while agreeing with bf16-on-FLASH_ATTN only 48% of the time. Pin it with `--attention-backend`: the `VLLM_ATTENTION_BACKEND` environment variable was removed in v0.27.1 and is now **silently ignored**.
- **`--max-num-batched-tokens` should be sized to how fast your GPU prefills, not to how much memory it has.** A chunked-prefill step blocks every decoding request until it finishes, so the worst-case decode stall is `max-num-batched-tokens ÷ prefill rate` — visible directly as p99 inter-token latency. The RTX PRO 6000 prefills the 27B at ~11k tok/s, so its 32768-token chunk costs a ~3 s stall; the GB10 prefilled at ~1.3k tok/s under that setting, so the *same value* cost it ~25 s (measured p99 ITL 28.2 s at 8k/c64). Hence 2048 on the Spark and 32768 on the RTX.

  On the Spark the smaller chunk also makes prefill itself faster, which is the bigger effect: an 8k prompt prefills in 3.64 s at 2048 versus 6.23 s at 32768, a 1.7x speedup. A 1k prompt — which fits in one chunk under either setting — is unchanged at 0.53 s, confirming the difference comes from chunk size rather than from anything else in the config. Sweeping 32768/8192/4096/2048/1024 put the knee at 2048; 1024 gains a further 7% at c64 but gives up 16% at c32.

  Two caveats. This was tuned on the **27B dense** model, which is what the compose file ships; the 35B-A3B MoE prefills ~2.7x faster and sees roughly no benefit — at 8k it measured −14/−5/+4% at c8/c32/c64 — so revisit the value if you swap models. And on the RTX PRO 6000 the sweep found **no dominant value** — 32768 is 23% faster at c8 while 8192 is 11% faster at c64 — so it keeps the larger chunk.

- **`--no-enable-prefix-caching` is pinned, and it is worth turning on only for long shared prefixes.** vLLM disables prefix caching by default for hybrid models and keeps it opt-in while the feature matures, so the flag is pinned rather than left to a default that may flip. To enable it, swap the flag for `--enable-prefix-caching`.

  The catch is block size. This model forces a **1600-token** KV block (the attention page size must be at least the mamba page size), and vLLM never reuses the last matched block, so the cacheable part of a shared prefix is `(floor(prefix ÷ 1600) − 1)` blocks. **A shared prefix below 3,200 tokens therefore hits nothing at all** — a typical few-hundred-token system prompt gains exactly zero. Measured on the RTX PRO 6000 at 8k total input, 16 prompts, c8, varying only how much of the input is a prefix shared by every request:

  | shared prefix | hit rate | median TTFT off → on | throughput off → on |
  | --- | --- | --- | --- |
  | 0 / 512 / 1,600 | 0% | no gain | no gain |
  | 3,200 | 17% | 2.59 s → 2.41 s | 235 → 264 tok/s (+12%) |
  | 6,400 | 52% | 2.60 s → 1.72 s (−34%) | 235 → 359 tok/s (+53%) |

  So it is a clear win for RAG or long-document workloads that resend a large fixed context, and no help for chat with a short system prompt. At the no-hit prefix lengths the caching-on runs also measured somewhat worse TTFT, but throughput was flat there and these are single runs, so treat that as "no benefit" rather than a quantified cost.

  Enabling it did **not** change what the model emits: greedy completions from a 8,597-token prompt were byte-identical cold versus 74% served from cache, with two cold runs matching each other first to confirm the comparison had any power. That is one prompt, not a correctness proof.

- `--max-num-seqs 64` is sized for a workstation serving a handful of concurrent clients; raise it for heavier batch serving. Note that memory headroom, not this setting, is what actually bounds usable concurrency — see the utilization note above.

## License

[MIT](LICENSE)
