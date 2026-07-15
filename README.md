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

The compose file targets Qwen3.6-27B, but any Qwen3.6 NVFP4 checkpoint works the same way. Change `--model` (and `--served-model-name`) in `docker-compose.yml`, then `docker compose up -d`.

Verified with [unsloth/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) (MoE, 3B active parameters): with only the two flags above changed, vLLM resolves the MoE architecture, loads the bundled MTP head, and picks the NVFP4 MoE fast path (`FLASHINFER_CUTLASS` backend). First boot reached healthy in ~7 minutes including the cold weight download, within the healthcheck's 10-minute allowance. See [Benchmarks](#benchmarks) for how it performs.

## Benchmarks

Measured on a single RTX PRO 6000 Blackwell (96 GB) with this compose file as-is (vLLM v0.25.0, MTP speculative decoding enabled): greedy chat completions generating 1024 tokens, decode rate timed from first to last token. Single-stream is the mean of 3 runs; the aggregate is one batch of 8 concurrent requests.

| Model | Single-stream decode | 8 concurrent, aggregate | MTP acceptance | KV cache capacity |
| --- | --- | --- | --- | --- |
| [Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 113 tok/s | 761 tok/s | 73% | 1.7M tokens |
| [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4) | 287 tok/s | 1,240 tok/s | 71% | 4.9M tokens |

The MoE's 3B active parameters make it ~2.5x faster per stream than the 27B dense model, and its smaller KV footprint nearly triples cache capacity at the same 262k context. Speculative-decode acceptance is prompt-dependent; expect a few points of variance either way.

## Tuning

- `--gpu-memory-utilization 0.92` leaves headroom for CUDA graph capture; pushing it higher can OOM after the KV cache is allocated.
- On GPUs with less memory, lower `--max-model-len` first — the full 262k context is the main memory consumer after the weights.
- `--max-num-seqs 64` and `--max-num-batched-tokens 32768` are sized for a workstation serving a handful of concurrent clients; raise them for heavier batch serving.

## License

[MIT](LICENSE)
