#!/usr/bin/env python3
"""Decode-throughput benchmark for the vLLM server this repo starts.

Reproduces the numbers in the README's Benchmarks table:

  - Greedy (temperature 0) chat completions generating 1024 tokens.
  - Decode rate is timed from the first streamed token to the last, so it
    measures decode only and excludes prefill/TTFT.
  - Single-stream figure is the mean of 3 runs; the aggregate is one batch
    of 8 concurrent requests.
  - Speculative-decode acceptance is read from the server's /metrics
    spec_decode counters, differenced across the run.

The table's fourth column, KV cache capacity, is not measured here - it is
reported once at startup, e.g.:

    docker compose logs | grep "GPU KV cache size"

Usage:
    ./bench.py                              # defaults to qwen3.6-27b
    ./bench.py --model qwen3.6-35b-a3b
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = (
    "Write a detailed technical explanation of how paged attention works in "
    "modern LLM inference engines, including block tables, KV cache layout, "
    "and how it enables continuous batching."
)


def stream_once(base_url, model, max_tokens):
    """Run one streaming completion.

    Returns (decode_seconds, tokens_after_first, total_tokens).
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    first_t = None
    last_t = None
    n_chunks = 0
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                completion_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count any token-bearing delta. With --reasoning-parser qwen3 the
            # thinking tokens arrive as `reasoning` rather than `content`, and
            # they are most of the output - missing them would time nothing.
            if (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            ):
                now = time.perf_counter()
                if first_t is None:
                    first_t = now
                last_t = now
                n_chunks += 1
    if first_t is None or last_t is None or last_t == first_t:
        raise RuntimeError("no streamed tokens observed")
    total = completion_tokens if completion_tokens else n_chunks
    # The decode window starts at the first token, so it covers the rest.
    return last_t - first_t, total - 1, total


def read_spec_metrics(base_url):
    """Snapshot the server's spec_decode counters, keyed by name[position]."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except OSError:
        return {}
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode"):
            continue
        m = re.match(r"(vllm:\S+?)\{([^}]*)\}\s+(\S+)", line)
        if not m:
            continue
        name, labels, val = m.groups()
        pos = re.search(r'position="(\d+)"', labels)
        out[name + (f"[{pos.group(1)}]" if pos else "")] = float(val)
    return out


def report_acceptance(before, after):
    """Print acceptance stats from counter deltas, if spec decode is on."""

    def delta(key):
        return after.get(key, 0.0) - before.get(key, 0.0)

    drafts = delta("vllm:spec_decode_num_drafts_total")
    draft_tokens = delta("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    if drafts <= 0 or draft_tokens <= 0:
        print("\nspeculative decoding: not enabled (no drafts recorded)")
        return

    print(
        f"\nspeculative decoding: {draft_tokens / drafts:.0f} tokens drafted per step"
    )
    print(f"  acceptance rate:  {accepted / draft_tokens * 100:5.1f}%")
    print(f"  accepted/draft:   {accepted / drafts:5.2f}")
    positions = []
    for i in range(64):
        key = f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]"
        if key not in after:
            break
        positions.append(delta(key) / drafts)
    if positions:
        joined = "  ".join(f"p{i}={v:.3f}" for i, v in enumerate(positions))
        print(f"  per-position:     {joined}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.label:
        print(f"=== {args.label} ===")

    # Warm up the server so the first timed run is not paying for a cold
    # sampler/CUDA-graph path. Not counted in any figure below.
    print("warmup...", flush=True)
    stream_once(args.base_url, args.model, 128)

    metrics_before = read_spec_metrics(args.base_url)

    print(f"single-stream: {args.runs} runs x {args.max_tokens} tokens", flush=True)
    rates = []
    for i in range(args.runs):
        secs, after_first, total = stream_once(
            args.base_url, args.model, args.max_tokens
        )
        rates.append(after_first / secs)
        print(
            f"  run {i + 1}: {rates[-1]:7.1f} tok/s  ({total} tokens in {secs:.2f}s decode)",
            flush=True,
        )
    print(f"  mean:  {statistics.mean(rates):7.1f} tok/s")

    print(
        f"\nconcurrent: {args.concurrency} requests x {args.max_tokens} tokens",
        flush=True,
    )
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(stream_once, args.base_url, args.model, args.max_tokens)
            for _ in range(args.concurrency)
        ]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    total_tokens = sum(r[2] for r in results)
    per_req = statistics.mean(r[1] / r[0] for r in results)
    print(
        f"  aggregate: {total_tokens / wall:7.1f} tok/s  ({total_tokens} tokens, {wall:.2f}s wall)"
    )
    print(f"  per-request mean: {per_req:7.1f} tok/s")

    report_acceptance(metrics_before, read_spec_metrics(args.base_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
