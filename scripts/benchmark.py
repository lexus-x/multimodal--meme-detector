#!/usr/bin/env python3
"""Benchmark inference speed for all fusion strategies."""
import time
import torch
import argparse
from core.models import MultimodalClassifier


def benchmark(fusion_type, n_runs=100, batch_size=1):
    emb = torch.randn(400000, 50)
    model = MultimodalClassifier(
        embedding_matrix=emb,
        text_hidden_dim=128,
        img_hidden_dim=256,
        fusion_type=fusion_type,
    )
    model.eval()

    text = torch.randint(0, 400000, (batch_size, 50))
    image = torch.randn(batch_size, 3, 224, 224)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(text, image)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(text, image)
            times.append(time.perf_counter() - t0)

    avg_ms = sum(times) / len(times) * 1000
    p50 = sorted(times)[len(times) // 2] * 1000
    p99 = sorted(times)[int(len(times) * 0.99)] * 1000
    params = model.count_parameters()

    return {
        "fusion": fusion_type,
        "avg_ms": avg_ms,
        "p50_ms": p50,
        "p99_ms": p99,
        "params_total": params["total"],
        "params_trainable": params["trainable"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    fusions = ["early", "cross_attention", "cross_attention_deep", "gated", "bilinear"]

    print(f"\n{'─' * 75}")
    print(f"{'Fusion':<25} {'Avg ms':>8} {'P50':>8} {'P99':>8} {'Params':>12}")
    print(f"{'─' * 75}")

    for fusion in fusions:
        r = benchmark(fusion, n_runs=args.runs, batch_size=args.batch)
        print(f"{r['fusion']:<25} {r['avg_ms']:8.2f} {r['p50_ms']:8.2f} {r['p99_ms']:8.2f} {r['params_total']:>12,}")

    print(f"{'─' * 75}")
    print(f"Batch size: {args.batch} | Runs per fusion: {args.runs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")


if __name__ == "__main__":
    main()
