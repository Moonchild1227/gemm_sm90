import json
import numpy as np

import multi_stage_gemm as msg


def check_correctness(m, n, k, warmup=2, iters=5, atol=1e-1, rtol=1e-1):
    # A/B are filled with ones in the binding; reference is k.
    result = msg.launch_gemm_with_output(m, n, k, warmup, iters)
    D = np.array(result["D"], copy=False)  # float32
    ref = np.full((m, n), float(k), dtype=np.float32)

    diff = D - ref
    max_abs = np.max(np.abs(diff))
    max_rel = np.max(np.abs(diff) / (np.abs(ref) + 1e-7))
    ok = max_abs <= atol or max_rel <= rtol
    return {
        "m": m,
        "n": n,
        "k": k,
        "avg_ms": result["avg_ms"],
        "gflops": result["gflops"],
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
        "pass": bool(ok),
    }


def bench_case(m, n, k, warmup=5, iters=50):
    result = msg.launch_gemm(m, n, k, warmup, iters)
    return {
        "m": m,
        "n": n,
        "k": k,
        "avg_ms": result["avg_ms"],
        "gflops": result["gflops"],
    }


def main():
    shapes_small = [
        (128, 128, 64),
        (256, 128, 64),
    ]
    shapes_perf = [
        (256, 256, 128),
        (512, 512, 128),
        (1024, 1024, 128),
    ]

    print("Correctness (A=B=1 => D=k):")
    corr_results = []
    for m, n, k in shapes_small:
        res = check_correctness(m, n, k)
        corr_results.append(res)
        print(f"  m={m} n={n} k={k} pass={res['pass']} "
              f"max_abs={res['max_abs']:.3e} max_rel={res['max_rel']:.3e} "
              f"avg_ms={res['avg_ms']:.3f} gflops={res['gflops']:.1f}")

    print("\nPerformance:")
    bench_results = []
    for m, n, k in shapes_perf:
        res = bench_case(m, n, k)
        bench_results.append(res)
        print(f"  m={m} n={n} k={k} avg_ms={res['avg_ms']:.3f} "
              f"gflops={res['gflops']:.1f}")

    summary = {
        "correctness": corr_results,
        "performance": bench_results,
    }
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

