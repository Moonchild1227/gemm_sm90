// Minimal pybind11 wrapper to launch the SM90 GEMM kernel and optionally copy back results.
#include "config.hpp"
#include "kernel.cu"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace py = pybind11;

// Host helper to launch the kernel. This builds a very lightweight TmaParams
// from raw pointers assuming row-major layouts for A (M x K), B (N x K),
// and D (M x N). The TMA descriptors rely on CUTLASS SM90 utilities.
template <typename T>
struct HostTensorDesc {
    T* ptr;
    int64_t dim0;
    int64_t dim1;
    int64_t stride0;
    int64_t stride1;
};

// Build a simple tensor wrapper compatible with the kernel's expectations.
template <typename T>
auto make_simple_tensor(HostTensorDesc<T> const& h) {
    using namespace cute;
    return make_tensor(make_gmem_ptr(h.ptr),
                       make_shape(Int<0>{} + h.dim0, Int<0>{} + h.dim1),
                       make_stride(Int<0>{} + h.stride0, Int<0>{} + h.stride1));
}

// Build TMA descriptors for A/B/D.
template <typename T>
auto make_tma_desc_for_AB(HostTensorDesc<T> const& h) {
    auto t = make_simple_tensor(h);
    return cutlass::make_tma_copy(t);
}

template <typename T>
auto make_tma_desc_for_D(HostTensorDesc<T> const& h) {
    auto t = make_simple_tensor(h);
    return cutlass::make_tma_store(t);
}

template <typename T>
struct DeviceBuffers {
    T* dA{nullptr};
    T* dB{nullptr};
    T* dD{nullptr};
    size_t bytesA{0}, bytesB{0}, bytesD{0};
};

using bf16 = cutlass::bfloat16_t;

DeviceBuffers<bf16> alloc_buffers(int m, int n, int k) {
    DeviceBuffers<bf16> buf;
    buf.bytesA = size_t(m) * k * sizeof(bf16);
    buf.bytesB = size_t(n) * k * sizeof(bf16);
    buf.bytesD = size_t(m) * n * sizeof(bf16);
    cudaMalloc(&buf.dA, buf.bytesA);
    cudaMalloc(&buf.dB, buf.bytesB);
    cudaMalloc(&buf.dD, buf.bytesD);
    return buf;
}

void free_buffers(DeviceBuffers<bf16>& buf) {
    if (buf.dA) cudaFree(buf.dA);
    if (buf.dB) cudaFree(buf.dB);
    if (buf.dD) cudaFree(buf.dD);
    buf = {};
}

// Fill A/B with ones (bf16) for deterministic correctness check.
void fill_ones(DeviceBuffers<bf16>& buf, int m, int n, int k) {
    std::vector<bf16> hA((size_t)m * k, bf16(1.0f));
    std::vector<bf16> hB((size_t)n * k, bf16(1.0f));
    cudaMemcpy(buf.dA, hA.data(), buf.bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(buf.dB, hB.data(), buf.bytesB, cudaMemcpyHostToDevice);
    cudaMemset(buf.dD, 0, buf.bytesD);
}

// Core runner shared by both bindings.
py::dict run_gemm(int m, int n, int k, int warmup, int iters, bool copy_back) {
    using namespace cute;

    DeviceBuffers<bf16> buf = alloc_buffers(m, n, k);
    fill_ones(buf, m, n, k);

    HostTensorDesc<bf16> hA{buf.dA, m, k, k, 1};
    HostTensorDesc<bf16> hB{buf.dB, n, k, k, 1};
    HostTensorDesc<bf16> hD{buf.dD, m, n, n, 1};

    auto tma_A = make_tma_desc_for_AB(hA);
    auto tma_B = make_tma_desc_for_AB(hB);
    auto tma_D = make_tma_desc_for_D(hD);

    auto shape_A = make_shape(Int<0>{} + m, Int<0>{} + k);
    auto shape_B = make_shape(Int<0>{} + n, Int<0>{} + k);
    auto shape_D = make_shape(Int<0>{} + m, Int<0>{} + n);

    TmaParams<decltype(shape_A), decltype(tma_A),
              decltype(shape_B), decltype(tma_B),
              decltype(shape_D), decltype(tma_D)>
        params{shape_A, tma_A, shape_B, tma_B, shape_D, tma_D};

    dim3 block(128);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M);
    size_t smem = SMEM_SIZE_BYTES;

    for (int i = 0; i < warmup; ++i) {
        gemm_sm90<<<grid, block, smem>>>(params);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        gemm_sm90<<<grid, block, smem>>>(params);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    py::dict out;
    out["avg_ms"] = ms;
    out["gflops"] = 2.0 * m * n * k / (ms * 1e6);

    if (copy_back) {
        std::vector<bf16> hDvec((size_t)m * n);
        cudaMemcpy(hDvec.data(), buf.dD, buf.bytesD, cudaMemcpyDeviceToHost);
        // Convert bf16 -> float for Python
        std::vector<float> hD_f32(hDvec.size());
        for (size_t i = 0; i < hDvec.size(); ++i) {
            hD_f32[i] = static_cast<float>(hDvec[i]);
        }
        auto vec_ptr = new std::vector<float>(std::move(hD_f32));
        auto capsule = py::capsule(vec_ptr, [](void* p) {
            delete static_cast<std::vector<float>*>(p);
        });
        py::array arr({m, n}, vec_ptr->data(), capsule);
        out["D"] = arr;
    }

    free_buffers(buf);
    return out;
}

py::dict launch_gemm(int m, int n, int k, int warmup, int iters) {
    return run_gemm(m, n, k, warmup, iters, /*copy_back=*/false);
}

py::dict launch_gemm_with_output(int m, int n, int k, int warmup, int iters) {
    return run_gemm(m, n, k, warmup, iters, /*copy_back=*/true);
}

PYBIND11_MODULE(multi_stage_gemm, m) {
    m.def("launch_gemm", &launch_gemm,
          py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("warmup") = 10, py::arg("iters") = 100,
          "Launch SM90 GEMM and return average latency and GFLOPs.");
    m.def("launch_gemm_with_output", &launch_gemm_with_output,
          py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("warmup") = 5, py::arg("iters") = 20,
          "Launch SM90 GEMM, return latency/GFLOPs and output matrix as float32.");
}

