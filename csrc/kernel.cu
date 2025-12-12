#include "config.hpp"

#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cute/algorithm/copy.hpp>

template <typename TmaParams>
__global__
__launch_bounds__(128, 1)
void
gemm_sm90 (
    __grid_constant__ const TmaParams tma_params
) 
{
    const int block_m_idx       = blockIdx.y;
    const int block_n_idx       = blockIdx.x;
    const int idx_in_warpgroup  = threadIdx.x % 128;
    const int warp_idx          = cutlass::canonical_warp_idx_sync();

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan & plan = 
        *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    
    const int k_dim = size<1>(tma_params.shape_A);
    const int ntile = k_dim / BLOCK_K;

    if (warp_idx == 0 && elect_one_sync()) {
        CUTE_UNROLL for (int i = 0; i < NUM_STAGES; ++i) {
            plan.bar_a_ready[i].init(1);
            plan.bar_b_ready[i].init(1);
            plan.bar_a_avail[i].init(1);
            plan.bar_b_avail[i].init(i);
        }
        cutlass::arch::fence_view_async_shared();
    }
    cutlass::cluster_arrive();

    // Single-WG pipeline: warp 0 issues TMA load/store, all warps compute.
    int ready_phase_a[NUM_STAGES] = {0};
    int ready_phase_b[NUM_STAGES] = {0};
    int avail_phase_a[NUM_STAGES] = {0};
    int avail_phase_b[NUM_STAGES] = {0};

    auto gA_slice = local_tile(tma_params.tma_A.get_tensor(),
                               make_tile(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
                               make_coord(block_m_idx, _));
    auto gB_slice = local_tile(tma_params.tma_B.get_tensor(),
                               make_tile(Int<BLOCK_N>{}, Int<BLOCK_K>{}),
                               make_coord(block_n_idx, _));

    Tensor rC = partition_fragment_C(TiledMMA{}, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{});
    clear(rC);

    CUTE_NO_UNROLL
    for (int itile = 0; itile < ntile + NUM_STAGES - 1; ++itile) {
        int load_buf = itile % NUM_STAGES;

        // 1) Issue TMA load for current tile
        if (itile < ntile && elect_one_sync()) {
            Tensor sA = make_tensor(make_smem_ptr(plan.buffer[load_buf].a_buf), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(plan.buffer[load_buf].b_buf), SmemLayoutB{});

            Tensor cur_gA = gA_slice(_, _, itile);
            Tensor cur_gB = gB_slice(_, _, itile);

            auto thr_tma_a = tma_params.tma_A.get_slice(_0{});
            cute::copy(
                tma_params.tma_A.with(reinterpret_cast<typename transac_bar_t::ValueType&>(plan.bar_a_ready[load_buf])),
                thr_tma_a.partition_S(cur_gA),
                thr_tma_a.partition_D(sA(_, _, load_buf))
            );
            auto thr_tma_b = tma_params.tma_B.get_slice(_0{});
            cute::copy(
                tma_params.tma_B.with(reinterpret_cast<typename transac_bar_t::ValueType&>(plan.bar_b_ready[load_buf])),
                thr_tma_b.partition_S(cur_gB),
                thr_tma_b.partition_D(sB(_, _, load_buf))
            );

            plan.bar_a_ready[load_buf].arrive_and_expect_tx(BLOCK_M * BLOCK_K * sizeof(bf16));
            plan.bar_b_ready[load_buf].arrive_and_expect_tx(BLOCK_N * BLOCK_K * sizeof(bf16));
        }

        int compute_tile = itile - (NUM_STAGES - 1);
        if (compute_tile >= 0 && compute_tile < ntile) {
            int compute_buf = compute_tile % NUM_STAGES;

            // Wait for A/B ready
            plan.bar_a_ready[compute_buf].wait(ready_phase_a[compute_buf]);
            plan.bar_b_ready[compute_buf].wait(ready_phase_b[compute_buf]);
            ready_phase_a[compute_buf] ^= 1;
            ready_phase_b[compute_buf] ^= 1;

            Tensor sA = make_tensor(make_smem_ptr(plan.buffer[compute_buf].a_buf), SmemLayoutA{});
            Tensor sB = make_tensor(make_smem_ptr(plan.buffer[compute_buf].b_buf), SmemLayoutB{});

            auto thr_mma = TiledMMA{}.get_thread_slice(idx_in_warpgroup);
            auto tAgA = thr_mma.partition_A(sA(_, _, compute_buf));
            auto tBgB = thr_mma.partition_B(sB(_, _, compute_buf));
            auto tCrC = thr_mma.partition_C(rC);

            gemm(TiledMMA{}, tCrC, tAgA, tBgB, tCrC);

            // Release buffer back to loader
            plan.bar_a_avail[compute_buf].arrive(avail_phase_a[compute_buf]++);
            plan.bar_b_avail[compute_buf].arrive(avail_phase_b[compute_buf]++);
        }

        cutlass::arch::tma_load_wait<0>();
    }

    // -------- Epilogue: rC (F32) -> Smem BF16 -> Gmem BF16 --------
    static constexpr int SMEM_SIZE_AB = SMEM_PER_STAGE * NUM_STAGES;
    bf16* Cshm_ptr = reinterpret_cast<bf16*>(plan.buffer[0].a_buf) + SMEM_SIZE_AB;
    Tensor sC = make_tensor(make_smem_ptr(Cshm_ptr), SmemLayoutC{});

    using R2SCopyOp = Copy_Atom<UniversalCopy<f32>, bf16>;
    auto r2s_tiled_copy = make_tiled_copy_C(R2SCopyOp{}, TiledMMA{});
    auto r2s_thr_copy = r2s_tiled_copy.get_slice(idx_in_warpgroup);

    cute::copy(r2s_tiled_copy,
               r2s_thr_copy.retile_S(rC),
               r2s_thr_copy.partition_D(sC));

    __syncthreads();

    auto gD_slice = local_tile(tma_params.tma_D.get_tensor(),
                               make_tile(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
                               make_coord(block_m_idx, block_n_idx));

    if (elect_one_sync()) {
        transac_bar_t bar_d_done;
        bar_d_done.init(1);
        auto thr_tma_d = tma_params.tma_D.get_slice(_0{});
        cute::copy(
            tma_params.tma_D.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar_d_done)),
            thr_tma_d.partition_S(sC),
            thr_tma_d.partition_D(gD_slice)
        );
        bar_d_done.arrive_and_expect_tx(BLOCK_M * BLOCK_N * sizeof(bf16));
    }

    cutlass::arch::tma_store_wait<0>();
    cutlass::cluster_sync();
}