#pragma once

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/atom/copy_traits.hpp>

using namespace cute;
using bf16          = cutlass::bfloat16_t;
using f32           = float;
// Transaction barrier used for coordinating TMA load/store across the cluster.
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

static constexpr int BLOCK_M    = 128;
static constexpr int BLOCK_N    = 128;
static constexpr int BLOCK_K    = 64;
static constexpr int NUM_STAGES  = 2;

// ******** Smem Layout ********
using WgMmaLayoutAtom = GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>;

using SmemLayoutA = 
    decltype(tile_to_shape(WgMmaLayoutAtom{}, make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{}, Int<NUM_STAGES>{})));
using SmemLayoutB = 
    decltype(tile_to_shape(WgMmaLayoutAtom{}, make_shape(Int<BLOCK_N>{}, Int<BLOCK_K>{}, Int<NUM_STAGES>{})));

// *********** WGMMA ***********
using WGMMA_Op = GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>;

// Repeat the 64x64x16 atom to cover a 128x128x64 block.
using MMA_EU_RepeatT = Shape<Int<2>, Int<2>, Int<4>>;            // (M, N, K)
using MMA_P_T        = Tile<Int<BLOCK_M>, Int<BLOCK_N>, Int<BLOCK_K>>;
using TiledMMA       = decltype(make_tiled_mma(WGMMA_Op{}, MMA_EU_RepeatT{}));

// Smem layout for the epilogue (BF16 store). Swizzle to reduce bank conflicts.
using SmemLayoutC = decltype(composition(
    Swizzle<2, 3, 3>{},
    make_layout(
        make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}),
        make_stride(Int<BLOCK_N>{}, Int<1>{})
    )
));

// Shared memory sizing helpers.
static constexpr int SMEM_PER_STAGE = cute::max(cosize_v<SmemLayoutA>, cosize_v<SmemLayoutB>);
static constexpr int SMEM_OFFSET_C  = SMEM_PER_STAGE * NUM_STAGES;
static constexpr int SMEM_SIZE_BYTES = (SMEM_OFFSET_C + cosize_v<SmemLayoutC>) * sizeof(bf16);

// ********* Smem Plan *********
struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutA>> a_buf;
        array_aligned<bf16, cosize_v<SmemLayoutB>> b_buf;
    } buffer[NUM_STAGES];

    transac_bar_t bar_a_ready[NUM_STAGES];
    transac_bar_t bar_b_ready[NUM_STAGES];

    transac_bar_t bar_a_avail[NUM_STAGES];
    transac_bar_t bar_b_avail[NUM_STAGES];
};

// ************ TMA ************
template <
    typename    Shape_A,    typename    TMA_A,
    typename    Shape_B,    typename    TMA_B,
    typename    Shape_D,    typename    TMA_D
>
struct TmaParams {
    Shape_A     shape_A;    TMA_A       tma_A;
    Shape_B     shape_B;    TMA_B       tma_B;
    Shape_D     shape_D;    TMA_D       tma_D;
};