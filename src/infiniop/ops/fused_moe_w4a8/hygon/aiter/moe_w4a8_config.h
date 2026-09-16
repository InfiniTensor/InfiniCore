// SPDX-License-Identifier: MIT
// Adapted from ROCm/AITER for InfiniCore's Hygon backend.

#ifndef MOE_W4A8_CONFIGS_HIP_H
#define MOE_W4A8_CONFIGS_HIP_H

#include <functional>
#include <hip/hip_runtime.h>
#include <unordered_map>
// MOE prefill和decode的分界线
#define MOE_THRESHOLD 1024

///////////////////////////////////////////////////////////////////////Gemm params///////////////////////////////////////////////////////////////////////////////////////

// GemmParams结构体定义
template <typename T, typename T_hidden = bhalf_t>
struct GemmParams_w4a8 {
    GemmParams_w4a8(const T *ptr_A,
                    const T *ptr_B0,
                    T_hidden *ptr_C,
                    float *ptr_A_scale,
                    float *ptr_B_scale,
                    const float *topk_weights,
                    const int32_t *sorted_token_ids,
                    const int32_t *expert_ids,
                    const int32_t num_tokens_post_pad,
                    const int32_t *num_tokens_post_pad_ptr,
                    uint32_t size_m,
                    uint32_t size_n,
                    uint32_t size_k,
                    uint32_t stride_asm,
                    uint32_t stride_ask,
                    uint32_t stride_bse,
                    uint32_t stride_bsn,
                    uint32_t stride_bsk,
                    uint32_t sorted_token_lens,
                    uint32_t top_k,
                    uint32_t real_topk,
                    bool is_marlin,
                    hipStream_t stream)
        : ptr_A(ptr_A),
          ptr_B0(ptr_B0),
          ptr_C(ptr_C),
          ptr_A_scale(ptr_A_scale),
          ptr_B_scale(ptr_B_scale),
          topk_weights(topk_weights),
          sorted_token_ids(sorted_token_ids),
          expert_ids(expert_ids),
          num_tokens_post_pad(num_tokens_post_pad),
          num_tokens_post_pad_ptr(num_tokens_post_pad_ptr),
          size_m(size_m),
          size_n(size_n),
          size_k(size_k),
          stride_asm(stride_asm),
          stride_ask(stride_ask),
          stride_bse(stride_bse),
          stride_bsn(stride_bsn),
          stride_bsk(stride_bsk),
          sorted_token_lens(sorted_token_lens),
          top_k(top_k),
          real_topk(real_topk),
          is_marlin(is_marlin),
          stream(stream) {}

    const T *ptr_A;                         // input
    const T *ptr_B0;                        // weight
    T_hidden *ptr_C;                        // output
    float *ptr_A_scale;                     // input scale
    float *ptr_B_scale;                     // weight scale
    const float *topk_weights;              // topk weights
    const int32_t *sorted_token_ids;        // sorted token ids
    const int32_t *expert_ids;              // expert ids
    const int32_t num_tokens_post_pad;      // num tokens after padding
    const int32_t *num_tokens_post_pad_ptr; // num tokens after padding
    uint32_t size_m;                        // input size m
    uint32_t size_n;                        // output size n
    uint32_t size_k;                        // input size k
    uint32_t stride_asm;                    // input scale stride m
    uint32_t stride_ask;                    // input scale stride k
    uint32_t stride_bse;                    // weight scale stride expert
    uint32_t stride_bsn;                    // weight scale stride n
    uint32_t stride_bsk;                    // weight scale stride k
    uint32_t sorted_token_lens;             // sorted token length
    uint32_t top_k;                         // top k
    uint32_t real_topk;                     // real_topk
    bool is_marlin;                         // 是否使用weight重排
    hipStream_t stream;                     // caller-owned execution stream
};

///////////////////////////////////////////////////////////////////////launch_moe_function///////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_decode(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_decode(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1N256(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1N384(const GemmParams_w4a8<T, T_hidden> &params);

template <int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1Nloop(const GemmParams_w4a8<T, T_hidden> &params);

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_K192(const GemmParams_w4a8<T, T_hidden> &params);

template <int FIXED_SIZE_K, int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_fixed_k(const GemmParams_w4a8<T, T_hidden> &params);

template <typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_general(
    const GemmParams_w4a8<T, T_hidden> &params,
    uint32_t block_size_m,
    uint32_t block_size_n,
    uint32_t block_size_k,
    uint32_t n_loop);

template <typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_general(
    const GemmParams_w4a8<T, T_hidden> &params,
    uint32_t block_size_m,
    uint32_t block_size_n,
    uint32_t block_size_k,
    uint32_t n_loop);

///////////////////////////////////////////////////////////////////////prefill_config///////////////////////////////////////////////////////////////////////////////////////
// 创建kernel映射
template <typename scalar_t>
using KernelFunc_w4a8 = std::function<void(const GemmParams_w4a8<char, scalar_t> &)>;

#define W4A8_GEMM1_NLOOP_BN128_ENTRY(MODE, BM, BK, N_LOOP)                                                                                            \
    {                                                                                                                                                 \
        MODE, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill_GEMM1Nloop<N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); } \
    }

#define W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY(BM, BK, N_LOOP) \
    W4A8_GEMM1_NLOOP_BN128_ENTRY(310000 + (BK)*1000 + (N_LOOP)*100 + (BM), BM, BK, N_LOOP)

#define W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM(BM) \
    W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY(BM, 64, 1), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY(BM, 64, 2), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY(BM, 64, 3), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY(BM, 64, 4)

#define W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES \
    W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM(16), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM(32), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM(48), W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM(64)

#define W4A8_GEMM1_N384_ALIAS_ENTRIES                                                               \
    W4A8_GEMM1_NLOOP_BN128_ENTRY(31316, 16, 64, 3), W4A8_GEMM1_NLOOP_BN128_ENTRY(31332, 32, 64, 3), \
        W4A8_GEMM1_NLOOP_BN128_ENTRY(31348, 48, 64, 3), W4A8_GEMM1_NLOOP_BN128_ENTRY(31364, 64, 64, 3)

#define W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(MODE, BM, BK, N_LOOP)                                                                                          \
    {                                                                                                                                                  \
        MODE, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill_fixed_k<0, N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); } \
    }

#define W4A8_GEMM2_DOWN_K192_FASTPATH_ENTRY(MODE, BM, BK, N_LOOP)                                                                                        \
    {                                                                                                                                                    \
        MODE, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill_fixed_k<192, N_LOOP, BM, 128, BK, BM, 32, BK, 2>(p); } \
    }

#define W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY(BM, BK, N_LOOP) \
    W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(320000 + (BK)*1000 + (N_LOOP)*100 + (BM), BM, BK, N_LOOP)

#define W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY(BM, BK, N_LOOP) \
    W4A8_GEMM2_DOWN_K192_FASTPATH_ENTRY(330000 + (BK)*1000 + (N_LOOP)*100 + (BM), BM, BK, N_LOOP)

#define W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM(BM)                                             \
    W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY(BM, 64, 1), W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY(BM, 64, 2), \
        W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY(BM, 64, 3), W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY(BM, 64, 4)

#define W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM(BM)                                                 \
    W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY(BM, 64, 1), W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY(BM, 64, 2), \
        W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY(BM, 64, 3), W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY(BM, 64, 4)

#define W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES                                                            \
    W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM(16), W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM(32), \
        W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM(48), W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM(64)

#define W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES                                                                \
    W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM(16), W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM(32), \
        W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM(48), W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM(64)

#define W4A8_GEMM2_DOWN_RUNTIME_K_NL4_ALIAS_ENTRIES                                                       \
    W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(41316, 16, 64, 4), W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(41332, 32, 64, 4), \
        W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(41348, 48, 64, 4), W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY(41364, 64, 64, 4)

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a8<scalar_t>> kernel_maps_gemm1_decode_w4a8 = {
    {121, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 32, 512, 16, 32, 128, 4>(p); }},
    {122, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 128, 64, 16, 32, 64, 4>(p); }},

    {123, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 32, 64, 16, 32, 64, 4>(p); }},

    {124, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 32, 512, 16, 32, 128, 2>(p); }},
    {125, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {126, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 32, 256, 16, 32, 64, 2>(p); }},
    {127, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_decode<16, 32, 256, 16, 32, 64, 4>(p); }},
    W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES,
    W4A8_GEMM1_N384_ALIAS_ENTRIES,

};

// gemm2
template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a8<scalar_t>> kernel_maps_gemm2_decode_w4a8 = {

    {32, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_decode<16, 128, 64, 16, 32, 64, 2>(p); }},
    {86, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_decode<32, 128, 64, 32, 32, 64, 2>(p); }},
    {118, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_decode<48, 128, 64, 48, 32, 64, 2>(p); }},
    {166, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_decode<64, 128, 64, 64, 32, 64, 2>(p); }},
    W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES,
    W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES,
    W4A8_GEMM2_DOWN_RUNTIME_K_NL4_ALIAS_ENTRIES,

};

///////////////////////////////////////////////////////////////////////decode_config///////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a8<scalar_t>> kernel_maps_gemm1_prefill_w4a8 = {
    {16, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill<16, 128, 64, 16, 32, 64, 2>(p); }},
    {53, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill<32, 128, 64, 32, 32, 64, 2>(p); }},
    {160, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill<64, 128, 64, 64, 32, 64, 2>(p); }},
    {290, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill<48, 128, 64, 48, 32, 64, 2>(p); }},
    W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES,
    W4A8_GEMM1_N384_ALIAS_ENTRIES,

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a8<scalar_t>> kernel_maps_gemm1_prefill_w4a8_gemm1n256 = {
    {16, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill_GEMM1N256<16, 128, 64, 16, 32, 64, 2>(p); }},
    {53, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill_GEMM1N256<32, 128, 64, 32, 32, 64, 2>(p); }},
    {160, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill_GEMM1N256<64, 128, 64, 64, 32, 64, 2>(p); }},
    {290, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_first_stage_prefill_GEMM1N256<48, 128, 64, 48, 32, 64, 2>(p); }},
    W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES,
    W4A8_GEMM1_N384_ALIAS_ENTRIES,

};

template <typename scalar_t>
static std::unordered_map<int, KernelFunc_w4a8<scalar_t>> kernel_maps_gemm2_prefill_w4a8 = {

    {32, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill<16, 128, 64, 16, 32, 64, 2>(p); }},
    {86, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill<32, 128, 64, 32, 32, 64, 2>(p); }},
    {118, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill<48, 128, 64, 48, 32, 64, 2>(p); }},
    {166, [](const GemmParams_w4a8<char, scalar_t> &p) { launch_moe_w4a8_second_stage_prefill<64, 128, 64, 64, 32, 64, 2>(p); }},
    W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES,
    W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES,
    W4A8_GEMM2_DOWN_RUNTIME_K_NL4_ALIAS_ENTRIES,

};

#undef W4A8_GEMM1_NLOOP_BN128_ENTRY
#undef W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRY
#undef W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES_FOR_BM
#undef W4A8_GEMM1_NLOOP_BN128_TUNE_ENTRIES
#undef W4A8_GEMM1_N384_ALIAS_ENTRIES
#undef W4A8_GEMM2_DOWN_RUNTIME_K_ENTRY
#undef W4A8_GEMM2_DOWN_K192_FASTPATH_ENTRY
#undef W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRY
#undef W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRY
#undef W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES_FOR_BM
#undef W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES_FOR_BM
#undef W4A8_GEMM2_DOWN_RUNTIME_K_TUNE_ENTRIES
#undef W4A8_GEMM2_DOWN_K192_FASTPATH_TUNE_ENTRIES
#undef W4A8_GEMM2_DOWN_RUNTIME_K_NL4_ALIAS_ENTRIES

#endif
