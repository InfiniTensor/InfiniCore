#if defined ENABLE_NVIDIA_API

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../gptq_marlin_gemm.h"
#include "../sgl_kernel/tensor.h"
#include "gptq_marlin_gemm_nvidia.cuh"

#include "../sgl_kernel/scalar_type.hpp"

#include "../marlin/kernel.h"
#include "../marlin/marlin_template.h"

namespace device::marlin {

INFINIOP_CUDA_KERNEL MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

INFINIOP_CUDA_KERNEL permute_cols_kernel(
    int4 const *__restrict__ a_int4_ptr,
    int const *__restrict__ perm_int_ptr,
    int4 *__restrict__ out_int4_ptr,
    int size_m,
    int size_k,
    int lda,
    int block_rows) {}

#else

// For a given "a" of size [M,K] performs a permutation of the K columns based
// on the given "perm" indices.
INFINIOP_CUDA_KERNEL permute_cols_kernel(
    int4 const *__restrict__ a_int4_ptr,
    int const *__restrict__ perm_int_ptr,
    int4 *__restrict__ out_int4_ptr,
    int size_m,
    int size_k,
    int lda,
    int block_rows) {
    auto start_row = block_rows * blockIdx.x;
    int finish_row = start_row + block_rows;
    if (finish_row > size_m) {
        finish_row = size_m;
    }
    int cur_block_rows = finish_row - start_row;

    int input_row_stride = lda * sizeof(half) / 16;
    int output_row_stride = size_k * sizeof(half) / 16;

    auto permute_row = [&](int row) {
        int iters = size_k / default_threads;
        int rest = size_k % default_threads;

        int input_offset = row * input_row_stride;
        int output_offset = row * output_row_stride;

        half const *a_row_half = reinterpret_cast<half const *>(a_int4_ptr + input_offset);
        half *out_half = reinterpret_cast<half *>(out_int4_ptr + output_offset);

        int base_k = 0;

        for (int i = 0; i < iters; i++) {
            auto cur_k = base_k + threadIdx.x;
            int src_pos = perm_int_ptr[cur_k];

            out_half[cur_k] = a_row_half[src_pos];

            base_k += default_threads;
        }

        if (rest) {
            if (threadIdx.x < rest) {
                auto cur_k = base_k + threadIdx.x;
                int src_pos = perm_int_ptr[cur_k];

                out_half[cur_k] = a_row_half[src_pos];
            }
        }
    };

    for (int i = 0; i < cur_block_rows; i++) {
        int cur_row = start_row + i;
        if (cur_row < size_m) {
            permute_row(cur_row);
        }
    }
}

typedef struct
{
    int thread_k;
    int thread_n;
    int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct
{
    int blocks_per_sm;
    thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(
    thread_config_t const &th_config,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full) {
    bool cache_scales_chunk = has_act_order && !is_k_full;

    int tb_n = th_config.thread_n;
    int tb_k = th_config.thread_k;

    // Get max scale groups per thread-block
    int tb_groups;
    if (group_size == -1) {
        tb_groups = 1;
    } else if (group_size == 0) {
        tb_groups = div_ceil(tb_k, 32); // Worst case is 32 group size
    } else {
        tb_groups = div_ceil(tb_k, group_size);
    }

    if (cache_scales_chunk) {
        int load_groups = tb_groups * pipe_stages * 2; // Chunk size is 2x pipeline over dim K
        load_groups = max(load_groups, 32);            // We load at least 32 scale groups
        return load_groups * tb_n * 2;
    } else {
        int tb_scales = tb_groups * tb_n * 2;

        return tb_scales * pipe_stages;
    }
}

int get_kernel_cache_size(
    thread_config_t const &th_config,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float) {
    int pack_factor = 32 / num_bits;

    // Get B size
    int tb_k = th_config.thread_k;
    int tb_n = th_config.thread_n;
    int tb_m = thread_m_blocks * 16;
    int sh_a_size = pipe_stages * (tb_m * tb_k) * 2;
    int sh_b_size = pipe_stages * (tb_k * tb_n / pack_factor) * 4;
    int sh_red_size = tb_m * (tb_n + 8);
    int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full);
    int sh_g_idx_size = has_act_order && !is_k_full ? pipe_stages * tb_k / 4 : 0;
    int sh_zp_size = 0;
    if (has_zp) {
        if (is_zp_float) {
            sh_zp_size = sh_s_size;
        } else if (num_bits == 4) {
            sh_zp_size = sh_s_size / 4;
        } else if (num_bits == 8) {
            sh_zp_size = sh_s_size / 2;
        }
    }

    int total_size = max(sh_b_size, sh_red_size) + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

    return total_size;
}

bool is_valid_config(
    thread_config_t const &th_config,
    int thread_m_blocks,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    int has_zp,
    int is_zp_float,
    int max_shared_mem) {
    // Sanity
    if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
        return false;
    }

    // Verify K/N are divisible by thread K/N
    if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
        return false;
    }

    // Verify min for thread K/N
    if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
        return false;
    }

    // num_threads must be at least 128 (= 4 warps)
    if (th_config.num_threads < 128) {
        return false;
    }

    // Check that pipeline fits into cache
    int cache_size = get_kernel_cache_size(
        th_config,
        thread_m_blocks,
        prob_m,
        prob_n,
        prob_k,
        num_bits,
        group_size,
        has_act_order,
        is_k_full,
        has_zp,
        is_zp_float);
    return cache_size <= max_shared_mem;
}

#define _GET_IF(                                                                                                                                                                                                                                                              \
    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT)                                                                                                                                                        \
    else if (                                                                                                                                                                                                                                                                 \
        q_type == W_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && m_block_size_8 == M_BLOCK_SIZE_8 && group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS && is_zp_float == IS_ZP_FLOAT) { \
        kernel = Marlin<                                                                                                                                                                                                                                                      \
            scalar_t,                                                                                                                                                                                                                                                         \
            W_TYPE.id(),                                                                                                                                                                                                                                                      \
            NUM_THREADS,                                                                                                                                                                                                                                                      \
            THREAD_M_BLOCKS,                                                                                                                                                                                                                                                  \
            THREAD_N_BLOCKS,                                                                                                                                                                                                                                                  \
            THREAD_K_BLOCKS,                                                                                                                                                                                                                                                  \
            M_BLOCK_SIZE_8,                                                                                                                                                                                                                                                   \
            pipe_stages,                                                                                                                                                                                                                                                      \
            GROUP_BLOCKS,                                                                                                                                                                                                                                                     \
            IS_ZP_FLOAT>;                                                                                                                                                                                                                                                     \
    }

// COMMON: cases for (group_blocks in [-1, 2, 4, 8] and is_zp_float == false)
//         this is the most common cases
// BIGGROUP: cases for big group size (group_blocks in [-1, 8])
// FZP: cases for float-zero-point (is_zp_float = true)
// ACT: cases for act order case (group_blocks == 0)
// FP4: cases for nvfp4(e2m1) (group_blocks == 1)
#define COMMON_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)         \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 2, NUM_THREADS, false)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, false)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                          \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
                                                                          \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 2, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define COMMON_GET_IF(W_TYPE)              \
    COMMON_GET_IF_M1(W_TYPE, 8, 8, 256)    \
    COMMON_GET_IF_M1(W_TYPE, 8, 4, 128)    \
    COMMON_GET_IF_M1(W_TYPE, 4, 8, 128)    \
    COMMON_GET_IF_M234(W_TYPE, 16, 4, 256) \
    COMMON_GET_IF_M234(W_TYPE, 8, 4, 128)  \
    COMMON_GET_IF_M234(W_TYPE, 4, 8, 128)

#define BIGGROUP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)       \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, -1, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 8, NUM_THREADS, false)   \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)     \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)  \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, -1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 8, NUM_THREADS, false)

#define BIGGROUP_GET_IF(W_TYPE)              \
    BIGGROUP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
    BIGGROUP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
    BIGGROUP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
    BIGGROUP_GET_IF_M234(W_TYPE, 16, 4, 256) \
    BIGGROUP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
    BIGGROUP_GET_IF_M234(W_TYPE, 4, 8, 128)

#define FP4_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)          \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)         \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 1, NUM_THREADS, false)

#define FP4_GET_IF(W_TYPE)              \
    FP4_GET_IF_M1(W_TYPE, 8, 8, 256)    \
    FP4_GET_IF_M1(W_TYPE, 8, 4, 128)    \
    FP4_GET_IF_M1(W_TYPE, 4, 8, 128)    \
    FP4_GET_IF_M234(W_TYPE, 16, 4, 256) \
    FP4_GET_IF_M234(W_TYPE, 8, 4, 128)  \
    FP4_GET_IF_M234(W_TYPE, 4, 8, 128)

// We currently have 4-bit models only with group_blocks == 4
#define FZP_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)         \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)        \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 4, NUM_THREADS, true)

#define FZP_GET_IF(W_TYPE)              \
    FZP_GET_IF_M1(W_TYPE, 8, 8, 256)    \
    FZP_GET_IF_M1(W_TYPE, 8, 4, 128)    \
    FZP_GET_IF_M1(W_TYPE, 4, 8, 128)    \
    FZP_GET_IF_M234(W_TYPE, 16, 4, 256) \
    FZP_GET_IF_M234(W_TYPE, 8, 4, 128)  \
    FZP_GET_IF_M234(W_TYPE, 4, 8, 128)

// We currently have 4-bit models only with group_blocks == 4
#define ACT_GET_IF_M1(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)          \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, true, 0, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 1, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF_M234(W_TYPE, N_BLOCKS, K_BLOCKS, NUM_THREADS)         \
    _GET_IF(W_TYPE, 2, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 3, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false) \
    _GET_IF(W_TYPE, 4, N_BLOCKS, K_BLOCKS, false, 0, NUM_THREADS, false)

#define ACT_GET_IF(W_TYPE)              \
    ACT_GET_IF_M1(W_TYPE, 8, 8, 256)    \
    ACT_GET_IF_M1(W_TYPE, 8, 4, 128)    \
    ACT_GET_IF_M1(W_TYPE, 4, 8, 128)    \
    ACT_GET_IF_M234(W_TYPE, 16, 4, 256) \
    ACT_GET_IF_M234(W_TYPE, 8, 4, 128)  \
    ACT_GET_IF_M234(W_TYPE, 4, 8, 128)

template <typename scalar_t>
MarlinFuncPtr get_marlin_kernel(
    const host::ScalarType q_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    bool m_block_size_8,
    bool has_act_order,
    bool has_zp,
    int group_blocks,
    int num_threads,
    bool is_zp_float) {
    int num_bits = q_type.size_bits();
    auto kernel = MarlinDefault;
    if (false) {
    }

    COMMON_GET_IF(host::kU4)
    COMMON_GET_IF(host::kU4B8)
    COMMON_GET_IF(host::kU8B128)

    FP4_GET_IF(host::kFE2M1f)

    BIGGROUP_GET_IF(host::kFE4M3fn)

    ACT_GET_IF(host::kU4B8)
    ACT_GET_IF(host::kU8B128)

    if (std::is_same<scalar_t, half>::value) {
        if (false) {
        }
        FZP_GET_IF(host::kU4)
    }

    return kernel;
}

template <typename scalar_t>
exec_config_t determine_exec_config(
    const host::ScalarType &q_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    bool m_block_size_8,
    int num_bits,
    int group_size,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    bool is_zp_float,
    int max_shared_mem,
    int sms) {
    exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
    thread_config_t *thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
    int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                  : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

    for (int i = 0; i < thread_configs_size; i++) {
        thread_config_t th_config = thread_configs[i];

        if (!is_valid_config(
                th_config,
                thread_m_blocks,
                prob_m,
                prob_n,
                prob_k,
                num_bits,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                is_zp_float,
                max_shared_mem)) {
            continue;
        }

        int cache_size = get_kernel_cache_size(
            th_config,
            thread_m_blocks,
            prob_m,
            prob_n,
            prob_k,
            num_bits,
            group_size,
            has_act_order,
            is_k_full,
            has_zp,
            is_zp_float);

        int group_blocks = 0;
        if (!has_act_order) {
            group_blocks = group_size == -1 ? -1 : group_size / 16;
        }

        auto kernel = get_marlin_kernel<scalar_t>(
            q_type,
            thread_m_blocks,
            th_config.thread_n / 16,
            th_config.thread_k / 16,
            m_block_size_8,
            has_act_order,
            has_zp,
            group_blocks,
            th_config.num_threads,
            is_zp_float);

        if (kernel == MarlinDefault) {
            continue;
        }

        // int m_tiles = div_ceil(prob_m, thread_m_blocks * 16);
        // int n_tiles = prob_n / th_config.thread_n;
        // int k_tiles = prob_k / th_config.thread_k;

        return {1, th_config};
    }

    return exec_cfg;
}

template <typename scalar_t>
void marlin_mm(
    const void *A,
    const void *B,
    void *C,
    void *C_tmp,
    void *s,
    void *s2,
    void *zp,
    void *g_idx,
    void *perm,
    void *a_tmp,
    int prob_m,
    int prob_n,
    int prob_k,
    int lda,
    void *workspace,
    host::ScalarType const &q_type,
    bool has_act_order,
    bool is_k_full,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int thread_k_init,
    int thread_n_init,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
    if (has_zp) {
        host::RuntimeCheck(
            q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());
    } else {
        host::RuntimeCheck(
            q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,
            "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
            "has_zp = False. Got = ",
            q_type.str());
    }

    host::RuntimeCheck(
        prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

    int group_blocks = 0;
    if (has_act_order) {
        if (is_k_full) {
            host::RuntimeCheck(group_size != -1);
            group_blocks = group_size / 16;
            host::RuntimeCheck(
                prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
        } else {
            host::RuntimeCheck(group_size == 0);
            group_blocks = 0;
        }
    } else {
        if (group_size == -1) {
            group_blocks = -1;
        } else {
            group_blocks = group_size / 16;
            host::RuntimeCheck(
                prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
        }
    }

    int num_bits = q_type.size_bits();
    const int4 *A_ptr = (const int4 *)A;
    const int4 *B_ptr = (const int4 *)B;
    int4 *C_ptr = (int4 *)C;
    int4 *C_tmp_ptr = (int4 *)C_tmp;
    const int4 *s_ptr = (const int4 *)s;
    const uint16_t *s2_ptr = (const uint16_t *)s2;
    const int4 *zp_ptr = (const int4 *)zp;
    const int *g_idx_ptr = (const int *)g_idx;
    const int *perm_ptr = (const int *)perm;
    int4 *a_tmp_ptr = (int4 *)a_tmp;

    int *locks = (int *)workspace;

    if (has_act_order) {
        // Permute A columns
        int block_rows = div_ceil(prob_m, sms);
        host::LaunchKernel(sms, default_threads, stream)(
            permute_cols_kernel, A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
        A_ptr = a_tmp_ptr;
        lda = prob_k;

        // If we have a full K, then we can run the non-act-order version of Marlin
        // (since the weight rows are reordered by increasing group ids, and by
        // having a full K, we have full original groups)
        if (is_k_full) {
            has_act_order = false;
        }
    }

    int max_shared_mem = 0;
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    host::RuntimeCheck(max_shared_mem > 0);

    int max_par = 16;
    if (prob_n <= 4096) {
        max_par = 16 * 8;
    }
    int max_shared_mem_new = max_shared_mem;
    int rest_m = prob_m;
    int max_thread_m_blocks = 4;
    while (rest_m) {
        int par_count = rest_m / (max_thread_m_blocks * 16);
        if (par_count > max_par) {
            par_count = max_par;
        }
        int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

        int thread_k = thread_k_init;
        int thread_n = thread_n_init;

        int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
        int m_block_size_8 = prob_m_split <= 8;

        // Set thread config
        exec_config_t exec_cfg;
        thread_config_t thread_tfg;
        if (thread_k != -1 && thread_n != -1) {
            thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
            exec_cfg = exec_config_t{1, thread_tfg};
            host::RuntimeCheck(prob_n % thread_n == 0, "prob_n = ", prob_n, " is not divisible by thread_n = ", thread_n);
            host::RuntimeCheck(prob_k % thread_k == 0, "prob_k = ", prob_k, " is not divisible by thread_k = ", thread_k);
        } else {
            // Auto config
            exec_cfg = determine_exec_config<scalar_t>(
                q_type,
                prob_m_split,
                prob_n,
                prob_k,
                thread_m_blocks,
                m_block_size_8,
                num_bits,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                is_zp_float,
                max_shared_mem,
                sms);
            thread_tfg = exec_cfg.tb_cfg;
            if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
                max_thread_m_blocks--;
                continue;
            }
        }

        int num_threads = thread_tfg.num_threads;
        thread_k = thread_tfg.thread_k;
        thread_n = thread_tfg.thread_n;
        int blocks = sms * exec_cfg.blocks_per_sm;
        if (exec_cfg.blocks_per_sm > 1) {
            max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
        }

        int thread_k_blocks = thread_k / 16;
        int thread_n_blocks = thread_n / 16;

        host::RuntimeCheck(
            is_valid_config(
                thread_tfg,
                thread_m_blocks,
                prob_m_split,
                prob_n,
                prob_k,
                num_bits,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                is_zp_float,
                max_shared_mem_new),
            "Invalid thread config: thread_m_blocks = ",
            thread_m_blocks,
            ", thread_k = ",
            thread_tfg.thread_k,
            ", thread_n = ",
            thread_tfg.thread_n,
            ", num_threads = ",
            thread_tfg.num_threads,
            " for MKN = [",
            prob_m,
            ", ",
            prob_k,
            ", ",
            prob_n,
            "] and num_bits = ",
            num_bits,
            ", prob_m_split = ",
            prob_m_split,
            ", group_size = ",
            group_size,
            ", has_act_order = ",
            has_act_order,
            ", is_k_full = ",
            is_k_full,
            ", has_zp = ",
            has_zp,
            ", is_zp_float = ",
            is_zp_float,
            ", max_shared_mem_new = ",
            max_shared_mem_new);

        auto kernel = get_marlin_kernel<scalar_t>(
            q_type,
            thread_m_blocks,
            thread_n_blocks,
            thread_k_blocks,
            m_block_size_8,
            has_act_order,
            has_zp,
            group_blocks,
            num_threads,
            is_zp_float);

        if (kernel == MarlinDefault) {
            host::Panic(
                "Unsupported shapes: MNK = [",
                prob_m,
                ", ",
                prob_n,
                ", ",
                prob_k,
                "]",
                ", has_act_order = ",
                has_act_order,
                ", num_groups = ",
                num_groups,
                ", group_size = ",
                group_size,
                ", prob_m_split = ",
                prob_m_split,
                ", thread_m_blocks = ",
                thread_m_blocks,
                ", thread_n_blocks = ",
                thread_n_blocks,
                ", thread_k_blocks = ",
                thread_k_blocks,
                ", num_threads = ",
                num_threads,
                ", num_bits = ",
                num_bits);
        }

        host::RuntimeDeviceCheck(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new));

        bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

        host::LaunchKernel(blocks, num_threads, stream, max_shared_mem_new)(
            kernel,
            A_ptr,
            B_ptr,
            C_ptr,
            C_tmp_ptr,
            s_ptr,
            s2_ptr,
            zp_ptr,
            g_idx_ptr,
            num_groups,
            prob_m_split,
            prob_n,
            prob_k,
            lda,
            locks,
            part_use_atomic_add,
            use_fp32_reduce,
            max_shared_mem_new);

        A_ptr += prob_m_split * (lda / 8);
        C_ptr += prob_m_split * (prob_n / 8);
        rest_m -= prob_m_split;
    }
}

#endif

} // namespace device::marlin

template <typename scalar_t>
void gptq_marlin_gemm(const void *a,
                      const void *b_q_weight,
                      void *b_scales,
                      void *global_scale,
                      void *b_zeros,
                      void *g_idx,
                      void *perm,
                      void *c,
                      void *c_tmp,
                      void *a_tmp,
                      void *workspace,
                      int64_t b_q_type_id,
                      bool is_k_full,
                      bool use_atomic_add,
                      bool use_fp32_reduce,
                      bool is_zp_float,
                      int64_t size_m,
                      int64_t size_k,
                      int64_t size_n,
                      int64_t b_q_size_1,
                      int64_t a_stride0,
                      int num_groups, cudaStream_t stream)

{
    using namespace host;

    // Verify a: [M, K]
    ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
    int pack_factor = 32 / b_q_type.size_bits();
    // Verify b_q_weight: [K/tile_size, packed_N]
    RuntimeCheck(
        size_k % device::marlin::tile_size == 0,
        "size_k = ",
        size_k,
        " is not divisible by tile_size = ",
        device::marlin::tile_size);

    int64_t expected_bqw_dim0 = size_k / device::marlin::tile_size;
    RuntimeCheck(
        b_q_size_1 % device::marlin::tile_size == 0,
        "b_q_weight.size(1) = ",
        b_q_size_1,
        " is not divisible by tile_size = ",
        device::marlin::tile_size);
    int64_t actual_size_n = (b_q_size_1 / device::marlin::tile_size) * pack_factor;
    RuntimeCheck(actual_size_n == size_n, "actual_size_n must = size_n");
    // size_n = actual_size_n
    RuntimeCheck(a_stride0 % 8 == 0, "a.stride(0) must be divisible by 8");
    // Verify b_scales: [num_groups, N]
    // Early return for zero-size M
    if (size_m == 0) {
        return;
    }

    // int64_t g_idx_size = g_idx.size(0);// g_idx_size == size_k
    // int64_t perm_size = perm.size(0);// perm_size == size_k
    bool has_act_order = (g_idx != nullptr && perm != nullptr);

    // Determine has_zp from b_zeros size
    // int64_t b_zeros_size = b_zeros.size(0);
    bool has_zp = (b_zeros != nullptr);

    if (has_zp) {
        RuntimeCheck(
            b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());
    } else {
        RuntimeCheck(
            b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,
            "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "
            "has_zp = False. Got = ",
            b_q_type.str());
    }

    if (has_zp && is_zp_float) {
        RuntimeCheck(
            std::is_same<scalar_t, fp16_t>::value, "Computation type must be float16 (half) when using float zero points.");
    }

    // int64_t global_scale_size = global_scale.size(0);
    if (global_scale != nullptr) {
        RuntimeCheck(b_q_type == kFE2M1f, "global_scale can only be used for float4_e2m1f.");
    } else {
        RuntimeCheck(!(b_q_type == kFE2M1f), "the global_scale parameter must be passed for float4_e2m1f.");
    }
    // Derive group_size
    int group_size = -1;
    if (has_act_order) {
        if (is_k_full) {
            RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");
            RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
            group_size = static_cast<int>(size_k / num_groups);
        } else {
            group_size = 0;
        }
    } else {
        if (num_groups > 1) {
            RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
            group_size = static_cast<int>(size_k / num_groups);
        } else {
            group_size = -1;
        }
    }

    // Verify workspace and get device info
    RuntimeCheck(
        size_n % device::marlin::min_thread_n == 0,
        "size_n = ",
        size_n,
        ", is not divisible by min_thread_n = ",
        device::marlin::min_thread_n);

    int device_id = 0;
    int sms = -1;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id));
    // RuntimeCheck(workspace.size(0) >= sms, "workspace.size(0) = ", workspace.size(0), " is below min_workspace_size = ", sms);
    // Hardcoded defaults (auto config)
    int thread_k_init = -1;
    int thread_n_init = -1;

    // Compute c_tmp and a_tmp pointers
    // c_tmp and a_tmp are pre-allocated by caller

    device::marlin::marlin_mm<scalar_t>(
        a,
        b_q_weight,
        c,
        c_tmp,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        a_tmp,
        static_cast<int>(size_m),
        static_cast<int>(size_n),
        static_cast<int>(size_k),
        static_cast<int>(a_stride0),
        workspace,
        b_q_type,
        has_act_order,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        device_id,
        stream,
        thread_k_init,
        thread_n_init,
        sms,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float);
}

template <typename scalar_t>
infiniStatus_t gptq_marlin_gemm_kernel(void *c,
                                       const void *a,
                                       const void *b_q_weight,
                                       void *b_scales,
                                       void *global_scale,
                                       void *b_zeros,
                                       void *g_idx,
                                       void *perm,
                                       int64_t b_q_type_id,
                                       bool is_k_full,
                                       bool use_atomic_add,
                                       bool use_fp32_reduce,
                                       bool is_zp_float,
                                       int64_t size_m,
                                       int64_t size_k,
                                       int64_t size_n,
                                       int64_t b_q_size_1,
                                       int64_t a_stride0,
                                       int num_groups, void *total_buffer, cudaStream_t stream) {
    int _MAX_THREAD_N = 256;
    int max_blocks_per_sm = 1;
    float *c_tmp = nullptr;
    void *a_tmp = nullptr;
    void *workspace = nullptr;

    // 获取设备 SM 数量（只查询 1 次！）
    int dev;
    cudaGetDevice(&dev); // 获取当前设备号

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    const int sms = prop.multiProcessorCount;

    // ===================== 1. 计算每块内存大小 =====================
    size_t c_tmp_bytes = 0;
    if (use_fp32_reduce) {
        int max_m_block = ((size_m + 15) / 16) * 16;
        max_m_block = min(max_m_block, 64);
        const size_t c_elems = (size_t)sms * max_m_block * _MAX_THREAD_N;
        c_tmp_bytes = c_elems * sizeof(float);
    }

    size_t a_tmp_bytes = 0;
    bool has_act_order = false;
    if (g_idx != nullptr && perm != nullptr) {
        has_act_order = true;
    }
    if (has_act_order) {
        a_tmp_bytes = (size_t)size_m * size_k * sizeof(scalar_t);
    }

    // workspace 大小（int 类型，必须分配）
    const size_t workspace_elems = (size_t)sms * max_blocks_per_sm;
    const size_t workspace_bytes = workspace_elems * sizeof(int);

    // ===================== 2. 计算总内存大小 =====================
    const size_t total_bytes = c_tmp_bytes + a_tmp_bytes + workspace_bytes;

    // ===================== 3. 单次 cudaMalloc 分配 =====================
    if (total_bytes > 0) {
        cudaMemset(total_buffer, 0, total_bytes);
    }

    // ===================== 4. 手动切分指针（核心！） =====================
    uint8_t *ptr = reinterpret_cast<uint8_t *>(total_buffer);

    // 分配 c_tmp
    if (use_fp32_reduce && c_tmp_bytes > 0) {
        c_tmp = reinterpret_cast<float *>(ptr);
        ptr += c_tmp_bytes;
    }

    // 分配 a_tmp
    if (has_act_order && a_tmp_bytes > 0) {
        a_tmp = ptr;
        ptr += a_tmp_bytes;
    }

    // 分配 workspace
    if (workspace_bytes > 0) {
        workspace = ptr;
        ptr += workspace_bytes;
    }

    gptq_marlin_gemm<scalar_t>(
        a,
        b_q_weight,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        c,
        c_tmp,
        a_tmp,
        workspace,
        b_q_type_id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        size_m,
        size_k,
        size_n,
        b_q_size_1,
        a_stride0,
        num_groups,
        stream);
    return INFINI_STATUS_SUCCESS;
}

int getCudaDeviceSMCount() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    return prop.multiProcessorCount;
}

namespace op::gptq_marlin_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t global_scale_desc,
    infiniopTensorDescriptor_t b_zeros_desc,
    infiniopTensorDescriptor_t g_idx_desc,
    infiniopTensorDescriptor_t perm_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto result = GptqMarlinGemmInfo::create(out_desc, a_desc, b_desc, b_scales_desc, global_scale_desc, b_zeros_desc, g_idx_desc, perm_desc);

    int sms = getCudaDeviceSMCount();
    int _MAX_THREAD_N = 256;
    int max_blocks_per_sm = 1;
    int max_m_block = ((out_desc->dim(0) + 15) / 16) * 16;
    max_m_block = min(max_m_block, 64);
    const size_t c_elems = (size_t)sms * max_m_block * _MAX_THREAD_N;
    size_t c_tmp_bytes = c_elems * sizeof(float);
    size_t a_tmp_bytes = (size_t)a_desc->dim(0) * a_desc->dim(1) * infiniSizeOf(a_desc->dtype());
    const size_t workspace_elems = (size_t)sms * max_blocks_per_sm;
    const size_t workspace_bytes = workspace_elems * sizeof(int);
    size_t workspace_size = c_tmp_bytes + a_tmp_bytes + workspace_bytes;

    *desc_ptr = new Descriptor(
        workspace_size,
        new Opaque{handle->internal()},
        result.take(),
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t
Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *b_scales,
    void *global_scale,
    void *b_zeros,
    void *g_idx,
    void *perm,
    int64_t b_q_type_id,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;
    int64_t M = static_cast<int64_t>(_info.M);
    int64_t K = static_cast<int64_t>(_info.K);
    int64_t N = static_cast<int64_t>(_info.N);
    int64_t b_q_size_1 = static_cast<int64_t>(_info.b_q_size_1);
    int64_t a_stride_0 = static_cast<int64_t>(_info.a_stride_0);
    int num_groups = _info.num_groups;

#define MARLIN(TDATA) \
    gptq_marlin_gemm_kernel<TDATA>(out, a, b, b_scales, global_scale, b_zeros, g_idx, perm, b_q_type_id, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, M, K, N, b_q_size_1, a_stride_0, num_groups, workspace, stream)

    if (_info.dtype == INFINI_DTYPE_F16) {
        return MARLIN(half);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        return MARLIN(__nv_bfloat16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gptq_marlin_gemm::nvidia

#endif
