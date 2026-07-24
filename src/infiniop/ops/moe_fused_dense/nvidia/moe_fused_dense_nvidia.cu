#ifdef ENABLE_NVIDIA_API

#include "moe_fused_dense_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#ifdef ENABLE_CUTLASS_API
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#endif

namespace op::moe_fused_dense::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void *advance_workspace(uint8_t *&ptr, size_t &remaining, size_t bytes, size_t alignment = 16) {
    auto address = reinterpret_cast<uintptr_t>(ptr);
    auto aligned = align_up(address, alignment);
    auto padding = aligned - address;
    if (padding + bytes > remaining) {
        return nullptr;
    }
    ptr += padding;
    remaining -= padding;
    void *out = ptr;
    ptr += bytes;
    remaining -= bytes;
    return out;
}

size_t workspace_size(const MoeFusedDenseInfo &info) {
#ifndef ENABLE_CUTLASS_API
    (void)info;
    return 0;
#else
    const size_t pairs = info.num_tokens * info.topk;
    const size_t rows = info.max_num_tokens_padded;
    const size_t dtype_size = info.dtype == INFINI_DTYPE_F16 ? sizeof(half) : sizeof(__nv_bfloat16);
    size_t bytes = 0;
    bytes += align_up((info.num_experts + 1) * sizeof(int), 16);
    bytes += align_up((info.num_experts + 1) * sizeof(int), 16);
    bytes += align_up(pairs * sizeof(int), 16);
    bytes += align_up(rows * info.hidden_size * dtype_size, 16);
    bytes += align_up(rows * info.intermediate_size * 2 * dtype_size, 16);
    bytes += align_up(rows * info.intermediate_size * dtype_size, 16);
    bytes += align_up(rows * info.hidden_size * dtype_size, 16);
    bytes += align_up(info.num_experts * sizeof(cutlass::gemm::GemmCoord), 16);
    bytes += align_up(info.num_experts * sizeof(void *), 16) * 4;
    bytes += align_up(info.num_experts * sizeof(int64_t), 16) * 4;
    bytes += align_up(sizeof(int), 16);
    return bytes + 256;
#endif
}

template <typename T>
__device__ float to_float(T v);

template <>
__device__ float to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ T from_float(float v);

template <>
__device__ half from_float<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__global__ void exclusive_prefix_counts_kernel(const int *counts, int *offsets, int num_experts) {
    extern __shared__ int scan[];
    int tid = threadIdx.x;
    if (tid < num_experts) {
        scan[tid] = counts[tid];
    }
    if (tid >= num_experts && tid < blockDim.x) {
        scan[tid] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        int value = 0;
        if (tid >= stride) {
            value = scan[tid - stride];
        }
        __syncthreads();
        scan[tid] += value;
        __syncthreads();
    }

    if (tid == 0) {
        offsets[0] = 0;
    }
    if (tid < num_experts) {
        offsets[tid + 1] = scan[tid];
    }
}

__global__ void count_aligned_experts_kernel(const int *expert_ids,
                                             const int *num_tokens_post_padded,
                                             int *counts,
                                             int num_experts,
                                             int block_size) {
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = (*num_tokens_post_padded + block_size - 1) / block_size;
    if (block >= num_blocks) {
        return;
    }
    int expert = expert_ids[block];
    if (expert >= 0 && expert < num_experts) {
        atomicAdd(counts + expert, block_size);
    }
}

template <typename T>
__global__ void pack_hidden_aligned_kernel(const T *hidden,
                                           const int *sorted_token_ids,
                                           int *output_permutation,
                                           T *packed_hidden,
                                           int pairs,
                                           int topk,
                                           int hidden_size,
                                           int max_num_tokens_padded) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= max_num_tokens_padded) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair >= 0 && pair < pairs) {
        if (tid == 0) {
            output_permutation[pair] = row;
        }
        int token = pair / topk;
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            packed_hidden[static_cast<size_t>(row) * hidden_size + h] = hidden[static_cast<size_t>(token) * hidden_size + h];
        }
    } else {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            packed_hidden[static_cast<size_t>(row) * hidden_size + h] = from_float<T>(0.0f);
        }
    }
}

template <typename T>
__global__ void swiglu_kernel(const T *gate_up, T *activated, int rows, int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * intermediate_size;
    if (idx >= total) {
        return;
    }
    int row = idx / intermediate_size;
    int col = idx - row * intermediate_size;
    const T *base = gate_up + static_cast<size_t>(row) * intermediate_size * 2;
    float gate = to_float<T>(base[col]);
    float up = to_float<T>(base[intermediate_size + col]);
    float silu = gate / (1.0f + expf(-gate));
    activated[idx] = from_float<T>(up * silu);
}

template <typename T>
__global__ void apply_shuffle_mul_sum_kernel(const T *__restrict__ expert_out,
                                             T *__restrict__ out,
                                             const int *__restrict__ output_permutation,
                                             const float *__restrict__ topk_weights,
                                             int num_tokens,
                                             int topk,
                                             int hidden_size) {
    int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) {
            int pair = token * topk + k;
            int src_row = output_permutation[pair];
            if (src_row >= 0) {
                sum += to_float<T>(expert_out[static_cast<size_t>(src_row) * hidden_size + h]) * topk_weights[pair];
            }
        }
        out[static_cast<size_t>(token) * hidden_size + h] = from_float<T>(sum);
    }
}

#ifdef ENABLE_CUTLASS_API
template <typename T>
__global__ void setup_prefill_gemm1_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                   void **ptr_a,
                                                   void **ptr_b,
                                                   void **ptr_c,
                                                   void **ptr_d,
                                                   int64_t *lda,
                                                   int64_t *ldb,
                                                   int64_t *ldc,
                                                   int64_t *ldd,
                                                   int *active_count,
                                                   const int *counts,
                                                   const int *offsets,
                                                   const T *packed_hidden,
                                                   const T *w13,
                                                   T *gate_up,
                                                   int num_experts,
                                                   int hidden_size,
                                                   int intermediate_size) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    if (m <= 0) {
        return;
    }
    int idx = atomicAdd(active_count, 1);
    int off = offsets[expert];
    problems[idx] = cutlass::gemm::GemmCoord(m, intermediate_size * 2, hidden_size);
    ptr_a[idx] = const_cast<T *>(packed_hidden + static_cast<size_t>(off) * hidden_size);
    ptr_b[idx] = const_cast<T *>(w13 + static_cast<size_t>(expert) * intermediate_size * 2 * hidden_size);
    ptr_c[idx] = gate_up + static_cast<size_t>(off) * intermediate_size * 2;
    ptr_d[idx] = gate_up + static_cast<size_t>(off) * intermediate_size * 2;
    lda[idx] = hidden_size;
    ldb[idx] = hidden_size;
    ldc[idx] = intermediate_size * 2;
    ldd[idx] = intermediate_size * 2;
}

template <typename T>
__global__ void setup_prefill_gemm2_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                   void **ptr_a,
                                                   void **ptr_b,
                                                   void **ptr_c,
                                                   void **ptr_d,
                                                   int64_t *lda,
                                                   int64_t *ldb,
                                                   int64_t *ldc,
                                                   int64_t *ldd,
                                                   int *active_count,
                                                   const int *counts,
                                                   const int *offsets,
                                                   const T *activated,
                                                   const T *w2,
                                                   T *expert_out,
                                                   int num_experts,
                                                   int hidden_size,
                                                   int intermediate_size) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    if (m <= 0) {
        return;
    }
    int idx = atomicAdd(active_count, 1);
    int off = offsets[expert];
    problems[idx] = cutlass::gemm::GemmCoord(m, hidden_size, intermediate_size);
    ptr_a[idx] = const_cast<T *>(activated + static_cast<size_t>(off) * intermediate_size);
    ptr_b[idx] = const_cast<T *>(w2 + static_cast<size_t>(expert) * hidden_size * intermediate_size);
    ptr_c[idx] = expert_out + static_cast<size_t>(off) * hidden_size;
    ptr_d[idx] = expert_out + static_cast<size_t>(off) * hidden_size;
    lda[idx] = intermediate_size;
    ldb[idx] = intermediate_size;
    ldc[idx] = hidden_size;
    ldd[idx] = hidden_size;
}

template <typename T>
__global__ void setup_decode_gemm1_aligned_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                          void **ptr_a,
                                                          void **ptr_b,
                                                          void **ptr_c,
                                                          void **ptr_d,
                                                          int64_t *lda,
                                                          int64_t *ldb,
                                                          int64_t *ldc,
                                                          int64_t *ldd,
                                                          int *active_count,
                                                          int *output_permutation,
                                                          const T *hidden,
                                                          const T *w13,
                                                          T *gate_up,
                                                          const int *sorted_token_ids,
                                                          const int *expert_ids,
                                                          const int *num_tokens_post_padded,
                                                          int pairs,
                                                          int topk,
                                                          int num_experts,
                                                          int hidden_size,
                                                          int intermediate_size,
                                                          int block_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = *num_tokens_post_padded;
    if (row >= total_rows) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair < 0 || pair >= pairs) {
        return;
    }
    int expert = expert_ids[row / block_size];
    if (expert < 0 || expert >= num_experts) {
        return;
    }

    int idx = atomicAdd(active_count, 1);
    int token = pair / topk;
    output_permutation[pair] = idx;
    problems[idx] = cutlass::gemm::GemmCoord(1, intermediate_size * 2, hidden_size);
    ptr_a[idx] = const_cast<T *>(hidden + static_cast<size_t>(token) * hidden_size);
    ptr_b[idx] = const_cast<T *>(w13 + static_cast<size_t>(expert) * intermediate_size * 2 * hidden_size);
    ptr_c[idx] = gate_up + static_cast<size_t>(idx) * intermediate_size * 2;
    ptr_d[idx] = gate_up + static_cast<size_t>(idx) * intermediate_size * 2;
    lda[idx] = hidden_size;
    ldb[idx] = hidden_size;
    ldc[idx] = intermediate_size * 2;
    ldd[idx] = intermediate_size * 2;
}

template <typename T>
__global__ void setup_decode_gemm2_aligned_compact_kernel(cutlass::gemm::GemmCoord *problems,
                                                          void **ptr_a,
                                                          void **ptr_b,
                                                          void **ptr_c,
                                                          void **ptr_d,
                                                          int64_t *lda,
                                                          int64_t *ldb,
                                                          int64_t *ldc,
                                                          int64_t *ldd,
                                                          const T *activated,
                                                          const T *w2,
                                                          T *expert_out,
                                                          const int *sorted_token_ids,
                                                          const int *expert_ids,
                                                          const int *num_tokens_post_padded,
                                                          const int *output_permutation,
                                                          int pairs,
                                                          int num_experts,
                                                          int hidden_size,
                                                          int intermediate_size,
                                                          int block_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = *num_tokens_post_padded;
    if (row >= total_rows) {
        return;
    }
    int pair = sorted_token_ids[row];
    if (pair < 0 || pair >= pairs) {
        return;
    }
    int expert = expert_ids[row / block_size];
    if (expert < 0 || expert >= num_experts) {
        return;
    }
    int idx = output_permutation[pair];
    if (idx < 0) {
        return;
    }
    problems[idx] = cutlass::gemm::GemmCoord(1, hidden_size, intermediate_size);
    ptr_a[idx] = const_cast<T *>(activated + static_cast<size_t>(idx) * intermediate_size);
    ptr_b[idx] = const_cast<T *>(w2 + static_cast<size_t>(expert) * hidden_size * intermediate_size);
    ptr_c[idx] = expert_out + static_cast<size_t>(idx) * hidden_size;
    ptr_d[idx] = expert_out + static_cast<size_t>(idx) * hidden_size;
    lda[idx] = intermediate_size;
    ldb[idx] = intermediate_size;
    ldc[idx] = hidden_size;
    ldd[idx] = hidden_size;
}

template <typename CutlassT>
infiniStatus_t launch_cutlass_gemm_grouped_device_meta(int problem_count,
                                                       cutlass::gemm::GemmCoord *d_problems,
                                                       void **d_ptr_a,
                                                       void **d_ptr_b,
                                                       void **d_ptr_c,
                                                       void **d_ptr_d,
                                                       int64_t *d_lda,
                                                       int64_t *d_ldb,
                                                       int64_t *d_ldc,
                                                       int64_t *d_ldd,
                                                       cudaStream_t stream) {
    if (problem_count == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    using Element = CutlassT;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using OutputOp = cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>;
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Element,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        8,
        Element,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        8,
        Element,
        LayoutC,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        OutputOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        4,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    const int threadblock_count = std::min(Gemm::sufficient(), problem_count);
    if (threadblock_count <= 0) {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    typename Gemm::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);
    typename Gemm::Arguments args(
        d_problems,
        problem_count,
        threadblock_count,
        epilogue_op,
        reinterpret_cast<Element **>(d_ptr_a),
        reinterpret_cast<Element **>(d_ptr_b),
        reinterpret_cast<Element **>(d_ptr_c),
        reinterpret_cast<Element **>(d_ptr_d),
        d_lda,
        d_ldb,
        d_ldc,
        d_ldd,
        nullptr);

    Gemm gemm;
    auto status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}
#endif

template <typename T, typename CutlassT>
infiniStatus_t calculate_typed(const MoeFusedDenseInfo &info,
                               void *workspace,
                               size_t workspace_size,
                               void *output,
                               const void *hidden_states,
                               const void *w13,
                               const void *w2,
                               const void *topk_weights,
                               const void *topk_ids,
                               const void *sorted_token_ids,
                               const void *expert_ids,
                               const void *num_tokens_post_padded,
                               cudaStream_t stream) {
#ifndef ENABLE_CUTLASS_API
    (void)info;
    (void)workspace;
    (void)workspace_size;
    (void)output;
    (void)hidden_states;
    (void)w13;
    (void)w2;
    (void)topk_weights;
    (void)topk_ids;
    (void)sorted_token_ids;
    (void)expert_ids;
    (void)num_tokens_post_padded;
    (void)stream;
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#else
    const int num_tokens = static_cast<int>(info.num_tokens);
    const int hidden_size = static_cast<int>(info.hidden_size);
    const int num_experts = static_cast<int>(info.num_experts);
    const int intermediate_size = static_cast<int>(info.intermediate_size);
    const int topk = static_cast<int>(info.topk);
    const int pairs = num_tokens * topk;
    const int max_num_tokens_padded = static_cast<int>(info.max_num_tokens_padded);
    const int block_size = static_cast<int>((info.max_num_tokens_padded + info.max_num_blocks - 1) / info.max_num_blocks);

    uint8_t *ptr = reinterpret_cast<uint8_t *>(workspace);
    size_t remaining = workspace_size;
    auto counts = reinterpret_cast<int *>(advance_workspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto offsets = reinterpret_cast<int *>(advance_workspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto output_permutation = reinterpret_cast<int *>(advance_workspace(ptr, remaining, pairs * sizeof(int)));
    auto packed_hidden = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * hidden_size * sizeof(T)));
    auto gate_up = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * intermediate_size * 2 * sizeof(T)));
    auto activated = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * intermediate_size * sizeof(T)));
    auto expert_out = reinterpret_cast<T *>(advance_workspace(ptr, remaining, static_cast<size_t>(max_num_tokens_padded) * hidden_size * sizeof(T)));
    auto grouped_problems = reinterpret_cast<cutlass::gemm::GemmCoord *>(
        advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(cutlass::gemm::GemmCoord), alignof(cutlass::gemm::GemmCoord)));
    auto grouped_ptr_a = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_b = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_c = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_d = reinterpret_cast<void **>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_lda = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldb = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldc = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldd = reinterpret_cast<int64_t *>(advance_workspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto active_count = reinterpret_cast<int *>(advance_workspace(ptr, remaining, sizeof(int), alignof(int)));

    if (!counts || !offsets || !output_permutation || !packed_hidden || !gate_up || !activated || !expert_out || !grouped_problems || !grouped_ptr_a || !grouped_ptr_b || !grouped_ptr_c || !grouped_ptr_d || !grouped_lda || !grouped_ldb || !grouped_ldc || !grouped_ldd || !active_count) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto w13_t = reinterpret_cast<const T *>(w13);
    auto w2_t = reinterpret_cast<const T *>(w2);
    if (num_tokens == 1 && topk <= num_experts) {
        cudaMemsetAsync(output_permutation, 0xff, pairs * sizeof(int), stream);
        cudaMemsetAsync(active_count, 0, sizeof(int), stream);
        setup_decode_gemm1_aligned_compact_kernel<T><<<(max_num_tokens_padded + 255) / 256, 256, 0, stream>>>(
            grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
            grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, active_count, output_permutation,
            reinterpret_cast<const T *>(hidden_states), w13_t, gate_up,
            reinterpret_cast<const int *>(sorted_token_ids),
            reinterpret_cast<const int *>(expert_ids),
            reinterpret_cast<const int *>(num_tokens_post_padded),
            pairs, topk, num_experts, hidden_size, intermediate_size, block_size);

        int host_active_count = 0;
        if (cudaMemcpyAsync(&host_active_count, active_count, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        if (host_active_count > 0) {
            CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
                host_active_count, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
                grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));

            swiglu_kernel<T><<<(host_active_count * intermediate_size + 255) / 256, 256, 0, stream>>>(gate_up, activated, host_active_count, intermediate_size);

            setup_decode_gemm2_aligned_compact_kernel<T><<<(max_num_tokens_padded + 255) / 256, 256, 0, stream>>>(
                grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
                grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, activated, w2_t, expert_out,
                reinterpret_cast<const int *>(sorted_token_ids),
                reinterpret_cast<const int *>(expert_ids),
                reinterpret_cast<const int *>(num_tokens_post_padded),
                output_permutation, pairs, num_experts, hidden_size, intermediate_size, block_size);
            CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
                host_active_count, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
                grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));
        }

        apply_shuffle_mul_sum_kernel<T><<<1, std::min(hidden_size, 1024), 0, stream>>>(
            expert_out, reinterpret_cast<T *>(output), output_permutation,
            reinterpret_cast<const float *>(topk_weights), 1, topk, hidden_size);
        return INFINI_STATUS_SUCCESS;
    }

    cudaMemsetAsync(output_permutation, 0xff, pairs * sizeof(int), stream);
    cudaMemsetAsync(counts, 0, (num_experts + 1) * sizeof(int), stream);
    count_aligned_experts_kernel<<<(info.max_num_blocks + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const int *>(expert_ids),
        reinterpret_cast<const int *>(num_tokens_post_padded),
        counts,
        num_experts,
        block_size);

    int scan_threads = 1;
    while (scan_threads < num_experts) {
        scan_threads <<= 1;
    }
    scan_threads = std::max(32, scan_threads);
    exclusive_prefix_counts_kernel<<<1, scan_threads, scan_threads * sizeof(int), stream>>>(counts, offsets, num_experts);

    pack_hidden_aligned_kernel<T><<<max_num_tokens_padded, 256, 0, stream>>>(
        reinterpret_cast<const T *>(hidden_states),
        reinterpret_cast<const int *>(sorted_token_ids),
        output_permutation,
        packed_hidden,
        pairs,
        topk,
        hidden_size,
        max_num_tokens_padded);

    cudaMemsetAsync(active_count, 0, sizeof(int), stream);
    setup_prefill_gemm1_compact_kernel<T><<<(num_experts + 255) / 256, 256, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, active_count, counts, offsets, packed_hidden,
        w13_t, gate_up, num_experts, hidden_size, intermediate_size);

    int host_active_count = 0;
    if (cudaMemcpyAsync(&host_active_count, active_count, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
        host_active_count, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));

    swiglu_kernel<T><<<(max_num_tokens_padded * intermediate_size + 255) / 256, 256, 0, stream>>>(
        gate_up, activated, max_num_tokens_padded, intermediate_size);

    cudaMemsetAsync(active_count, 0, sizeof(int), stream);
    setup_prefill_gemm2_compact_kernel<T><<<(num_experts + 255) / 256, 256, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, active_count, counts, offsets, activated,
        w2_t, expert_out, num_experts, hidden_size, intermediate_size);
    CHECK_STATUS(launch_cutlass_gemm_grouped_device_meta<CutlassT>(
        host_active_count, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));

    apply_shuffle_mul_sum_kernel<T><<<num_tokens, std::min(hidden_size, 1024), 0, stream>>>(
        expert_out, reinterpret_cast<T *>(output), output_permutation,
        reinterpret_cast<const float *>(topk_weights), num_tokens, topk, hidden_size);
    return INFINI_STATUS_SUCCESS;
#endif
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t w13_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc) {
    auto result = MoeFusedDenseInfo::create(
        output_desc,
        hidden_states_desc,
        w13_desc,
        w2_desc,
        topk_weights_desc,
        topk_ids_desc,
        sorted_token_ids_desc,
        expert_ids_desc,
        num_tokens_post_padded_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size(info),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *hidden_states,
    const void *w13,
    const void *w2,
    const void *topk_weights,
    const void *topk_ids,
    const void *sorted_token_ids,
    const void *expert_ids,
    const void *num_tokens_post_padded,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
#ifdef ENABLE_CUTLASS_API
    case INFINI_DTYPE_F16:
        return calculate_typed<half, cutlass::half_t>(
            _info, workspace, workspace_size, output, hidden_states, w13, w2,
            topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            cuda_stream);
    case INFINI_DTYPE_BF16:
        return calculate_typed<__nv_bfloat16, cutlass::bfloat16_t>(
            _info, workspace, workspace_size, output, hidden_states, w13, w2,
            topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            cuda_stream);
#else
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
#endif
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::moe_fused_dense::nvidia

#endif // ENABLE_NVIDIA_API
