#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "fused_moe_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>

#ifdef ENABLE_CUTLASS_API
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#endif

namespace op::fused_moe::nvidia {

namespace {

constexpr size_t ALIGN_BYTES = 256;

size_t alignUp(size_t x, size_t align = ALIGN_BYTES) {
    return (x + align - 1) / align * align;
}

size_t dtypeSize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_BF16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    default:
        return 0;
    }
}

size_t fusedMoeWorkspaceBytes(const FusedMoeInfo &info) {
    const size_t elem_size = dtypeSize(info.dtype);
    const size_t route_count = info.N * info.topk;
    size_t size = 0;
    size += alignUp(route_count * info.w1_cols * elem_size);
    size += alignUp(route_count * info.inter_size * elem_size);
    size += alignUp(info.N * info.hidden_size * sizeof(float));
    return size;
}

void *advanceWorkspace(uint8_t *&ptr, size_t &remaining, size_t bytes, size_t alignment = 16) {
    auto address = reinterpret_cast<uintptr_t>(ptr);
    auto aligned = alignUp(address, alignment);
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

#ifdef ENABLE_CUTLASS_API
size_t cutlassFusedMoeWorkspaceBytes(const FusedMoeInfo &info) {
    const size_t elem_size = dtypeSize(info.dtype);
    const size_t route_count = info.N * info.topk;
    size_t bytes = 0;
    bytes += alignUp((info.num_experts + 1) * sizeof(int), 16);
    bytes += alignUp((info.num_experts + 1) * sizeof(int), 16);
    bytes += alignUp((info.num_experts + 1) * sizeof(int), 16);
    bytes += alignUp(route_count * sizeof(int), 16);
    bytes += alignUp(route_count * sizeof(int), 16);
    bytes += alignUp(route_count * info.hidden_size * elem_size, 16);
    bytes += alignUp(route_count * info.w1_cols * elem_size, 16);
    bytes += alignUp(route_count * info.inter_size * elem_size, 16);
    bytes += alignUp(route_count * info.hidden_size * elem_size, 16);
    bytes += alignUp(info.num_experts * sizeof(cutlass::gemm::GemmCoord), 16);
    bytes += alignUp(info.num_experts * sizeof(void *), 16) * 4;
    bytes += alignUp(info.num_experts * sizeof(int64_t), 16) * 4;
    return bytes + 256;
}
#endif

size_t workspaceBytes(const FusedMoeInfo &info) {
#ifdef ENABLE_CUTLASS_API
    if (info.dtype == INFINI_DTYPE_F16 || info.dtype == INFINI_DTYPE_BF16) {
        return cutlassFusedMoeWorkspaceBytes(info);
    }
#endif
    return fusedMoeWorkspaceBytes(info);
}

__global__ void countExpertsKernel(const int *selected_experts, int *counts, int pairs, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) {
        return;
    }
    int expert = selected_experts[idx];
    if (expert >= 0 && expert < num_experts) {
        atomicAdd(counts + expert, 1);
    }
}

__global__ void exclusivePrefixCountsKernel(const int *counts, int *offsets, int num_experts) {
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

template <typename T>
__global__ void packRoutesKernel(const T *input,
                                 const int *selected_experts,
                                 const int *offsets,
                                 int *positions,
                                 int *output_permutation,
                                 int *row_expert,
                                 T *packed_input,
                                 int pairs,
                                 int topk,
                                 int hidden_size,
                                 int num_experts) {
    int pair = blockIdx.x;
    int tid = threadIdx.x;
    if (pair >= pairs) {
        return;
    }
    int expert = selected_experts[pair];
    if (expert < 0 || expert >= num_experts) {
        return;
    }

    __shared__ int row;
    if (tid == 0) {
        int local = atomicAdd(positions + expert, 1);
        row = offsets[expert] + local;
        output_permutation[pair] = row;
        row_expert[row] = expert;
    }
    __syncthreads();

    int token = pair / topk;
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        packed_input[static_cast<size_t>(row) * hidden_size + h] = input[static_cast<size_t>(token) * hidden_size + h];
    }
}

template <typename T>
__global__ void addGroupedBiasKernel(T *matrix,
                                     const T *bias,
                                     const int *row_expert,
                                     int rows,
                                     int cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(rows) * cols;
    if (idx >= total) {
        return;
    }
    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx - static_cast<size_t>(row) * cols);
    int expert = row_expert[row];
    float v = moeToFloat(matrix[idx]) + moeToFloat(bias[static_cast<size_t>(expert) * cols + col]);
    matrix[idx] = moeFromFloat<T>(v);
}

template <typename T>
__global__ void groupedActivationKernel(T *activated,
                                        const T *w1_out,
                                        int rows,
                                        int inter_size,
                                        int w1_cols,
                                        int activation) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(rows) * inter_size;
    if (idx >= total) {
        return;
    }
    int row = static_cast<int>(idx / inter_size);
    int col = static_cast<int>(idx - static_cast<size_t>(row) * inter_size);
    const T *base = w1_out + static_cast<size_t>(row) * w1_cols;
    float gate = moeToFloat(base[col]);
    float act;
    if (activation == INFINIOP_FUSED_MOE_ACT_SWIGLU) {
        float up = moeToFloat(base[inter_size + col]);
        act = moeSilu(gate) * up;
    } else {
        act = moeSilu(gate);
    }
    activated[idx] = moeFromFloat<T>(act);
}

template <typename T>
__global__ void gatherWeightedOutputKernel(const T *__restrict__ expert_out,
                                           T *__restrict__ out,
                                           const int *__restrict__ output_permutation,
                                           const float *__restrict__ final_scales,
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
                sum += moeToFloat(expert_out[static_cast<size_t>(src_row) * hidden_size + h]) * final_scales[pair];
            }
        }
        out[static_cast<size_t>(token) * hidden_size + h] = moeFromFloat<T>(sum);
    }
}

#ifdef ENABLE_CUTLASS_API
template <typename T>
__global__ void setupW1GroupedGemmKernel(cutlass::gemm::GemmCoord *problems,
                                         void **ptr_a,
                                         void **ptr_b,
                                         void **ptr_c,
                                         void **ptr_d,
                                         int64_t *lda,
                                         int64_t *ldb,
                                         int64_t *ldc,
                                         int64_t *ldd,
                                         const int *counts,
                                         const int *offsets,
                                         const T *packed_input,
                                         const T *w1,
                                         T *w1_out,
                                         int num_experts,
                                         int hidden_size,
                                         int w1_cols) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    int off = offsets[expert];
    problems[expert] = cutlass::gemm::GemmCoord(m, w1_cols, hidden_size);
    ptr_a[expert] = const_cast<T *>(packed_input + static_cast<size_t>(off) * hidden_size);
    ptr_b[expert] = const_cast<T *>(w1 + static_cast<size_t>(expert) * w1_cols * hidden_size);
    ptr_c[expert] = w1_out + static_cast<size_t>(off) * w1_cols;
    ptr_d[expert] = w1_out + static_cast<size_t>(off) * w1_cols;
    lda[expert] = hidden_size;
    ldb[expert] = hidden_size;
    ldc[expert] = w1_cols;
    ldd[expert] = w1_cols;
}

template <typename T>
__global__ void setupW2GroupedGemmKernel(cutlass::gemm::GemmCoord *problems,
                                         void **ptr_a,
                                         void **ptr_b,
                                         void **ptr_c,
                                         void **ptr_d,
                                         int64_t *lda,
                                         int64_t *ldb,
                                         int64_t *ldc,
                                         int64_t *ldd,
                                         const int *counts,
                                         const int *offsets,
                                         const T *activated,
                                         const T *w2,
                                         T *expert_out,
                                         int num_experts,
                                         int hidden_size,
                                         int inter_size) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts) {
        return;
    }
    int m = counts[expert];
    int off = offsets[expert];
    problems[expert] = cutlass::gemm::GemmCoord(m, hidden_size, inter_size);
    ptr_a[expert] = const_cast<T *>(activated + static_cast<size_t>(off) * inter_size);
    ptr_b[expert] = const_cast<T *>(w2 + static_cast<size_t>(expert) * hidden_size * inter_size);
    ptr_c[expert] = expert_out + static_cast<size_t>(off) * hidden_size;
    ptr_d[expert] = expert_out + static_cast<size_t>(off) * hidden_size;
    lda[expert] = inter_size;
    ldb[expert] = inter_size;
    ldc[expert] = hidden_size;
    ldd[expert] = hidden_size;
}

template <typename CutlassT>
infiniStatus_t launchCutlassGroupedGemm(int problem_count,
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

    static const int threadblock_count = Gemm::sufficient();
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

template <typename T, typename CutlassT>
infiniStatus_t launchCutlassFusedMoe(
    const FusedMoeInfo &info,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    cudaStream_t stream) {
    if (workspace_size < cutlassFusedMoeWorkspaceBytes(info)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (workspace == nullptr && cutlassFusedMoeWorkspaceBytes(info) != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }

    const int num_tokens = static_cast<int>(info.N);
    const int hidden_size = static_cast<int>(info.hidden_size);
    const int inter_size = static_cast<int>(info.inter_size);
    const int w1_cols = static_cast<int>(info.w1_cols);
    const int num_experts = static_cast<int>(info.num_experts);
    const int topk = static_cast<int>(info.topk);
    const int pairs = num_tokens * topk;

    uint8_t *ptr = reinterpret_cast<uint8_t *>(workspace);
    size_t remaining = workspace_size;
    auto counts = reinterpret_cast<int *>(advanceWorkspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto offsets = reinterpret_cast<int *>(advanceWorkspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto positions = reinterpret_cast<int *>(advanceWorkspace(ptr, remaining, (num_experts + 1) * sizeof(int)));
    auto output_permutation = reinterpret_cast<int *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * sizeof(int)));
    auto row_expert = reinterpret_cast<int *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * sizeof(int)));
    auto packed_input = reinterpret_cast<T *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * hidden_size * sizeof(T)));
    auto w1_out = reinterpret_cast<T *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * w1_cols * sizeof(T)));
    auto activated = reinterpret_cast<T *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * inter_size * sizeof(T)));
    auto expert_out = reinterpret_cast<T *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(pairs) * hidden_size * sizeof(T)));
    auto grouped_problems = reinterpret_cast<cutlass::gemm::GemmCoord *>(
        advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(cutlass::gemm::GemmCoord), alignof(cutlass::gemm::GemmCoord)));
    auto grouped_ptr_a = reinterpret_cast<void **>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_b = reinterpret_cast<void **>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_c = reinterpret_cast<void **>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_ptr_d = reinterpret_cast<void **>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(void *), alignof(void *)));
    auto grouped_lda = reinterpret_cast<int64_t *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldb = reinterpret_cast<int64_t *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldc = reinterpret_cast<int64_t *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));
    auto grouped_ldd = reinterpret_cast<int64_t *>(advanceWorkspace(ptr, remaining, static_cast<size_t>(num_experts) * sizeof(int64_t), alignof(int64_t)));

    if (!counts || !offsets || !positions || !output_permutation || !row_expert || !packed_input || !w1_out || !activated || !expert_out || !grouped_problems || !grouped_ptr_a || !grouped_ptr_b || !grouped_ptr_c || !grouped_ptr_d || !grouped_lda || !grouped_ldb || !grouped_ldc || !grouped_ldd) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    const int threads = 256;
    CHECK_CUDA(cudaMemsetAsync(counts, 0, (num_experts + 1) * sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(positions, 0, (num_experts + 1) * sizeof(int), stream));
    CHECK_CUDA(cudaMemsetAsync(output_permutation, 0xff, static_cast<size_t>(pairs) * sizeof(int), stream));

    countExpertsKernel<<<(pairs + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const int *>(token_selected_experts), counts, pairs, num_experts);
    CHECK_CUDA(cudaGetLastError());

    int scan_threads = 1;
    while (scan_threads < num_experts) {
        scan_threads <<= 1;
    }
    scan_threads = std::max(32, scan_threads);
    exclusivePrefixCountsKernel<<<1, scan_threads, scan_threads * sizeof(int), stream>>>(
        counts, offsets, num_experts);
    CHECK_CUDA(cudaGetLastError());

    packRoutesKernel<T><<<pairs, threads, 0, stream>>>(
        static_cast<const T *>(input),
        static_cast<const int *>(token_selected_experts),
        offsets,
        positions,
        output_permutation,
        row_expert,
        packed_input,
        pairs,
        topk,
        hidden_size,
        num_experts);
    CHECK_CUDA(cudaGetLastError());

    setupW1GroupedGemmKernel<T><<<(num_experts + threads - 1) / threads, threads, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, counts, offsets, packed_input,
        static_cast<const T *>(w1), w1_out, num_experts, hidden_size, w1_cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_STATUS(launchCutlassGroupedGemm<CutlassT>(
        num_experts, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));

    if (b1 != nullptr) {
        size_t b1_total = static_cast<size_t>(pairs) * w1_cols;
        addGroupedBiasKernel<T><<<(b1_total + threads - 1) / threads, threads, 0, stream>>>(
            w1_out, static_cast<const T *>(b1), row_expert, pairs, w1_cols);
        CHECK_CUDA(cudaGetLastError());
    }

    size_t act_total = static_cast<size_t>(pairs) * inter_size;
    groupedActivationKernel<T><<<(act_total + threads - 1) / threads, threads, 0, stream>>>(
        activated, w1_out, pairs, inter_size, w1_cols, static_cast<int>(info.activation));
    CHECK_CUDA(cudaGetLastError());

    setupW2GroupedGemmKernel<T><<<(num_experts + threads - 1) / threads, threads, 0, stream>>>(
        grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, counts, offsets, activated,
        static_cast<const T *>(w2), expert_out, num_experts, hidden_size, inter_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_STATUS(launchCutlassGroupedGemm<CutlassT>(
        num_experts, grouped_problems, grouped_ptr_a, grouped_ptr_b, grouped_ptr_c, grouped_ptr_d,
        grouped_lda, grouped_ldb, grouped_ldc, grouped_ldd, stream));

    if (b2 != nullptr) {
        size_t b2_total = static_cast<size_t>(pairs) * hidden_size;
        addGroupedBiasKernel<T><<<(b2_total + threads - 1) / threads, threads, 0, stream>>>(
            expert_out, static_cast<const T *>(b2), row_expert, pairs, hidden_size);
        CHECK_CUDA(cudaGetLastError());
    }

    gatherWeightedOutputKernel<T><<<num_tokens, std::min(hidden_size, 1024), 0, stream>>>(
        expert_out,
        static_cast<T *>(out),
        output_permutation,
        static_cast<const float *>(token_final_scales),
        num_tokens,
        topk,
        hidden_size);
    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}
#endif

template <typename T>
infiniStatus_t launchFusedMoe(
    const FusedMoeInfo &info,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    cudaStream_t stream) {
    if (workspace_size < fusedMoeWorkspaceBytes(info)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (workspace == nullptr && fusedMoeWorkspaceBytes(info) != 0) {
        return INFINI_STATUS_NULL_POINTER;
    }

    const size_t route_count = info.N * info.topk;
    const size_t out_count = info.N * info.hidden_size;
    const int threads = 256;

    auto *base = reinterpret_cast<std::byte *>(workspace);
    T *w1_out = reinterpret_cast<T *>(base);
    base += alignUp(route_count * info.w1_cols * sizeof(T));
    T *activated = reinterpret_cast<T *>(base);
    base += alignUp(route_count * info.inter_size * sizeof(T));
    float *out_accum = reinterpret_cast<float *>(base);

    CHECK_CUDA(cudaMemsetAsync(out_accum, 0, out_count * sizeof(float), stream));

    size_t w1_total = route_count * info.w1_cols;
    int w1_blocks = static_cast<int>((w1_total + threads - 1) / threads);
    fusedMoeW1Kernel<T><<<w1_blocks, threads, 0, stream>>>(
        w1_out,
        static_cast<const T *>(input),
        static_cast<const int32_t *>(token_selected_experts),
        static_cast<const T *>(w1),
        static_cast<const T *>(b1),
        route_count,
        info.hidden_size,
        info.topk,
        info.w1_cols,
        info.num_experts);
    CHECK_CUDA(cudaGetLastError());

    size_t act_total = route_count * info.inter_size;
    int act_blocks = static_cast<int>((act_total + threads - 1) / threads);
    fusedMoeActivationKernel<T><<<act_blocks, threads, 0, stream>>>(
        activated,
        w1_out,
        route_count,
        info.inter_size,
        info.w1_cols,
        static_cast<int>(info.activation));
    CHECK_CUDA(cudaGetLastError());

    size_t w2_total = route_count * info.hidden_size;
    int w2_blocks = static_cast<int>((w2_total + threads - 1) / threads);
    fusedMoeW2ScatterKernel<T><<<w2_blocks, threads, 0, stream>>>(
        out_accum,
        activated,
        static_cast<const int32_t *>(token_selected_experts),
        static_cast<const float *>(token_final_scales),
        static_cast<const T *>(w2),
        static_cast<const T *>(b2),
        route_count,
        info.hidden_size,
        info.inter_size,
        info.topk,
        info.num_experts);
    CHECK_CUDA(cudaGetLastError());

    int cast_blocks = static_cast<int>((out_count + threads - 1) / threads);
    fusedMoeCastKernel<T><<<cast_blocks, threads, 0, stream>>>(
        static_cast<T *>(out), out_accum, out_count);
    CHECK_CUDA(cudaGetLastError());

    return INFINI_STATUS_SUCCESS;
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t token_selected_experts_desc,
    infiniopTensorDescriptor_t token_final_scales_desc,
    infiniopTensorDescriptor_t w1_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t b1_desc,
    infiniopTensorDescriptor_t b2_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeInfo::create(out_desc, input_desc, token_selected_experts_desc,
                                     token_final_scales_desc, w1_desc, w2_desc,
                                     b1_desc, b2_desc, activation);
    CHECK_RESULT(info);
    auto taken = info.take();
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        taken, workspaceBytes(taken), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    void *stream_) const {
    if (out == nullptr || input == nullptr || token_selected_experts == nullptr || token_final_scales == nullptr || w1 == nullptr || w2 == nullptr || (_info.has_b1 && b1 == nullptr) || (_info.has_b2 && b2 == nullptr)) {
        return INFINI_STATUS_NULL_POINTER;
    }

    cudaStream_t stream = (cudaStream_t)stream_;
    if (_info.dtype == INFINI_DTYPE_F16) {
#ifdef ENABLE_CUTLASS_API
        return launchCutlassFusedMoe<half, cutlass::half_t>(_info, workspace, workspace_size, out, input,
                                                            token_selected_experts, token_final_scales,
                                                            w1, w2, b1, b2, stream);
#else
        return launchFusedMoe<half>(_info, workspace, workspace_size, out, input,
                                    token_selected_experts, token_final_scales,
                                    w1, w2, b1, b2, stream);
#endif
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
#ifdef ENABLE_CUTLASS_API
        return launchCutlassFusedMoe<__nv_bfloat16, cutlass::bfloat16_t>(_info, workspace, workspace_size, out, input,
                                                                         token_selected_experts, token_final_scales,
                                                                         w1, w2, b1, b2, stream);
#else
        return launchFusedMoe<__nv_bfloat16>(_info, workspace, workspace_size, out, input,
                                             token_selected_experts, token_final_scales,
                                             w1, w2, b1, b2, stream);
#endif
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return launchFusedMoe<float>(_info, workspace, workspace_size, out, input,
                                     token_selected_experts, token_final_scales,
                                     w1, w2, b1, b2, stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fused_moe::nvidia
