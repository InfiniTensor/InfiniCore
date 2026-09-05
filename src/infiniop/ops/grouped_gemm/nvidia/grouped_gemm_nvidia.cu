#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "grouped_gemm_nvidia.cuh"

#include <cstdint>
#include <vector>

namespace op::grouped_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t group_sizes_desc) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    CHECK_DTYPE(c_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = GroupedGemmInfo::create(c_desc, a_desc, b_desc, group_sizes_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    // Workspace holds the host-side copy of `group_sizes` so calculate() can
    // walk each group's slab without round-tripping through the user's pointer.
    size_t workspace_size = info.num_groups * sizeof(int32_t);

    *desc_ptr = new Descriptor(
        info, workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

struct CublasDtypes {
    cudaDataType ab_type;
    cudaDataType c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif
};

infiniStatus_t resolveDtypes(infiniDtype_t dtype, CublasDtypes &out) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        out.ab_type = out.c_type = CUDA_R_16F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        out.compute_type = CUDA_R_32F;
#else
        out.compute_type = CUBLAS_COMPUTE_32F;
#endif
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        out.ab_type = out.c_type = CUDA_R_16BF;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        out.compute_type = CUDA_R_32F;
#else
        out.compute_type = CUBLAS_COMPUTE_32F;
#endif
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        out.ab_type = out.c_type = CUDA_R_32F;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        out.compute_type = CUDA_R_32F;
#else
        out.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

inline size_t element_size(infiniDtype_t dtype) {
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

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *group_sizes,
    const void *group_sizes_host,
    float alpha,
    float beta,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CublasDtypes dtypes;
    CHECK_STATUS(resolveDtypes(_info.dtype, dtypes));

    auto cuda_stream = static_cast<cudaStream_t>(stream);

    // Resolve the host-side group sizes. When the caller already holds them on
    // the host (e.g. MoE routing computed on CPU) it passes `group_sizes_host`,
    // letting us skip the per-call device->host copy + stream sync -- the only
    // host/device round-trip on this hot path. Otherwise we fall back to syncing
    // the small (num_groups * 4 bytes) array down through the workspace.
    // (The true batched path, cublasGemmGroupedBatched, needs CUDA 12.4+ and is
    // left for a future revision.)
    const int32_t *sizes_host;
    if (group_sizes_host != nullptr) {
        sizes_host = reinterpret_cast<const int32_t *>(group_sizes_host);
    } else {
        auto sizes_ws = reinterpret_cast<int32_t *>(workspace);
        CHECK_CUDA(cudaMemcpyAsync(
            sizes_ws, group_sizes,
            _info.num_groups * sizeof(int32_t),
            cudaMemcpyDeviceToHost, cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
        sizes_host = sizes_ws;
    }

    const size_t elem_size = element_size(_info.dtype);
    auto a_bytes = reinterpret_cast<const uint8_t *>(a);
    auto b_bytes = reinterpret_cast<const uint8_t *>(b);
    auto c_bytes = reinterpret_cast<uint8_t *>(c);

    // C[rows_g, N] = alpha * A[rows_g, K] @ B[g][N, K]^T + beta * C[rows_g, N]
    // (all row-major). cuBLAS is column-major, so we ask it to compute the
    // transpose `C^T[N, rows_g] = B[g] @ A^T` instead — same memory.
    //
    // With cuBLAS conventions, when we view row-major data as column-major it
    // appears mathematically transposed. So for `op(A_cb) @ op(B_cb)`:
    //   cuBLAS A_cb = B[g] memory, op = OP_T -> recovers the N x K math matrix
    //   cuBLAS B_cb = A   memory, op = OP_N -> recovers the K x rows_g matrix
    //   cuBLAS C_cb = C   memory, no transpose -> N x rows_g output
    // The leading dimensions equal the row stride of the row-major tensor
    // because that is the stride between successive columns once viewed as
    // column-major.
    CHECK_STATUS(_opaque->internal->useCublas(
        cuda_stream,
        [&](cublasHandle_t handle) {
            ptrdiff_t row_offset = 0;
            for (size_t g = 0; g < _info.num_groups; ++g) {
                int32_t rows = sizes_host[g];
                if (rows <= 0) {
                    continue;
                }
                const void *a_g = a_bytes + size_t(row_offset) * size_t(_info.a_row_stride) * elem_size;
                const void *b_g = b_bytes + g * size_t(_info.b_group_stride) * elem_size;
                void *c_g = c_bytes + size_t(row_offset) * size_t(_info.c_row_stride) * elem_size;

                CHECK_CUBLAS(cublasGemmEx(
                    handle,
                    CUBLAS_OP_T, // op on B[g] view -> N x K
                    CUBLAS_OP_N, // op on A view -> K x rows_g
                    static_cast<int>(_info.n),    // m_cb (rows of C^T)
                    static_cast<int>(rows),       // n_cb (cols of C^T)
                    static_cast<int>(_info.k),    // k_cb
                    &alpha,
                    b_g, dtypes.ab_type,
                    static_cast<int>(_info.b_row_stride), // lda = stride between cols of B view
                    a_g, dtypes.ab_type,
                    static_cast<int>(_info.a_row_stride), // ldb = stride between cols of A view
                    &beta,
                    c_g, dtypes.c_type,
                    static_cast<int>(_info.c_row_stride), // ldc = stride between cols of C view
                    dtypes.compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

                row_offset += rows;
            }
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::grouped_gemm::nvidia
