#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    infiniDtype_t a_dtype;
    infiniDtype_t b_dtype;
    infiniDtype_t c_dtype;
};

namespace {

constexpr size_t ALIGNMENT = 256;

size_t alignUp(size_t value, size_t alignment = ALIGNMENT) {
    return (value + alignment - 1) / alignment * alignment;
}

infiniStatus_t toCudaDataType(infiniDtype_t dtype, cudaDataType *cuda_type) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        *cuda_type = CUDA_R_16F;
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        *cuda_type = CUDA_R_16BF;
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        *cuda_type = CUDA_R_32F;
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

size_t matrixSpanElements(const BlasMatrix &matrix) {
    size_t batch_offset = matrix.batch > 1 ? static_cast<size_t>(std::abs(matrix.stride)) * (matrix.batch - 1) : 0;
    size_t row_offset = matrix.rows > 0 ? static_cast<size_t>(std::abs(matrix.row_stride)) * (matrix.rows - 1) : 0;
    size_t col_offset = matrix.cols > 0 ? static_cast<size_t>(std::abs(matrix.col_stride)) * (matrix.cols - 1) : 0;
    return batch_offset + row_offset + col_offset + 1;
}

size_t matrixCastWorkspaceSize(infiniDtype_t dtype, const BlasMatrix &matrix) {
    if (dtype == INFINI_DTYPE_F32) {
        return 0;
    }
    return alignUp(matrixSpanElements(matrix) * sizeof(float));
}

size_t gemmCastWorkspaceSize(infiniDtype_t c_dtype,
                             infiniDtype_t a_dtype,
                             infiniDtype_t b_dtype,
                             const MatmulInfo &info) {
    if (c_dtype != INFINI_DTYPE_F32) {
        return 0;
    }
    return matrixCastWorkspaceSize(a_dtype, info.a_matrix)
           + matrixCastWorkspaceSize(b_dtype, info.b_matrix);
}

template <typename Src>
__global__ void castMatrixToF32Kernel(float *dst,
                                      const Src *src,
                                      size_t total,
                                      size_t rows,
                                      size_t cols,
                                      ptrdiff_t matrix_stride,
                                      ptrdiff_t row_stride,
                                      ptrdiff_t col_stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    size_t matrix_size = rows * cols;
    size_t batch = idx / matrix_size;
    size_t offset = idx % matrix_size;
    size_t row = offset / cols;
    size_t col = offset % cols;
    ptrdiff_t elem_offset = static_cast<ptrdiff_t>(batch) * matrix_stride
                            + static_cast<ptrdiff_t>(row) * row_stride
                            + static_cast<ptrdiff_t>(col) * col_stride;
    dst[elem_offset] = static_cast<float>(src[elem_offset]);
}

template <>
__global__ void castMatrixToF32Kernel<__half>(float *dst,
                                              const __half *src,
                                              size_t total,
                                              size_t rows,
                                              size_t cols,
                                              ptrdiff_t matrix_stride,
                                              ptrdiff_t row_stride,
                                              ptrdiff_t col_stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    size_t matrix_size = rows * cols;
    size_t batch = idx / matrix_size;
    size_t offset = idx % matrix_size;
    size_t row = offset / cols;
    size_t col = offset % cols;
    ptrdiff_t elem_offset = static_cast<ptrdiff_t>(batch) * matrix_stride
                            + static_cast<ptrdiff_t>(row) * row_stride
                            + static_cast<ptrdiff_t>(col) * col_stride;
    dst[elem_offset] = __half2float(src[elem_offset]);
}

template <>
__global__ void castMatrixToF32Kernel<__nv_bfloat16>(float *dst,
                                                     const __nv_bfloat16 *src,
                                                     size_t total,
                                                     size_t rows,
                                                     size_t cols,
                                                     ptrdiff_t matrix_stride,
                                                     ptrdiff_t row_stride,
                                                     ptrdiff_t col_stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    size_t matrix_size = rows * cols;
    size_t batch = idx / matrix_size;
    size_t offset = idx % matrix_size;
    size_t row = offset / cols;
    size_t col = offset % cols;
    ptrdiff_t elem_offset = static_cast<ptrdiff_t>(batch) * matrix_stride
                            + static_cast<ptrdiff_t>(row) * row_stride
                            + static_cast<ptrdiff_t>(col) * col_stride;
    dst[elem_offset] = __bfloat162float(src[elem_offset]);
}

infiniStatus_t castMatrixToF32(float *dst,
                               const void *src,
                               infiniDtype_t src_dtype,
                               const BlasMatrix &matrix,
                               cudaStream_t stream) {
    size_t total = matrix.batch * matrix.rows * matrix.cols;
    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);
    switch (src_dtype) {
    case INFINI_DTYPE_F16:
        castMatrixToF32Kernel<<<grid, block, 0, stream>>>(
            dst,
            static_cast<const __half *>(src),
            total,
            matrix.rows,
            matrix.cols,
            matrix.stride,
            matrix.row_stride,
            matrix.col_stride);
        break;
    case INFINI_DTYPE_BF16:
        castMatrixToF32Kernel<<<grid, block, 0, stream>>>(
            dst,
            static_cast<const __nv_bfloat16 *>(src),
            total,
            matrix.rows,
            matrix.cols,
            matrix.stride,
            matrix.row_stride,
            matrix.col_stride);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto a_dtype = a_desc->dtype();
    auto b_dtype = b_desc->dtype();
    auto c_dtype = c_desc->dtype();

    CHECK_DTYPE(a_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_DTYPE(b_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_DTYPE(c_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto effective_a_dtype = a_dtype;
    auto effective_b_dtype = b_dtype;
    if (info.is_transed) {
        std::swap(effective_a_dtype, effective_b_dtype);
    }
    size_t workspace_size = gemmCastWorkspaceSize(c_dtype, effective_a_dtype, effective_b_dtype, info);

    *desc_ptr = new Descriptor(
        c_dtype, info, workspace_size,
        new Opaque{handle->internal(), a_dtype, b_dtype, c_dtype},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    cudaDataType a_type, b_type, c_type;
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
    cudaDataType compute_type;
#else
    cublasComputeType_t compute_type;
#endif

    CHECK_STATUS(toCudaDataType(_opaque->a_dtype, &a_type));
    CHECK_STATUS(toCudaDataType(_opaque->b_dtype, &b_type));
    CHECK_STATUS(toCudaDataType(_opaque->c_dtype, &c_type));

    switch (_opaque->c_dtype) {
    case INFINI_DTYPE_F16:
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_BF16:
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = CUBLAS_COMPUTE_32F;
#endif
        break;
    case INFINI_DTYPE_F32:
#if defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
        compute_type = CUDA_R_32F;
#else
        compute_type = (_opaque->a_dtype == INFINI_DTYPE_F32 && _opaque->b_dtype == INFINI_DTYPE_F32)
                         ? CUBLAS_COMPUTE_32F_FAST_TF32
                         : CUBLAS_COMPUTE_32F;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
        std::swap(a_type, b_type);
    }

    auto effective_a_dtype = _info.is_transed ? _opaque->b_dtype : _opaque->a_dtype;
    auto effective_b_dtype = _info.is_transed ? _opaque->a_dtype : _opaque->b_dtype;
    auto required_workspace = gemmCastWorkspaceSize(_opaque->c_dtype, effective_a_dtype, effective_b_dtype, _info);
    CHECK_OR_RETURN(workspace_size >= required_workspace, INFINI_STATUS_INSUFFICIENT_WORKSPACE);
    auto workspace_ptr = static_cast<char *>(workspace);
    auto stream_ = static_cast<cudaStream_t>(stream);
    if (_opaque->c_dtype == INFINI_DTYPE_F32 && effective_a_dtype != INFINI_DTYPE_F32) {
        auto cast_a = reinterpret_cast<float *>(workspace_ptr);
        CHECK_STATUS(castMatrixToF32(cast_a, a, effective_a_dtype, _info.a_matrix, stream_));
        a = cast_a;
        a_type = CUDA_R_32F;
        workspace_ptr += matrixCastWorkspaceSize(effective_a_dtype, _info.a_matrix);
    }
    if (_opaque->c_dtype == INFINI_DTYPE_F32 && effective_b_dtype != INFINI_DTYPE_F32) {
        auto cast_b = reinterpret_cast<float *>(workspace_ptr);
        CHECK_STATUS(castMatrixToF32(cast_b, b, effective_b_dtype, _info.b_matrix, stream_));
        b = cast_b;
        b_type = CUDA_R_32F;
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    CHECK_STATUS(_opaque->internal->useCublas(
        (cudaStream_t)stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(
                cublasGemmStridedBatchedEx(
                    handle,
                    op_a,
                    op_b,
                    static_cast<int>(_info.m),
                    static_cast<int>(_info.n),
                    static_cast<int>(_info.k),
                    &alpha,
                    a,
                    a_type,
                    static_cast<int>(_info.a_matrix.ld()),
                    _info.a_matrix.stride,
                    b,
                    b_type,
                    static_cast<int>(_info.b_matrix.ld()),
                    _info.b_matrix.stride,
                    &beta,
                    c,
                    c_type,
                    static_cast<int>(_info.c_matrix.ld()),
                    _info.c_matrix.stride,
                    static_cast<int>(_info.batch),
                    compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::nvidia
