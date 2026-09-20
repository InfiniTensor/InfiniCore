#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "gemm_nvidia.cuh"
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#include <cublasLt.h>
#endif
#include <cuda_bf16.h>

namespace op::gemm::nvidia {

namespace {

constexpr int kMixedSmallNMax = 16;
constexpr int kMixedWarpsPerBlock = 8;

// C[m, n] = A_bf16[m, k] @ B_f32[k, n].  This is the transposed
// representation used when a row-major LM head is evaluated as W @ hidden^T.
// One warp owns a weight row and reuses each BF16 value across a tile of up to
// 16 hidden columns.  grid.y tiles longer prompts without growing registers.
__global__ void mixed_bf16_f32_small_n_kernel(
    const __nv_bfloat16 *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ c,
    size_t m, size_t n, size_t k,
    ptrdiff_t a_row_stride, ptrdiff_t a_col_stride,
    ptrdiff_t b_row_stride, ptrdiff_t b_col_stride,
    ptrdiff_t c_row_stride, ptrdiff_t c_col_stride,
    float alpha, float beta) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kMixedWarpsPerBlock
        + static_cast<size_t>(warp);
    const size_t col_base = static_cast<size_t>(blockIdx.y) * kMixedSmallNMax;
    if (row >= m || col_base >= n) {
        return;
    }
    const size_t remaining = n - col_base;
    const size_t tile_n = remaining < static_cast<size_t>(kMixedSmallNMax)
        ? remaining : static_cast<size_t>(kMixedSmallNMax);

    float acc[kMixedSmallNMax];
#pragma unroll
    for (int j = 0; j < kMixedSmallNMax; ++j) {
        acc[j] = 0.0f;
    }
    for (size_t kk = static_cast<size_t>(lane); kk < k; kk += 32) {
        const float av = __bfloat162float(a[row * a_row_stride + kk * a_col_stride]);
#pragma unroll
        for (int j = 0; j < kMixedSmallNMax; ++j) {
            if (static_cast<size_t>(j) >= tile_n) {
                break;
            }
            acc[j] += av * b[kk * b_row_stride
                + (col_base + static_cast<size_t>(j)) * b_col_stride];
        }
    }
    constexpr unsigned mask = 0xffffffffu;
#pragma unroll
    for (int j = 0; j < kMixedSmallNMax; ++j) {
        if (static_cast<size_t>(j) >= tile_n) {
            break;
        }
        float value = acc[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(mask, value, offset);
        }
        if (lane == 0) {
            const size_t idx = row * c_row_stride
                + (col_base + static_cast<size_t>(j)) * c_col_stride;
            c[idx] = alpha * value + (beta == 0.0f ? 0.0f : beta * c[idx]);
        }
    }
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    infiniDtype_t a_dtype;
    infiniDtype_t b_dtype;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;

    void destroyLtDescriptors();
    bool createBf16LtDescriptors(const MatmulInfo &info);
#endif
};

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
static size_t ltLayoutRows(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.rows : matrix.cols;
}

static size_t ltLayoutCols(const BlasMatrix &matrix) {
    return matrix.row_stride == 1 ? matrix.cols : matrix.rows;
}

static bool setLtLayoutBatch(cublasLtMatrixLayout_t layout, const BlasMatrix &matrix, size_t batch) {
    int32_t batch_count = static_cast<int32_t>(batch);
    int64_t stride = static_cast<int64_t>(matrix.stride);
    return cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
               &batch_count, sizeof(batch_count))
            == CUBLAS_STATUS_SUCCESS
        && cublasLtMatrixLayoutSetAttribute(
               layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
               &stride, sizeof(stride))
               == CUBLAS_STATUS_SUCCESS;
}

void Descriptor::Opaque::destroyLtDescriptors() {
    if (a_layout) {
        cublasLtMatrixLayoutDestroy(a_layout);
        a_layout = nullptr;
    }
    if (b_layout) {
        cublasLtMatrixLayoutDestroy(b_layout);
        b_layout = nullptr;
    }
    if (c_layout) {
        cublasLtMatrixLayoutDestroy(c_layout);
        c_layout = nullptr;
    }
    if (lt_desc) {
        cublasLtMatmulDescDestroy(lt_desc);
        lt_desc = nullptr;
    }
    if (lt_handle) {
        cublasLtDestroy(lt_handle);
        lt_handle = nullptr;
    }
}

bool Descriptor::Opaque::createBf16LtDescriptors(const MatmulInfo &info) {
    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (cublasLtCreate(&lt_handle) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescCreate(&lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSA,
            &op_a, sizeof(op_a))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(
            lt_desc, CUBLASLT_MATMUL_DESC_TRANSB,
            &op_b, sizeof(op_b))
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (cublasLtMatrixLayoutCreate(
            &a_layout, CUDA_R_16BF,
            ltLayoutRows(info.a_matrix), ltLayoutCols(info.a_matrix),
            info.a_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &b_layout, CUDA_R_16BF,
            ltLayoutRows(info.b_matrix), ltLayoutCols(info.b_matrix),
            info.b_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatrixLayoutCreate(
            &c_layout, CUDA_R_16BF,
            ltLayoutRows(info.c_matrix), ltLayoutCols(info.c_matrix),
            info.c_matrix.ld())
        != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    return setLtLayoutBatch(a_layout, info.a_matrix, info.batch)
        && setLtLayoutBatch(b_layout, info.b_matrix, info.batch)
        && setLtLayoutBatch(c_layout, info.c_matrix, info.batch);
}
#endif

Descriptor::~Descriptor() {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (_opaque) {
        _opaque->destroyLtDescriptors();
    }
#endif
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto a_dtype = a_desc->dtype();
    auto b_dtype = b_desc->dtype();
    CHECK_DTYPE(a_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_DTYPE(b_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    auto opaque = new Opaque{handle->internal(), a_dtype, b_dtype};
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (dtype == INFINI_DTYPE_BF16 && !opaque->createBf16LtDescriptors(info)) {
        opaque->destroyLtDescriptors();
    }
#endif

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        opaque,
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

    auto cuda_dtype = [](infiniDtype_t dtype) {
        switch (dtype) {
        case INFINI_DTYPE_F16:
            return CUDA_R_16F;
        case INFINI_DTYPE_BF16:
            return CUDA_R_16BF;
        case INFINI_DTYPE_F32:
            return CUDA_R_32F;
        default:
            return CUDA_R_32F;
        }
    };
    a_type = cuda_dtype(_opaque->a_dtype);
    b_type = cuda_dtype(_opaque->b_dtype);
    c_type = cuda_dtype(_dtype);

    switch (_dtype) {
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
        compute_type =
            _opaque->a_dtype == INFINI_DTYPE_F32
                    && _opaque->b_dtype == INFINI_DTYPE_F32
                ? CUBLAS_COMPUTE_32F_FAST_TF32
                : CUBLAS_COMPUTE_32F;
#endif
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
        // Row-major output is evaluated as the transposed product B^T @ A^T.
        // Keep the runtime CUDA element types attached to the pointers as well;
        // this was invisible for equal-dtype GEMMs but breaks mixed F32/BF16.
        std::swap(a_type, b_type);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (_dtype == INFINI_DTYPE_F32
        && a_type == CUDA_R_16BF && b_type == CUDA_R_32F
        && _info.batch == 1) {
        const dim3 blocks(
            static_cast<unsigned>(
                (_info.m + kMixedWarpsPerBlock - 1) / kMixedWarpsPerBlock),
            static_cast<unsigned>(
                (_info.n + kMixedSmallNMax - 1) / kMixedSmallNMax));
        mixed_bf16_f32_small_n_kernel<<<
            blocks, kMixedWarpsPerBlock * 32, 0,
            reinterpret_cast<cudaStream_t>(stream)>>>(
                reinterpret_cast<const __nv_bfloat16 *>(a),
                reinterpret_cast<const float *>(b),
                reinterpret_cast<float *>(c),
                _info.m, _info.n, _info.k,
                _info.a_matrix.row_stride, _info.a_matrix.col_stride,
                _info.b_matrix.row_stride, _info.b_matrix.col_stride,
                _info.c_matrix.row_stride, _info.c_matrix.col_stride,
                alpha, beta);
        return cudaGetLastError() == cudaSuccess
            ? INFINI_STATUS_SUCCESS
            : INFINI_STATUS_INTERNAL_ERROR;
    }

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
    if (_dtype == INFINI_DTYPE_BF16 && _opaque->lt_handle && _opaque->lt_desc
        && _opaque->a_layout && _opaque->b_layout && _opaque->c_layout) {
        auto lt_status = cublasLtMatmul(
            _opaque->lt_handle,
            _opaque->lt_desc,
            &alpha,
            a,
            _opaque->a_layout,
            b,
            _opaque->b_layout,
            &beta,
            c,
            _opaque->c_layout,
            c,
            _opaque->c_layout,
            nullptr,
            workspace,
            workspace_size,
            (cudaStream_t)stream);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            return INFINI_STATUS_SUCCESS;
        }
    }
#endif

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
