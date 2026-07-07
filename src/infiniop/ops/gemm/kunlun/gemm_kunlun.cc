#include "gemm_kunlun.h"
#include "../../../devices/kunlun/kunlun_common.h"
#include "../../../devices/kunlun/kunlun_xblas.h"
#include "gemm_kunlun_cast.h"
#include <algorithm>
#include <cublasLt.h>

namespace op::gemm::kunlun {

typedef device::kunlun::blas::Handle::Internal HandleInternal;

static bool isLargeBf16SkinnyGemm(const MatmulInfo &info) {
    return info.is_transed && info.n == 1 && info.m > 2048;
}

static bool useBf16Lt(const MatmulInfo &info) {
    return !isLargeBf16SkinnyGemm(info);
}

static size_t packedMatrixSize(size_t rows, size_t cols, size_t batch) {
    return rows * cols * batch;
}

static size_t bf16WorkspaceSize(const MatmulInfo &info) {
    return (packedMatrixSize(info.m, info.k, info.batch)
            + packedMatrixSize(info.k, info.n, info.batch)
            + packedMatrixSize(info.m, info.n, info.batch))
         * sizeof(float);
}

struct Descriptor::Opaque {
    std::shared_ptr<HandleInternal> internal;
    cublasLtHandle_t lt_handle = nullptr;
    cublasLtMatmulDesc_t lt_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    void destroyLtDescriptors();
    bool createBf16LtDescriptors(const MatmulInfo &info);
};

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
    }
    if (b_layout) {
        cublasLtMatrixLayoutDestroy(b_layout);
    }
    if (c_layout) {
        cublasLtMatrixLayoutDestroy(c_layout);
    }
    if (lt_desc) {
        cublasLtMatmulDescDestroy(lt_desc);
    }
    if (lt_handle) {
        cublasLtDestroy(lt_handle);
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

Descriptor::~Descriptor() {
    if (_opaque) {
        _opaque->destroyLtDescriptors();
    }
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::kunlun::blas::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();
    auto workspace_size = dtype == INFINI_DTYPE_BF16 ? bf16WorkspaceSize(info) : 0;

    auto opaque = new Opaque{handle->internal()};
    bool use_bf16_lt = dtype == INFINI_DTYPE_BF16 && useBf16Lt(info);
    if (use_bf16_lt && !opaque->createBf16LtDescriptors(info)) {
        opaque->destroyLtDescriptors();
        opaque->lt_handle = nullptr;
        opaque->lt_desc = nullptr;
        opaque->a_layout = nullptr;
        opaque->b_layout = nullptr;
        opaque->c_layout = nullptr;
    }

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
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
    cublasComputeType_t compute_type;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
        break;
    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
        break;
    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = CUDA_R_32F;
        compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_info.is_transed) {
        std::swap(a, b);
    }

    auto op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    if (_dtype == INFINI_DTYPE_BF16) {
        auto use_bf16_lt = useBf16Lt(_info);
        if (use_bf16_lt && _opaque->lt_handle && _opaque->lt_desc && _opaque->a_layout && _opaque->b_layout && _opaque->c_layout) {
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
                xpu_wait(stream);
                return INFINI_STATUS_SUCCESS;
            }
        }

        if (workspace_size < _workspace_size) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }

        auto temp_element_size = sizeof(float);
        auto workspace_bytes = reinterpret_cast<char *>(workspace);
        auto a_tmp = workspace_bytes;
        auto b_tmp = a_tmp + packedMatrixSize(_info.m, _info.k, _info.batch) * temp_element_size;
        auto c_tmp = b_tmp + packedMatrixSize(_info.k, _info.n, _info.batch) * temp_element_size;
        auto temp_type = CUDA_R_32F;
        auto a_stride = static_cast<long long>(packedMatrixSize(_info.m, _info.k, 1));
        auto b_stride = static_cast<long long>(packedMatrixSize(_info.k, _info.n, 1));
        auto c_stride = static_cast<long long>(packedMatrixSize(_info.m, _info.n, 1));

        CHECK_STATUS(castBf16ToF32(a, a_tmp, _info.a_matrix, _info.batch, (kunlunStream_t)stream));
        CHECK_STATUS(castBf16ToF32(b, b_tmp, _info.b_matrix, _info.batch, (kunlunStream_t)stream));
        if (beta == 0.0f) {
            CHECK_STATUS(zeroPackedBuffer(c_tmp, packedMatrixSize(_info.m, _info.n, _info.batch) * temp_element_size, (kunlunStream_t)stream));
        } else {
            CHECK_STATUS(castBf16ToF32(c, c_tmp, _info.c_matrix, _info.batch, (kunlunStream_t)stream));
        }

        CHECK_STATUS(_opaque->internal->useCublas(
            (cudaStream_t)stream,
            [&](cublasHandle_t handle) {
                CHECK_CUBLAS(
                    cublasGemmStridedBatchedEx(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        static_cast<int>(_info.m),
                        static_cast<int>(_info.n),
                        static_cast<int>(_info.k),
                        &alpha,
                        a_tmp,
                        temp_type,
                        static_cast<int>(_info.m),
                        a_stride,
                        b_tmp,
                        temp_type,
                        static_cast<int>(_info.k),
                        b_stride,
                        &beta,
                        c_tmp,
                        temp_type,
                        static_cast<int>(_info.m),
                        c_stride,
                        static_cast<int>(_info.batch),
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
                return INFINI_STATUS_SUCCESS;
            }));

        CHECK_STATUS(castF32ToBf16(c_tmp, c, _info.c_matrix, _info.batch, (kunlunStream_t)stream));
        xpu_wait(stream);
        return INFINI_STATUS_SUCCESS;
    }

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

    xpu_wait(stream);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::kunlun
