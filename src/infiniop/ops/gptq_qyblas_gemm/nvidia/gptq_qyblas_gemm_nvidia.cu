#if defined ENABLE_QY_API
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "dlblas_ext.h"
#include "gptq_qyblas_gemm_nvidia.cuh"

namespace op::gptq_qyblas_gemm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc) {

    auto info = GptqQyblasGemmInfo::createGptqQyblasGemmInfo(out_desc, a_desc, b_desc, b_scales_desc, b_zeros_desc);

    CHECK_RESULT(info);

    size_t workspace_size = 0;
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *out,
                                     const void *a,
                                     const void *b,
                                     void *b_scales,
                                     void *b_zeros,
                                     int64_t quant_type,
                                     int64_t bit,
                                     void *stream) const {

    int64_t M = static_cast<int64_t>(_info.M);
    int64_t K = static_cast<int64_t>(_info.K);
    int64_t N = static_cast<int64_t>(_info.N);
    int64_t scales_size_0 = static_cast<int64_t>(_info.scales_size_0);
    int64_t scales_size_1 = static_cast<int64_t>(_info.scales_size_1);
    int64_t lda = static_cast<int64_t>(_info.lda);
    int64_t ldb = static_cast<int64_t>(_info.ldb);
    int64_t result_ld = static_cast<int64_t>(_info.result_ld);
    bool transpose_mat_1 = _info.transpose_mat_1;
    bool transpose_mat_2 = _info.transpose_mat_2;

    cudaDataType_t computeType_ = (cudaDataType_t)CUDA_R_32F;
    cudaDataType_t kernel_Atype_, kernel_Btype_, kernel_Ctype_, kernel_Stype_, kernel_Ztype_;

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        kernel_Atype_ = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        kernel_Atype_ = CUDA_R_16BF;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (quant_type == 0) {
        if (8 == bit) {
            kernel_Atype_ = (cudaDataType_t)CUDA_R_8U;
        }

        if (4 == bit) {
            kernel_Atype_ = (cudaDataType_t)CUDA_R_4U;
            K = K * 2;
        }
    }

    switch (_info.weight_dtype) {
    case INFINI_DTYPE_F8:
        kernel_Btype_ = (cudaDataType_t)CUDA_R_8F_E4M3;
        break;
    case INFINI_DTYPE_U8:
        kernel_Btype_ = CUDA_R_8U;
        break;
    case INFINI_DTYPE_I8:
        kernel_Btype_ = CUDA_R_8I;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    kernel_Ctype_ = kernel_Atype_;

    switch (_info.scales_dtype) {
    case INFINI_DTYPE_F32:
        kernel_Stype_ = CUDA_R_32F;
        break;
    case INFINI_DTYPE_F16:
        kernel_Stype_ = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        kernel_Stype_ = CUDA_R_16BF;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    switch (_info.zeros_dtype) {
    case INFINI_DTYPE_F32:
        kernel_Ztype_ = CUDA_R_32F;
        break;
    case INFINI_DTYPE_F16:
        kernel_Ztype_ = CUDA_R_16F;
        break;
    case INFINI_DTYPE_BF16:
        kernel_Ztype_ = CUDA_R_16BF;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    dlblasExtQuantParametersV2_t extParameters;

    if (quant_type == 0) {
        extParameters.a_group_size_m = M / scales_size_0;
        extParameters.a_group_size_k = K / scales_size_1;
        extParameters.a_zeropoints_type = kernel_Ztype_;
        extParameters.a_zeropoints = b_zeros;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_scales = b_scales;
    } else if (quant_type == 1) {
        extParameters.a_group_size_m = 1;
        extParameters.a_group_size_k = K;
        extParameters.a_zeropoints = nullptr;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_scales = b_scales;

    } else if (quant_type == 2 || quant_type == 3) {
        // calculate block_shape according weight/scales shape
        int block_shape = 128;
        while ((N + block_shape - 1) / block_shape < scales_size_0) {
            block_shape /= 2;
            if (block_shape < 32) {
                fprintf(stderr,
                        "INTERNAL ASSERT FAILED: block_shape >= 32\n"
                        "Invalid fp blockwise linear arguments. Weight: [%d, %d]. Scales: [%d, %d].\n",
                        (int)N, (int)K, (int)scales_size_0, (int)scales_size_1);
                abort();
            }
        }
        if (!((K + block_shape - 1) / block_shape == scales_size_1)) {
            fprintf(stderr,
                    "CHECK FAILED: (K + block_shape - 1) / block_shape == scales_size_1\n");
            abort();
        }
        extParameters.a_group_size_m = block_shape;
        extParameters.a_group_size_k = block_shape;
        extParameters.a_scales_type = kernel_Stype_;
        extParameters.a_zeropoints = nullptr;
        extParameters.a_scales = b_scales;
    }

    cublasOperation_t transa = transpose_mat_2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transpose_mat_1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (_info.dtype == INFINI_DTYPE_F16 || _info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(_opaque->internal->useCublas(
            (cudaStream_t)stream,
            [&](cublasHandle_t handle) {
                CHECK_CUBLAS(
                    dlblasGemmExV2(handle,
                                   transa,
                                   transb,
                                   N,
                                   M,
                                   K,
                                   &alpha,
                                   b,
                                   kernel_Btype_,
                                   ldb,
                                   a,
                                   kernel_Atype_,
                                   lda,
                                   &beta,
                                   out,
                                   kernel_Ctype_,
                                   result_ld,
                                   computeType_,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                                   &extParameters));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gptq_qyblas_gemm::nvidia
#endif
