#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "linear_nvidia.cuh"

namespace op::linear::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t d_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t c_desc) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = d_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F8_E4M3, INFINI_DTYPE_F8_E5M2, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    auto result = op::linear::MatmulInfo::create(a_desc, b_desc, c_desc, d_desc, op::linear::MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    float alpha,
    const void *a,
    const void *a_scale,
    const void *b,
    const void *b_scale,
    float beta,
    const void *c,
    const void *c_scale,
    const void *bias,
    void *d,
    const void *d_scale,
    bool is_blockwise,
    bool is_a_1d_scaled,
    bool is_b_1d_scaled,
    void *workspace,
    size_t workspace_size,
    void *stream) const {
    cublasComputeType_t compute_type;
    int returnedResults = 0;
    const int8_t fast_accum_mode = 0;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtEpilogue_t epilogue;

    cublasLtMatmulDesc_t lt_desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, c_layout = NULL, d_layout = NULL;

    cudaDataType a_type, b_type, c_type, d_type, scale_type, bias_type;

    switch (_info.a_matrix.dtype) {
    case INFINI_DTYPE_F8_E4M3:
        a_type = CUDA_R_8F_E4M3;
        compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
        switch (_info.b_matrix.dtype) {
        case INFINI_DTYPE_F8_E4M3:
            b_type = CUDA_R_8F_E4M3;
            switch (_info.c_matrix.dtype) {
            case INFINI_DTYPE_BF16:
                c_type = CUDA_R_16BF;
                switch (_info.d_matrix.dtype) {
                case INFINI_DTYPE_BF16:
                    d_type = CUDA_R_16BF;
                    bias_type = CUDA_R_16BF;
                    break;
                case INFINI_DTYPE_F8_E4M3:
                    d_type = CUDA_R_8F_E4M3;
                    bias_type = CUDA_R_16BF;
                    break;
                default:
                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                }
                break;
            case INFINI_DTYPE_F16:
                c_type = CUDA_R_16F;
                switch (_info.d_matrix.dtype) {
                case INFINI_DTYPE_F16:
                    d_type = CUDA_R_16F;
                    bias_type = CUDA_R_16F;
                    break;
                case INFINI_DTYPE_F8_E4M3:
                    d_type = CUDA_R_8F_E4M3;
                    bias_type = CUDA_R_16F;
                    break;
                default:
                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                }
                break;
            case INFINI_DTYPE_F32:
                c_type = CUDA_R_32F;
                switch (_info.d_matrix.dtype) {
                case INFINI_DTYPE_F32:
                    d_type = CUDA_R_32F;
                    bias_type = CUDA_R_32F;
                    break;
                default:
                    return INFINI_STATUS_BAD_TENSOR_DTYPE;
                }
                break;
            default:
                return INFINI_STATUS_NOT_IMPLEMENTED;
            }
            break;
        default:
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }
        break;

    case INFINI_DTYPE_F16:
        a_type = b_type = c_type = d_type = CUDA_R_16F;
        bias_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
        break;

    case INFINI_DTYPE_BF16:
        a_type = b_type = c_type = d_type = CUDA_R_16BF;
        bias_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
        break;

    case INFINI_DTYPE_F32:
        a_type = b_type = c_type = d_type = CUDA_R_32F;
        bias_type = CUDA_R_16BF;
        compute_type = CUBLAS_COMPUTE_32F;
        scale_type = CUDA_R_32F;
        break;

    default:
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }
    // auto info = result.take();

    /* To use tensor- or block-scaled FP8 kernels:
     * A must be transposed and B non-transposed (The “TN” format)
     * on Ada (compute capability 8.9), Hopper (compute capability 9.0),
     * and Blackwell GeForce (compute capability 12.x) GPUs.
     */
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&lt_desc, compute_type, scale_type));
    BlasMatrix a_matrix = _info.a_matrix, b_matrix = _info.b_matrix;
    cublasOperation_t op_a, op_b;
    if (_info.a_matrix.dtype == INFINI_DTYPE_F8_E4M3) {
        bool transa = true;
        bool transb = false;
        a_matrix = _info.b_matrix;
        b_matrix = _info.a_matrix;

        const int m = transa ? a_matrix.rows : a_matrix.cols;
        const int k = transa ? a_matrix.cols : a_matrix.rows;
        const int n = transb ? b_matrix.cols : b_matrix.rows;
        int lda = k, ldb = k, ldc = m, ldd = m;

        op_a = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
        op_b = transb ? CUBLAS_OP_T : CUBLAS_OP_N;

// Note: in cuBLAS term, tensor name A and B are swapped.
#if SUPPORT_FP8_BLOCKWISE_SCALE
        if (is_blockwise) {
            cublasLtMatmulMatrixScale_t a_scale_mode, b_scale_mode;
            if (is_b_1d_scaled && is_a_1d_scaled) {
                a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
                b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
            } else if (!is_b_1d_scaled && is_a_1d_scaled) {
                // So this corresponds to 2Dx1D GEMM.
                a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
                b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
            } else if (is_b_1d_scaled && !is_a_1d_scaled) {
                // So this corresponds to 1Dx2D GEMM.
                a_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
                b_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
            } else {
                return INFINI_STATUS_NOT_IMPLEMENTED;
            }
            CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
            CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
        }
#endif

        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, a_type, op_a == CUBLAS_OP_N ? m : k, op_a == CUBLAS_OP_N ? k : m, lda));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, b_type, op_b == CUBLAS_OP_N ? k : n, op_b == CUBLAS_OP_N ? n : k, ldb));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, c_type, m, n, ldc));

        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_layout, d_type, m, n, ldd));

    } else {
        op_a = _info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
        op_b = _info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, a_type, op_a ? _info.k : _info.m, op_a ? _info.m : _info.k, _info.a_matrix.ld()));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, b_type, op_b ? _info.n : _info.k, op_b ? _info.k : _info.n, _info.b_matrix.ld()));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, c_type, _info.m, _info.n, _info.c_matrix.ld()));
        CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_layout, d_type, _info.m, _info.n, _info.c_matrix.ld()));
    }

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    // CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_PE, &, sizeof(scale_type)));
    // cuBlasLt requires C in fp8 mode to be BF16 or FP32

    int batch = static_cast<int>(_info.batch);

    if (batch > 1) {
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));

        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_matrix.stride, sizeof(a_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_matrix.stride, sizeof(b_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.c_matrix.stride, sizeof(_info.c_matrix.stride)));
        CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_info.c_matrix.stride, sizeof(_info.c_matrix.stride)));
    }

    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum_mode, sizeof(fast_accum_mode)));

    if (_info.is_transed && a_type != CUDA_R_8F_E4M3 && a_type != CUDA_R_8F_E5M2) {
        std::swap(a, b);
    } else if (a_type == CUDA_R_8F_E4M3 || a_type == CUDA_R_8F_E5M2) {
        std::swap(a, b);
        std::swap(a_scale, b_scale);
    }

    if ((a_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E4M3) && (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)c % 16) != 0 || ((uintptr_t)d % 16) != 0)) {
        return INFINI_STATUS_NOT_ALIGNED;
    }

    if (a_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    }
    if (b_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    }
    if (c_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &c_scale, sizeof(c_scale)));
    }
    if (d_scale) {
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
    }

    if (bias) {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
        CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(lt_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    CHECK_STATUS(_opaque->internal->useCublasLt(
        (cudaStream_t)stream,
        [&](cublasLtHandle_t handle) {
            CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
                handle,
                lt_desc,
                a_layout,
                b_layout,
                c_layout,
                d_layout,
                preference,
                1,
                &heuristicResult,
                &returnedResults));
            if (returnedResults == 0) {
                return INFINI_STATUS_NOT_IMPLEMENTED;
            }

            CHECK_CUBLASLT(
                cublasLtMatmul(
                    handle,
                    lt_desc,
                    &alpha,
                    a,
                    a_layout,
                    b,
                    b_layout,
                    &beta,
                    c,
                    c_layout,
                    d,
                    d_layout,
                    nullptr,
                    workspace,
                    workspace_size,
                    reinterpret_cast<cudaStream_t>(stream)));

            if (preference) {
                CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
            }
            if (d_layout) {
                CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(d_layout));
            }
            if (c_layout) {
                CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(c_layout));
            }
            if (b_layout) {
                CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(b_layout));
            }
            if (a_layout) {
                CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(a_layout));
            }
            if (lt_desc) {
                CHECK_CUBLASLT(cublasLtMatmulDescDestroy(lt_desc));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::linear::nvidia
