#include "gemm_hcdnn.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"
#include "../../../devices/metax/metax_kernel_common.h"

#include <hcdnn/hcdnn.h>
#include <hcdnn/hcdnn_backend.h>
#include <hpcc_fp16.h>
#include <array>
#include <memory>

namespace op::gemm::hcdnn {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculate(
    const MatmulInfo &info,
    std::shared_ptr<device::metax::Handle::Internal> &_internal,
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream)
{
    // 0. For HCDNN development, refer to the official documentation and the following headers:
    // - /opt/hpcc/include/hcdnn/hcdnn.h
    // - /opt/hpcc/include/hcdnn/hcdnn_backend.h
    // - /opt/hpcc/include/hcdnn/hcdnn_ops_infer.h

    // 1. Use _internal->useMcdnn to manage HCDNN handle
    return _internal->useMcdnn((hcStream_t)stream, [&](hcdnnHandle_t hcdnn_handle) -> infiniStatus_t {

        // 2. Create HCDNN tensor descriptors for A, B, C
        hcdnnTensorDescriptor_t aDesc, bDesc, cDesc;
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&aDesc));
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&bDesc));
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&cDesc));

        // 3. Determine HCDNN data type
        hcdnnDataType_t hcdnn_dtype;
        if constexpr (std::is_same<Tdata, __half>::value) {
            hcdnn_dtype = HCDNN_DATA_HALF;
        } else if constexpr (std::is_same<Tdata, __hpcc_bfloat16>::value) {
            hcdnn_dtype = HCDNN_DATA_BFLOAT16;
        } else {
            hcdnn_dtype = HCDNN_DATA_FLOAT;
        }

        // 4. Configure tensor A (left)
        std::array<int, 3> a_dims;
        std::array<int, 3> a_strides;
        if (info.a_matrix.col_stride != 1) {
            // Transposed: [batch, k, m]
            a_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.k),
                      static_cast<int>(info.m) };
        } else {
            // Normal: [batch, m, k]
            a_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.m),
                      static_cast<int>(info.k) };
        }
        a_strides = { static_cast<int>(info.a_matrix.stride),
                     static_cast<int>(info.a_matrix.ld()),
                     1 };
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            aDesc,
            HCDNN_TENSOR_NCHW,
            hcdnn_dtype,
            3,
            a_dims.data(),
            a_strides.data()));

        // 5. Configure tensor B (right)
        std::array<int, 3> b_dims;
        std::array<int, 3> b_strides;
        if (info.b_matrix.col_stride != 1) {
            // Transposed: [batch, n, k]
            b_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.n),
                      static_cast<int>(info.k) };
        } else {
            // Normal: [batch, k, n]
            b_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.k),
                      static_cast<int>(info.n) };
        }
        b_strides = { static_cast<int>(info.b_matrix.stride),
                     static_cast<int>(info.b_matrix.ld()),
                     1 };
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            bDesc,
            HCDNN_TENSOR_NCHW,
            hcdnn_dtype,
            3,
            b_dims.data(),
            b_strides.data()));

        // 6. Configure tensor C (output)
        std::array<int, 3> c_dims = { static_cast<int>(info.batch),
                                    static_cast<int>(info.m),
                                    static_cast<int>(info.n) };
        std::array<int, 3> c_strides = { static_cast<int>(info.c_matrix.stride),
                                        static_cast<int>(info.c_matrix.ld()),
                                        1 };
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            cDesc,
            HCDNN_TENSOR_NCHW,
            hcdnn_dtype,
            3,
            c_dims.data(),
            c_strides.data()));

        // 7. Determine transpose flags
        bool trans_a = (info.a_matrix.col_stride != 1);
        bool trans_b = (info.b_matrix.col_stride != 1);

        // 8. Use HCDNN backend API for matmul
        // Create backend tensor descriptors
        hcdnnBackendDescriptor_t aTensorDesc, bTensorDesc, cTensorDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &aTensorDesc));
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &bTensorDesc));
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &cTensorDesc));

        // Set tensor descriptor attributes - convert to int64_t
        int64_t a_uid = 1, b_uid = 2, c_uid = 3;
        std::array<int64_t, 3> a_dims64, b_dims64, c_dims64;
        std::array<int64_t, 3> a_strides64, b_strides64, c_strides64;
        for (int i = 0; i < 3; i++) {
            a_dims64[i] = a_dims[i];
            a_strides64[i] = a_strides[i];
            b_dims64[i] = b_dims[i];
            b_strides64[i] = b_strides[i];
            c_dims64[i] = c_dims[i];
            c_strides64[i] = c_strides[i];
        }

        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_DIMENSIONS,
                                             HCDNN_TYPE_INT64, 3, a_dims64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_STRIDES,
                                             HCDNN_TYPE_INT64, 3, a_strides64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_DATA_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &hcdnn_dtype));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_UNIQUE_ID,
                                             HCDNN_TYPE_INT64, 1, &a_uid));
        CHECK_MCDNN(hcdnnBackendFinalize(aTensorDesc));

        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_DIMENSIONS,
                                             HCDNN_TYPE_INT64, 3, b_dims64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_STRIDES,
                                             HCDNN_TYPE_INT64, 3, b_strides64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_DATA_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &hcdnn_dtype));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_UNIQUE_ID,
                                             HCDNN_TYPE_INT64, 1, &b_uid));
        CHECK_MCDNN(hcdnnBackendFinalize(bTensorDesc));

        CHECK_MCDNN(hcdnnBackendSetAttribute(cTensorDesc, HCDNN_ATTR_TENSOR_DIMENSIONS,
                                             HCDNN_TYPE_INT64, 3, c_dims64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(cTensorDesc, HCDNN_ATTR_TENSOR_STRIDES,
                                             HCDNN_TYPE_INT64, 3, c_strides64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(cTensorDesc, HCDNN_ATTR_TENSOR_DATA_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &hcdnn_dtype));
        CHECK_MCDNN(hcdnnBackendSetAttribute(cTensorDesc, HCDNN_ATTR_TENSOR_UNIQUE_ID,
                                             HCDNN_TYPE_INT64, 1, &c_uid));
        CHECK_MCDNN(hcdnnBackendFinalize(cTensorDesc));

        // Create matmul descriptor
        hcdnnBackendDescriptor_t matmulDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_MATMUL_DESCRIPTOR, &matmulDesc));

        // Set compute type (use same as data type for now)
        hcdnnDataType_t compute_type = hcdnn_dtype;
        CHECK_MCDNN(hcdnnBackendSetAttribute(matmulDesc, HCDNN_ATTR_MATMUL_COMP_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &compute_type));

        // Create operation matmul descriptor
        hcdnnBackendDescriptor_t opMatmulDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &opMatmulDesc));

        // Set operation attributes
        CHECK_MCDNN(hcdnnBackendSetAttribute(opMatmulDesc, HCDNN_ATTR_OPERATION_MATMUL_ADESC,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &aTensorDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(opMatmulDesc, HCDNN_ATTR_OPERATION_MATMUL_BDESC,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &bTensorDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(opMatmulDesc, HCDNN_ATTR_OPERATION_MATMUL_CDESC,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cTensorDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(opMatmulDesc, HCDNN_ATTR_OPERATION_MATMUL_DESC,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmulDesc));

        // Finalize operation descriptor
        CHECK_MCDNN(hcdnnBackendFinalize(opMatmulDesc));

        // Create operation graph
        hcdnnBackendDescriptor_t opGraphDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &opGraphDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(opGraphDesc, HCDNN_ATTR_OPERATIONGRAPH_OPS,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opMatmulDesc));
        CHECK_MCDNN(hcdnnBackendFinalize(opGraphDesc));

        // Create engine config (use default)
        hcdnnBackendDescriptor_t engineConfigDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfigDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(engineConfigDesc, HCDNN_ATTR_ENGINECFG_OPERATION_GRAPH,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &opGraphDesc));
        CHECK_MCDNN(hcdnnBackendFinalize(engineConfigDesc));

        // Create execution plan
        hcdnnBackendDescriptor_t executionPlanDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &executionPlanDesc));
        CHECK_MCDNN(hcdnnBackendSetAttribute(executionPlanDesc, HCDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                             HCDNN_TYPE_HANDLE, 1, &hcdnn_handle));
        CHECK_MCDNN(hcdnnBackendSetAttribute(executionPlanDesc, HCDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                             HCDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engineConfigDesc));
        CHECK_MCDNN(hcdnnBackendFinalize(executionPlanDesc));

        // Create variant pack with data pointers
        hcdnnBackendDescriptor_t variantPackDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &variantPackDesc));

        int64_t uids[] = {a_uid, b_uid, c_uid};
        void* data_ptrs[] = {const_cast<void*>(a), const_cast<void*>(b), c};
        CHECK_MCDNN(hcdnnBackendSetAttribute(variantPackDesc, HCDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                             HCDNN_TYPE_INT64, 3, uids));
        CHECK_MCDNN(hcdnnBackendSetAttribute(variantPackDesc, HCDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                             HCDNN_TYPE_VOID_PTR, 3, data_ptrs));
        if (workspace && workspace_size > 0) {
            CHECK_MCDNN(hcdnnBackendSetAttribute(variantPackDesc, HCDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                 HCDNN_TYPE_VOID_PTR, 1, &workspace));
        }
        CHECK_MCDNN(hcdnnBackendFinalize(variantPackDesc));

        // Execute
        CHECK_MCDNN(hcdnnBackendExecute(hcdnn_handle, executionPlanDesc, variantPackDesc));

        // Cleanup backend descriptors
        hcdnnBackendDestroyDescriptor(variantPackDesc);
        hcdnnBackendDestroyDescriptor(executionPlanDesc);
        hcdnnBackendDestroyDescriptor(engineConfigDesc);
        hcdnnBackendDestroyDescriptor(opGraphDesc);
        hcdnnBackendDestroyDescriptor(opMatmulDesc);
        hcdnnBackendDestroyDescriptor(matmulDesc);
        hcdnnBackendDestroyDescriptor(cTensorDesc);
        hcdnnBackendDestroyDescriptor(bTensorDesc);
        hcdnnBackendDestroyDescriptor(aTensorDesc);

        // Cleanup regular tensor descriptors
        hcdnnDestroyTensorDescriptor(aDesc);
        hcdnnDestroyTensorDescriptor(bDesc);
        hcdnnDestroyTensorDescriptor(cDesc);

        return INFINI_STATUS_SUCCESS;
    });
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *c,
                                     float beta,
                                     const void *a,
                                     const void *b,
                                     float alpha,
                                     void *stream) const {
    switch (_dtype) {
        case INFINI_DTYPE_F16:
            return hcdnn::calculate<__half>(_info, _opaque->internal, workspace, workspace_size, c, beta, a, b, alpha, stream);
        case INFINI_DTYPE_F32:
            return hcdnn::calculate<float>(_info, _opaque->internal, workspace, workspace_size, c, beta, a, b, alpha, stream);
        case INFINI_DTYPE_BF16:
            return hcdnn::calculate<__hpcc_bfloat16>(_info, _opaque->internal, workspace, workspace_size, c, beta, a, b, alpha, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::gemm::hcdnn
