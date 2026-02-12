#include "gemm_hcdnn.h"
#include "../../../../devices/metax/metax_common.h"
#include "../../../../devices/metax/metax_handle.h"
#include "../../../../devices/metax/metax_kernel_common.h"

#include <hcdnn/hcdnn.h>
#include <hcdnn/hcdnn_backend.h>
#include <common/hpcc_fp16.h>
#include <array>
#include <memory>
#include <algorithm>

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

        // Calculate byte alignment based on data type size
        // According to mcDNN API: alignment must be divisible by data type size
        // Use sizeof to get the actual data type size
        int64_t byte_alignment = static_cast<int64_t>(sizeof(Tdata));

        // 4. Configure tensor A (left)
        // Always use 3D descriptor: [batch, m, k] or [batch, k, m]
        std::array<int, 3> a_dims;
        std::array<int, 3> a_strides;
        if (info.a_matrix.col_stride != 1) {
            // Transposed: [batch, k, m]
            a_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.k),
                      static_cast<int>(info.m) };
            int inner_stride = static_cast<int>(info.a_matrix.ld());
            int inner_dim = static_cast<int>(info.m);
            // Compute batch stride: ensure it satisfies stride[0] >= stride[1] * dim[1]
            int min_batch_stride = inner_stride * inner_dim;
            int batch_stride = (info.a_matrix.stride != 0) ? static_cast<int>(info.a_matrix.stride)
                                                           : static_cast<int>(info.k * info.m);
            batch_stride = std::max(batch_stride, min_batch_stride);
            a_strides = { batch_stride, inner_stride, 1 };
        } else {
            // Normal: [batch, m, k]
            a_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.m),
                      static_cast<int>(info.k) };
            int inner_stride = static_cast<int>(info.a_matrix.ld());
            int inner_dim = static_cast<int>(info.k);
            // Compute batch stride: ensure it satisfies stride[0] >= stride[1] * dim[1]
            int min_batch_stride = inner_stride * inner_dim;
            int batch_stride = (info.a_matrix.stride != 0) ? static_cast<int>(info.a_matrix.stride)
                                                           : static_cast<int>(info.m * info.k);
            batch_stride = std::max(batch_stride, min_batch_stride);
            a_strides = { batch_stride, inner_stride, 1 };
        }
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            aDesc,
            hcdnn_dtype,
            3,
            a_dims.data(),
            a_strides.data()));

        // 5. Configure tensor B (right)
        // Always use 3D descriptor: [batch, k, n] or [batch, n, k]
        std::array<int, 3> b_dims;
        std::array<int, 3> b_strides;
        if (info.b_matrix.col_stride != 1) {
            // Transposed: [batch, n, k]
            b_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.n),
                      static_cast<int>(info.k) };
            int inner_stride = static_cast<int>(info.b_matrix.ld());
            int inner_dim = static_cast<int>(info.k);
            // Compute batch stride: ensure it satisfies stride[0] >= stride[1] * dim[1]
            int min_batch_stride = inner_stride * inner_dim;
            int batch_stride = (info.b_matrix.stride != 0) ? static_cast<int>(info.b_matrix.stride)
                                                           : static_cast<int>(info.n * info.k);
            batch_stride = std::max(batch_stride, min_batch_stride);
            b_strides = { batch_stride, inner_stride, 1 };
        } else {
            // Normal: [batch, k, n]
            b_dims = { static_cast<int>(info.batch),
                      static_cast<int>(info.k),
                      static_cast<int>(info.n) };
            int inner_stride = static_cast<int>(info.b_matrix.ld());
            int inner_dim = static_cast<int>(info.n);
            // Compute batch stride: ensure it satisfies stride[0] >= stride[1] * dim[1]
            int min_batch_stride = inner_stride * inner_dim;
            int batch_stride = (info.b_matrix.stride != 0) ? static_cast<int>(info.b_matrix.stride)
                                                           : static_cast<int>(info.k * info.n);
            batch_stride = std::max(batch_stride, min_batch_stride);
            b_strides = { batch_stride, inner_stride, 1 };
        }
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            bDesc,
            hcdnn_dtype,
            3,
            b_dims.data(),
            b_strides.data()));

        // 6. Configure tensor C (output)
        // Always use 3D descriptor: [batch, m, n]
        std::array<int, 3> c_dims = { static_cast<int>(info.batch),
                                    static_cast<int>(info.m),
                                    static_cast<int>(info.n) };
        int c_inner_stride = static_cast<int>(info.c_matrix.ld());
        int c_inner_dim = static_cast<int>(info.n);
        // Compute batch stride: ensure it satisfies stride[0] >= stride[1] * dim[1]
        int min_c_batch_stride = c_inner_stride * c_inner_dim;
        int c_batch_stride_int = (info.c_matrix.stride != 0) ? static_cast<int>(info.c_matrix.stride)
                                                              : static_cast<int>(info.m * info.n);
        c_batch_stride_int = std::max(c_batch_stride_int, min_c_batch_stride);
        std::array<int, 3> c_strides = { c_batch_stride_int, c_inner_stride, 1 };
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            cDesc,
            hcdnn_dtype,
            3,
            c_dims.data(),
            c_strides.data()));

        // 7. Determine transpose flags (currently unused but kept for future use)
        // bool trans_a = (info.a_matrix.col_stride != 1);
        // bool trans_b = (info.b_matrix.col_stride != 1);

        // 8. Use HCDNN backend API for matmul
        // Create backend tensor descriptors
        hcdnnBackendDescriptor_t aTensorDesc, bTensorDesc, cTensorDesc;
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &aTensorDesc));
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &bTensorDesc));
        CHECK_MCDNN(hcdnnBackendCreateDescriptor(HCDNN_BACKEND_TENSOR_DESCRIPTOR, &cTensorDesc));

        // Set tensor descriptor attributes - convert to int64_t
        // Backend API requires strides to match regular tensor descriptor exactly
        int64_t a_uid = 1, b_uid = 2, c_uid = 3;
        std::array<int64_t, 3> a_dims64, b_dims64, c_dims64;
        std::array<int64_t, 3> a_strides64, b_strides64, c_strides64;

        // Convert regular tensor descriptor strides to int64_t for backend API
        // Use exact same values as regular descriptors to ensure consistency
        if (info.a_matrix.col_stride != 1) {
            a_dims64 = { static_cast<int64_t>(a_dims[0]), static_cast<int64_t>(a_dims[1]), static_cast<int64_t>(a_dims[2]) };
            a_strides64 = { static_cast<int64_t>(a_strides[0]), static_cast<int64_t>(a_strides[1]), static_cast<int64_t>(a_strides[2]) };
        } else {
            a_dims64 = { static_cast<int64_t>(a_dims[0]), static_cast<int64_t>(a_dims[1]), static_cast<int64_t>(a_dims[2]) };
            a_strides64 = { static_cast<int64_t>(a_strides[0]), static_cast<int64_t>(a_strides[1]), static_cast<int64_t>(a_strides[2]) };
        }

        if (info.b_matrix.col_stride != 1) {
            b_dims64 = { static_cast<int64_t>(b_dims[0]), static_cast<int64_t>(b_dims[1]), static_cast<int64_t>(b_dims[2]) };
            b_strides64 = { static_cast<int64_t>(b_strides[0]), static_cast<int64_t>(b_strides[1]), static_cast<int64_t>(b_strides[2]) };
        } else {
            b_dims64 = { static_cast<int64_t>(b_dims[0]), static_cast<int64_t>(b_dims[1]), static_cast<int64_t>(b_dims[2]) };
            b_strides64 = { static_cast<int64_t>(b_strides[0]), static_cast<int64_t>(b_strides[1]), static_cast<int64_t>(b_strides[2]) };
        }

        c_dims64 = { static_cast<int64_t>(c_dims[0]), static_cast<int64_t>(c_dims[1]), static_cast<int64_t>(c_dims[2]) };
        c_strides64 = { static_cast<int64_t>(c_strides[0]), static_cast<int64_t>(c_strides[1]), static_cast<int64_t>(c_strides[2]) };

        // Set byte alignment first (required attribute)
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                             HCDNN_TYPE_INT64, 1, &byte_alignment));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_DIMENSIONS,
                                             HCDNN_TYPE_INT64, 3, a_dims64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_STRIDES,
                                             HCDNN_TYPE_INT64, 3, a_strides64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_DATA_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &hcdnn_dtype));
        CHECK_MCDNN(hcdnnBackendSetAttribute(aTensorDesc, HCDNN_ATTR_TENSOR_UNIQUE_ID,
                                             HCDNN_TYPE_INT64, 1, &a_uid));
        CHECK_MCDNN(hcdnnBackendFinalize(aTensorDesc));

        // Set byte alignment first (required attribute)
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                             HCDNN_TYPE_INT64, 1, &byte_alignment));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_DIMENSIONS,
                                             HCDNN_TYPE_INT64, 3, b_dims64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_STRIDES,
                                             HCDNN_TYPE_INT64, 3, b_strides64.data()));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_DATA_TYPE,
                                             HCDNN_TYPE_DATA_TYPE, 1, &hcdnn_dtype));
        CHECK_MCDNN(hcdnnBackendSetAttribute(bTensorDesc, HCDNN_ATTR_TENSOR_UNIQUE_ID,
                                             HCDNN_TYPE_INT64, 1, &b_uid));
        CHECK_MCDNN(hcdnnBackendFinalize(bTensorDesc));

        // Set byte alignment first (required attribute)
        CHECK_MCDNN(hcdnnBackendSetAttribute(cTensorDesc, HCDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                                             HCDNN_TYPE_INT64, 1, &byte_alignment));
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
        CHECK_MCDNN(hcdnnBackendSetAttribute(engineConfigDesc, HCDNN_ATTR_ENGINE_OPERATION_GRAPH,
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
