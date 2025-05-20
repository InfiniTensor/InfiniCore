#include "quantize_gptq_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include "aclnnop/aclnn_ascend_anti_quant.h"
#include "aclnnop/level2/aclnn_gemm.h"

namespace op::quantize_gptq::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c_ascend_desc;
    aclnnTensorDescriptor_t a_ascend_desc;
    aclnnTensorDescriptor_t w_ascend_desc;
    aclnnTensorDescriptor_t b_ascend_desc;
    aclnnTensorDescriptor_t z_ascend_desc;
    aclnnTensorDescriptor_t antiquant_ascend_desc;
    aclnnTensorDescriptor_t new_antiquant_ascend_desc;
    void *antiquant_addr;

    size_t workspacesize;

    aclOpExecutor *executor;

    ~Opaque() {
        delete c_ascend_desc;
        delete a_ascend_desc;
        delete w_ascend_desc;
        delete b_ascend_desc;
        delete z_ascend_desc;
        delete antiquant_ascend_desc;
        delete new_antiquant_ascend_desc;

        aclrtFree(antiquant_addr);

        // Delete useless executor

        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t packed_weights_desc,
    infiniopTensorDescriptor_t b_scale_desc,
    infiniopTensorDescriptor_t zero_desc) {
    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);

    auto result = MatmulGptqInfo::createMatmulGptqInfo(c_desc, a_desc, packed_weights_desc, b_scale_desc, zero_desc);
    CHECK_RESULT(result);
    MatmulGptqInfo info = result.take();

    void *antiquant_addr = nullptr;

    aclOpExecutor *antiquant_executor = nullptr;
    aclOpExecutor *executor = nullptr;
    aclnnTensorDescriptor_t c_ascend_desc = nullptr;
    aclnnTensorDescriptor_t a_ascend_desc = nullptr;
    aclnnTensorDescriptor_t w_ascend_desc = nullptr;
    aclnnTensorDescriptor_t b_ascend_desc = nullptr;
    aclnnTensorDescriptor_t z_ascend_desc = nullptr;
    aclnnTensorDescriptor_t antiquant_ascend_desc = nullptr;
    aclnnTensorDescriptor_t new_antiquant_ascend_desc = nullptr;

    std::vector<int64_t> c_shape = {static_cast<int64_t>(info.n), static_cast<int64_t>(info.m)};
    std::vector<int64_t> c_strides = {static_cast<int64_t>(info.m), static_cast<int64_t>(1)};
    c_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), c_shape, c_strides);

    std::vector<int64_t> a_shape = {static_cast<int64_t>(info.k), static_cast<int64_t>(info.m)};
    std::vector<int64_t> a_strides = {static_cast<int64_t>(info.m), static_cast<int64_t>(1)};
    a_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), a_shape, a_strides);

    std::vector<int64_t> w_shape = {static_cast<int64_t>(info.k / 8)};
    std::vector<int64_t> w_strides = {static_cast<int64_t>(1)};
    w_ascend_desc = new aclnnTensorDescriptor(aclDataType::ACL_INT32, w_shape, w_strides);

    std::vector<int64_t> b_shape = {static_cast<int64_t>(1)};
    std::vector<int64_t> b_strides = {static_cast<int64_t>(1)};
    b_ascend_desc = new aclnnTensorDescriptor(aclDataType::ACL_BF16, b_shape, b_strides);

    std::vector<int64_t> z_shape = {static_cast<int64_t>(1)};
    std::vector<int64_t> z_strides = {static_cast<int64_t>(1)};
    z_ascend_desc = new aclnnTensorDescriptor(aclDataType::ACL_BF16, z_shape, z_strides);

    size_t antiquant_workspace_size = 0;
    size_t matmul_workspace_size = 0;

    std::vector<int64_t> antiquant_shape = {static_cast<int64_t>(info.k)};
    std::vector<int64_t> antiquant_strides = {static_cast<int64_t>(1)};
    antiquant_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), antiquant_shape, antiquant_strides);

    aclTensor *weight = w_ascend_desc->tensor;
    aclTensor *antiquant = antiquant_ascend_desc->tensor;
    aclTensor *scale = b_ascend_desc->tensor;
    aclTensor *offset = z_ascend_desc->tensor;
    int64_t dstType = 1;
    bool sqrtMode = false;
    CHECK_ACL(aclnnAscendAntiQuantGetWorkspaceSize(weight, scale, offset, dstType, sqrtMode, antiquant, &antiquant_workspace_size, &antiquant_executor));

    CHECK_ACL(aclrtMalloc(&antiquant_addr, info.n * info.k * infiniSizeOf(info.atype), ACL_MEM_MALLOC_HUGE_FIRST));

    aclTensor *xFp16 = a_ascend_desc->tensor;
    aclTensor *yFp16 = c_ascend_desc->tensor;

    float alpha = 1.0f;
    float beta = 0.0f;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = 1;
    std::vector<int64_t> new_antiquant_shape = {static_cast<int64_t>(info.n), static_cast<int64_t>(info.k)};
    std::vector<int64_t> new_antiquant_strides = {static_cast<int64_t>(info.k), static_cast<int64_t>(1)};
    new_antiquant_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), new_antiquant_shape, new_antiquant_strides);
    aclTensor *new_antiquant = new_antiquant_ascend_desc->tensor;
    CHECK_ACL(aclnnGemmGetWorkspaceSize(new_antiquant, xFp16, yFp16, alpha, beta, transA, transB, yFp16, cubeMathType, &matmul_workspace_size, &executor));

    aclSetAclOpExecutorRepeatable(executor);
    size_t min_workspace_size = std::max(antiquant_workspace_size, matmul_workspace_size);
    *desc_ptr = new Descriptor(info, new Opaque{c_ascend_desc, a_ascend_desc, w_ascend_desc, b_ascend_desc, z_ascend_desc, antiquant_ascend_desc, new_antiquant_ascend_desc, antiquant_addr, min_workspace_size, executor},
                               min_workspace_size, handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::quant(
    void *workspace,
    size_t workspace_size,
    void *packed_weights,
    void *b_scale,
    void *zero,
    const void *a,
    const void *b,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    void *packed_weights,
    void *b_scale,
    void *zero,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_info.atype == INFINI_DTYPE_F16) {
        size_t antiquant_workspace_size = 0;
        size_t matmul_workspace_size = 0;
        aclOpExecutor *antiquant_executor = nullptr;
        aclTensor *weight = _opaque->w_ascend_desc->tensor;
        aclTensor *antiquant = _opaque->antiquant_ascend_desc->tensor;
        aclTensor *scale = _opaque->b_ascend_desc->tensor;
        aclTensor *offset = _opaque->z_ascend_desc->tensor;
        int64_t dstType = 1;
        bool sqrtMode = false;
        for (size_t i = 0; i < _info.n; i++) {
            AclSetTensorAddr(antiquant_executor, 0, weight, static_cast<void *>(static_cast<char *>(packed_weights) + i * static_cast<size_t>(_info.k / 8)));
            AclSetTensorAddr(antiquant_executor, 1, antiquant, static_cast<void *>(static_cast<char *>(_opaque->antiquant_addr) + i * _info.k));
            AclSetTensorAddr(antiquant_executor, 2, scale, static_cast<void *>(static_cast<char *>(b_scale) + i * _info.num_groups));
            AclSetTensorAddr(antiquant_executor, 3, offset, static_cast<void *>(static_cast<char *>(zero) + i * _info.num_groups));
            CHECK_ACL(aclnnAscendAntiQuantGetWorkspaceSize(weight, scale, offset, dstType, sqrtMode, antiquant, &antiquant_workspace_size, &antiquant_executor));
            CHECK_ACL(aclnnAscendAntiQuant(workspace, antiquant_workspace_size, antiquant_executor, stream));
        }

        aclTensor *xFp16 = _opaque->a_ascend_desc->tensor;
        aclTensor *new_antiquant = _opaque->new_antiquant_ascend_desc->tensor;
        aclTensor *yFp16 = _opaque->c_ascend_desc->tensor;
        AclSetTensorAddr(_opaque->executor, 0, xFp16, (void *)a);
        AclSetTensorAddr(_opaque->executor, 1, new_antiquant, _opaque->antiquant_addr);
        AclSetTensorAddr(_opaque->executor, 2, yFp16, c);

        CHECK_ACL(aclnnGemm(workspace, matmul_workspace_size, _opaque->executor, stream));

    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::quantize_gptq::ascend
