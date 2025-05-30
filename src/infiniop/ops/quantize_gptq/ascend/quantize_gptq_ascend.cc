#include "quantize_gptq_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

namespace op::quantize_gptq::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t c_ascend_desc;
    aclnnTensorDescriptor_t a_ascend_desc;
    aclnnTensorDescriptor_t w_ascend_desc;
    aclnnTensorDescriptor_t s_ascend_desc;
    aclnnTensorDescriptor_t z_ascend_desc;
    int32_t innerPrecise;
    aclOpExecutor *executor;

    ~Opaque() {
        delete c_ascend_desc;
        delete a_ascend_desc;
        delete w_ascend_desc;
        delete s_ascend_desc;
        delete z_ascend_desc;

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

    aclOpExecutor *executor = nullptr;
    aclnnTensorDescriptor_t c_ascend_desc = nullptr;
    aclnnTensorDescriptor_t a_ascend_desc = nullptr;
    aclnnTensorDescriptor_t w_ascend_desc = nullptr;
    aclnnTensorDescriptor_t s_ascend_desc = nullptr;
    aclnnTensorDescriptor_t z_ascend_desc = nullptr;

    std::vector<int64_t> c_shape = {static_cast<int64_t>(info.m), static_cast<int64_t>(info.n)};
    std::vector<int64_t> c_strides = {static_cast<int64_t>(info.n), static_cast<int64_t>(1)};
    c_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), c_shape, c_strides);

    std::vector<int64_t> a_shape = {static_cast<int64_t>(info.m), static_cast<int64_t>(info.k)};
    std::vector<int64_t> a_strides = {static_cast<int64_t>(info.k), static_cast<int64_t>(1)};
    a_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), a_shape, a_strides);

    std::vector<int64_t> w_shape = {static_cast<int64_t>(info.k), static_cast<int64_t>(info.n)};
    std::vector<int64_t> w_strides = {static_cast<int64_t>(info.n), static_cast<int64_t>(1)};
    w_ascend_desc = new aclnnTensorDescriptor(aclDataType::ACL_INT4, w_shape, w_strides);
    // aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    // w_ascend_desc->format = weightFormat;
    // std::vector<int64_t> nzShape = {static_cast<int64_t>(info.k / 64), static_cast<int64_t>(info.n / 16), 16, 64};
    // w_ascend_desc->storageNdim = 2;
    // w_ascend_desc->storageShape = nzShape;

    // aclInitTensor(nullptr, w_shape.data(), w_shape.size(), aclDataType::ACL_INT4, w_strides.data(), 0,
    //               weightFormat, nzShape.data(), nzShape.size(), nullptr);

    std::vector<int64_t> s_shape = {static_cast<int64_t>(info.num_groups), static_cast<int64_t>(info.n)};
    std::vector<int64_t> s_strides = {static_cast<int64_t>(info.n), static_cast<int64_t>(1)};
    s_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), s_shape, s_strides);

    std::vector<int64_t> z_shape = {static_cast<int64_t>(info.num_groups), static_cast<int64_t>(info.n)};
    std::vector<int64_t> z_strides = {static_cast<int64_t>(info.n), static_cast<int64_t>(1)};
    z_ascend_desc = new aclnnTensorDescriptor(toAclDataType(info.atype), z_shape, z_strides);

    size_t workspace_size = 0;

    aclTensor *yFp16 = c_ascend_desc->tensor;
    aclTensor *xFp16 = a_ascend_desc->tensor;
    aclTensor *weight = w_ascend_desc->tensor;
    aclTensor *anti_scale = s_ascend_desc->tensor;
    aclTensor *anti_offset = z_ascend_desc->tensor;
    int32_t innerPrecise = 1;
    CHECK_ACL(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(xFp16, weight, anti_scale, anti_offset, nullptr, nullptr, nullptr, 0, innerPrecise, yFp16, &workspace_size, &executor));

    aclSetAclOpExecutorRepeatable(executor);
    size_t min_workspace_size = workspace_size;
    *desc_ptr = new Descriptor(info, new Opaque{c_ascend_desc, a_ascend_desc, w_ascend_desc, s_ascend_desc, z_ascend_desc, innerPrecise, executor},
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
        size_t workspace_size = 0;

        aclTensor *weight = _opaque->w_ascend_desc->tensor;
        aclTensor *anti_scale = _opaque->s_ascend_desc->tensor;
        aclTensor *anti_offset = _opaque->z_ascend_desc->tensor;

        aclTensor *xFp16 = _opaque->a_ascend_desc->tensor;
        aclTensor *yFp16 = _opaque->c_ascend_desc->tensor;
        AclSetTensorAddr(_opaque->executor, 0, xFp16, (void *)a);
        AclSetTensorAddr(_opaque->executor, 1, weight, packed_weights);
        AclSetTensorAddr(_opaque->executor, 2, anti_scale, b_scale);
        AclSetTensorAddr(_opaque->executor, 3, anti_offset, zero);
        AclSetTensorAddr(_opaque->executor, 4, yFp16, c);

        CHECK_ACL(aclnnWeightQuantBatchMatmulV3(workspace, workspace_size, _opaque->executor, stream));

    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::quantize_gptq::ascend
