#include "matmul_all_reduce_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

#include <aclnnop/aclnn_matmul_all_reduce.h>

namespace op::matmul_all_reduce::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t output;
    aclnnTensorDescriptor_t input;
    aclnnTensorDescriptor_t weight;
    aclnnTensorDescriptor_t bias;
    aclOpExecutor *executor;

    ~Opaque() {
        delete output;
        delete input;
        delete weight;
        delete bias;
        aclDestroyAclOpExecutor(executor);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    const char *group_name) {
    if (desc_ptr == nullptr || group_name == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
    CHECK_API_OR(output_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(weight_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->dtype() == dtype, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);

    CHECK_API_OR(input_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(weight_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_desc->ndim() == 2, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->ndim() == 1, true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);

    const auto &input_shape = input_desc->shape();
    const auto &weight_shape = weight_desc->shape();
    const auto &output_shape = output_desc->shape();
    CHECK_API_OR(input_shape[1] == weight_shape[0], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_shape[0] == input_shape[0], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_shape[1] == weight_shape[1], true,
                 return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(bias_desc == nullptr || bias_desc->shape()[0] == output_shape[1],
                 true, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto output = new aclnnTensorDescriptor(output_desc);
    auto input = new aclnnTensorDescriptor(input_desc);
    auto weight = new aclnnTensorDescriptor(weight_desc);
    auto bias = bias_desc == nullptr
                  ? nullptr
                  : new aclnnTensorDescriptor(bias_desc);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnMatmulAllReduceGetWorkspaceSize(
        input->tensor,
        weight->tensor,
        bias == nullptr ? nullptr : bias->tensor,
        group_name,
        "sum",
        0,
        1,
        output->tensor,
        &workspace_size,
        &executor));
    CHECK_ACL(aclSetAclOpExecutorRepeatable(executor));

    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);
    *desc_ptr = new Descriptor(
        workspace_size,
        new Opaque{output, input, weight, bias, executor},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream) const {
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    CHECK_ACL(AclSetTensorAddr(
        _opaque->executor, 0, _opaque->input->tensor,
        const_cast<void *>(input)));
    CHECK_ACL(AclSetTensorAddr(
        _opaque->executor, 1, _opaque->weight->tensor,
        const_cast<void *>(weight)));
    if (_opaque->bias != nullptr) {
        CHECK_ACL(AclSetTensorAddr(
            _opaque->executor, 2, _opaque->bias->tensor,
            const_cast<void *>(bias)));
    }
    CHECK_ACL(AclSetTensorAddr(
        _opaque->executor, 3, _opaque->output->tensor, output));
    CHECK_ACL(aclnnMatmulAllReduce(
        workspace, workspace_size, _opaque->executor, stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matmul_all_reduce::ascend
