#include "gelu_ascend.h"

#include "../../../devices/ascend/aclnn_executor.h"
#include <aclnnop/aclnn_gelu.h>
#include <memory>

namespace op::gelu::ascend {

struct Descriptor::Opaque {
    device::ascend::AclnnExecutor op;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {

    auto status = device::ascend::validateAclnnElementwise(output_desc, input_descs, 1);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    CHECK_DTYPE(output_desc->dtype(),
                INFINI_DTYPE_F16, INFINI_DTYPE_F32,
                INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto opaque = std::make_unique<Opaque>();
    opaque->op.tensors = {
        new aclnnTensorDescriptor(input_descs[0]),
        new aclnnTensorDescriptor(output_desc),
    };

    CHECK_ACL(aclnnGeluGetWorkspaceSize(
        opaque->op.tensors[0]->tensor,
        opaque->op.tensors[1]->tensor,
        &opaque->op.workspace_size,
        &opaque->op.executor));
    aclSetAclOpExecutorRepeatable(opaque->op.executor);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    auto workspace_size = opaque->op.workspace_size;
    *desc_ptr = new Descriptor(
        opaque.release(), workspace_size,
        handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output, std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    _opaque->op.bind({const_cast<void *>(inputs[0]), output});
    CHECK_ACL(aclnnGelu(
        workspace, workspace_size, _opaque->op.executor,
        static_cast<aclrtStream>(stream)));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gelu::ascend
