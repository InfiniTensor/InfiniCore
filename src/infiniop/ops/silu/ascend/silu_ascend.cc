#include "silu_ascend.h"

#include "../../../devices/ascend/aclnn_executor.h"
#include "../../swiglu/ascend/swiglu_ascend.h"
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <algorithm>
#include <memory>

namespace op::silu::ascend {

struct Descriptor::Opaque {
    std::unique_ptr<op::swiglu::ascend::SwigluInfo> swiglu_info;
    device::ascend::AclnnExecutor sigmoid;
    device::ascend::AclnnExecutor mul;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs) {
    if (false) {
        auto result = op::swiglu::ascend::SwigluInfo::create(
            output_desc, input_descs[0], input_descs[0]);
        CHECK_RESULT(result);
        auto opaque = std::make_unique<Opaque>();
        opaque->swiglu_info = std::make_unique<op::swiglu::ascend::SwigluInfo>(
            result.take());
        auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
        *desc_ptr = new Descriptor(
            opaque.release(), 0,
            handle_ascend->device, handle_ascend->device_id);
        return INFINI_STATUS_SUCCESS;
    }

    auto status = device::ascend::validateAclnnElementwise(output_desc, input_descs, 1);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    CHECK_DTYPE(output_desc->dtype(),
                INFINI_DTYPE_F16, INFINI_DTYPE_F32,
                INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto opaque = std::make_unique<Opaque>();
    opaque->sigmoid.tensors = {
        new aclnnTensorDescriptor(input_descs[0]),
        new aclnnTensorDescriptor(output_desc),
    };
    opaque->mul.tensors = {
        new aclnnTensorDescriptor(output_desc),
        new aclnnTensorDescriptor(input_descs[0]),
        new aclnnTensorDescriptor(output_desc),
    };

    CHECK_ACL(aclnnSigmoidGetWorkspaceSize(
        opaque->sigmoid.tensors[0]->tensor,
        opaque->sigmoid.tensors[1]->tensor,
        &opaque->sigmoid.workspace_size,
        &opaque->sigmoid.executor));
    aclSetAclOpExecutorRepeatable(opaque->sigmoid.executor);
    CHECK_ACL(aclnnMulGetWorkspaceSize(
        opaque->mul.tensors[0]->tensor,
        opaque->mul.tensors[1]->tensor,
        opaque->mul.tensors[2]->tensor,
        &opaque->mul.workspace_size,
        &opaque->mul.executor));
    aclSetAclOpExecutorRepeatable(opaque->mul.executor);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    auto workspace_size = std::max(
        opaque->sigmoid.workspace_size,
        opaque->mul.workspace_size);
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
    if (false) {
        const auto &info = *_opaque->swiglu_info;
        auto batch = info.ndim == 2 ? 1 : info.shape[0];
        auto seq_len = info.ndim == 2 ? info.shape[0] : info.shape[1];
        auto hidden_size = info.shape[info.ndim - 1];
        auto stride_batch_out = info.ndim == 2 ? 1 : info.c_strides[0];
        auto stride_batch_in = info.ndim == 2 ? 1 : info.a_strides[0];
        auto stride_seq_out = info.ndim == 2 ? info.c_strides[0] : info.c_strides[1];
        auto stride_seq_in = info.ndim == 2 ? info.a_strides[0] : info.a_strides[1];
        return op::swiglu::ascend::swiglu_kernel_launch(
            output,
            const_cast<void *>(inputs[0]),
            const_cast<void *>(inputs[0]),
            info.dtype, batch, seq_len, hidden_size,
            stride_batch_out, stride_batch_in, stride_batch_in,
            stride_seq_out, stride_seq_in, stride_seq_in,
            stream);
    }
    _opaque->sigmoid.bind({const_cast<void *>(inputs[0]), output});
    CHECK_ACL(aclnnSigmoid(
        workspace, workspace_size, _opaque->sigmoid.executor,
        static_cast<aclrtStream>(stream)));

    _opaque->mul.bind({
        output,
        const_cast<void *>(inputs[0]),
        output,
    });
    CHECK_ACL(aclnnMul(
        workspace, workspace_size, _opaque->mul.executor,
        static_cast<aclrtStream>(stream)));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::silu::ascend
