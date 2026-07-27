#include "upsample_bilinear_ascend.h"

#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_upsample_bilinear_2d.h>
#include <memory>

namespace op::upsample_bilinear::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t input;
    aclnnTensorDescriptor_t output;
    aclIntArray *output_size;
    aclOpExecutor *executor;

    Opaque(aclnnTensorDescriptor_t input_,
           aclnnTensorDescriptor_t output_,
           aclIntArray *output_size_,
           aclOpExecutor *executor_)
        : input(input_), output(output_), output_size(output_size_),
          executor(executor_) {}

    ~Opaque() {
        delete input;
        delete output;
        if (output_size != nullptr) {
            aclDestroyIntArray(output_size);
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int align_corners) {

    auto result = UpsampleBilinearInfo::create(
        output_desc, input_desc, align_corners);
    CHECK_RESULT(result);
    auto info = result.take();

    CHECK_DTYPE(output_desc->dtype(),
                INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto input = std::make_unique<aclnnTensorDescriptor>(
        input_desc, nullptr, ACL_FORMAT_NCHW);
    auto output = std::make_unique<aclnnTensorDescriptor>(
        output_desc, nullptr, ACL_FORMAT_NCHW);
    std::vector<int64_t> output_size_data = {
        static_cast<int64_t>(info.h_out()),
        static_cast<int64_t>(info.w_out()),
    };
    aclIntArray *output_size = aclCreateIntArray(
        output_size_data.data(), output_size_data.size());
    if (output_size == nullptr) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto acl_status = aclnnUpsampleBilinear2dGetWorkspaceSize(
        input->tensor, output_size, info.align_corners(),
        0.0, 0.0, output->tensor,
        &workspace_size, &executor);
    if (acl_status != ACL_SUCCESS) {
        GetRecentErrMsg();
        aclDestroyIntArray(output_size);
        CHECK_ACL(acl_status);
    }
    aclSetAclOpExecutorRepeatable(executor);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{input.release(), output.release(), output_size, executor},
        std::move(info), workspace_size,
        handle_ascend->device, handle_ascend->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output, const void *input,
    void *stream) const {

    if (workspace_size < workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    AclSetTensorAddr(_opaque->executor, 0, _opaque->input->tensor,
                     const_cast<void *>(input));
    AclSetTensorAddr(_opaque->executor, 1, _opaque->output->tensor, output);
    CHECK_ACL(aclnnUpsampleBilinear2d(
        workspace, workspace_size, _opaque->executor,
        static_cast<aclrtStream>(stream)));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::upsample_bilinear::ascend
