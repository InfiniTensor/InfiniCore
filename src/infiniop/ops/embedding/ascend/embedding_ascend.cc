#include "embedding_ascend.h"
#include "../../../devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_embedding.h>

namespace op::embedding::ascend {

struct Descriptor::Opaque {
    aclnnTensorDescriptor_t output;
    aclnnTensorDescriptor_t input;
    aclnnTensorDescriptor_t weight;
    void *workspace;
    uint64_t workspace_size;
    aclOpExecutor *executor;

    ~Opaque() {
        delete output;
        delete input;
        delete weight;
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
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
    infiniopTensorDescriptor_t weight_desc) {

    auto handle = reinterpret_cast<device::ascend::Handle *>(handle_);

    CHECK_API_OR(input_desc->dtype() == INFINI_DTYPE_I32 || input_desc->dtype() == INFINI_DTYPE_I64, true,
                 return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(output_desc->dtype() == weight_desc->dtype(), true, return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(weight_desc->ndim() == 2, true, return INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_API_OR(output_desc->ndim() == input_desc->ndim() + 1, true, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    auto weight_shape = weight_desc->shape();
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        CHECK_API_OR(output_shape[i] == input_shape[i], true, return INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_API_OR(output_shape.back() == weight_shape[1], true, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    size_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }

    auto output = new aclnnTensorDescriptor(output_desc);
    auto input = new aclnnTensorDescriptor(input_desc);
    auto weight = new aclnnTensorDescriptor(weight_desc);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnEmbeddingGetWorkspaceSize(weight->tensor, input->tensor, output->tensor,
                                             &workspace_size, &executor));
    aclSetAclOpExecutorRepeatable(executor);

    void *workspace = nullptr;
    if (workspace_size != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    *desc_ptr = new Descriptor(
        num_indices,
        weight_shape[1],
        weight_shape[0],
        input_desc->dtype(),
        weight_desc->dtype(),
        new Opaque{output, input, weight, workspace, workspace_size, executor},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *input,
    const void *weight,
    void *stream) const {

    auto tweight = _opaque->weight->tensor;
    auto tinput = _opaque->input->tensor;
    auto toutput = _opaque->output->tensor;

    AclSetTensorAddr(_opaque->executor, 0, tweight, const_cast<void *>(weight));
    AclSetTensorAddr(_opaque->executor, 1, tinput, const_cast<void *>(input));
    AclSetTensorAddr(_opaque->executor, 2, toutput, output);

    CHECK_ACL(aclnnEmbedding(_opaque->workspace, _opaque->workspace_size,
                             _opaque->executor, stream));
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::embedding::ascend
