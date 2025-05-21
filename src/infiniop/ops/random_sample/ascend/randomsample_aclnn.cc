#include "../../../devices/ascend/common_ascend.h"
#include "random_sample_aclnn.h"

namespace op::random_sample::ascend {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t resule_desc,
    infiniopTensorDescriptor_t probs_desc) {

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::random_sample::ascend
