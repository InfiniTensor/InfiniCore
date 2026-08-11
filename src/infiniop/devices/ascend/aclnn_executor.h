#ifndef __INFINIOP_ACLNN_EXECUTOR_H__
#define __INFINIOP_ACLNN_EXECUTOR_H__

#include "common_ascend.h"

#include <initializer_list>
#include <vector>

namespace device::ascend {

struct AclnnExecutor {
    std::vector<aclnnTensorDescriptor_t> tensors;
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    AclnnExecutor() = default;
    AclnnExecutor(const AclnnExecutor &) = delete;
    AclnnExecutor &operator=(const AclnnExecutor &) = delete;

    ~AclnnExecutor() {
        for (auto tensor : tensors) {
            delete tensor;
        }
        if (executor != nullptr) {
            aclDestroyAclOpExecutor(executor);
        }
    }

    void bind(std::initializer_list<void *> addresses) const {
        size_t index = 0;
        for (auto address : addresses) {
            AclSetTensorAddr(executor, index, tensors[index]->tensor, address);
            ++index;
        }
    }
};

inline infiniStatus_t validateAclnnElementwise(
    infiniopTensorDescriptor_t output_desc,
    const std::vector<infiniopTensorDescriptor_t> &input_descs,
    size_t expected_inputs) {

    if (output_desc == nullptr || input_descs.size() != expected_inputs) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (output_desc->hasBroadcastDim()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    for (auto input_desc : input_descs) {
        if (input_desc == nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (input_desc->dtype() != output_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (input_desc->ndim() != output_desc->ndim()
            || input_desc->shape() != output_desc->shape()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::ascend

#endif // __INFINIOP_ACLNN_EXECUTOR_H__
