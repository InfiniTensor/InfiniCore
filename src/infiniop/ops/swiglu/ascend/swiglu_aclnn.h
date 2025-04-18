#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../../utils.h"
#include "../../../../utils/check.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::swiglu::ascend {
class SwigluInfo {

    SwigluInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<size_t> shape;
    int32_t ndim;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

    static utils::Result<SwigluInfo> create(infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
        if (!c_desc || !a_desc || !b_desc) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (c_desc->hasBroadcastDim()) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (c_desc->ndim() != a_desc->ndim() || c_desc->ndim() != b_desc->ndim() || (c_desc->ndim() != 2 && c_desc->ndim() != 3)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        CHECK_SAME_SHAPE(c_desc->shape(), a_desc->shape(), b_desc->shape());
        int32_t ndim = c_desc->ndim();
        if (c_desc->stride(ndim - 1) != 1 || a_desc->stride(ndim - 1) != 1 || b_desc->stride(ndim - 1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (c_desc->dtype() != a_desc->dtype() || c_desc->dtype() != b_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return utils::Result<SwigluInfo>(SwigluInfo{
            c_desc->dtype(),
            std::move(c_desc->shape()),
            ndim,
            std::move(c_desc->strides()),
            std::move(a_desc->strides()),
            std::move(b_desc->strides()),
        });
    }
};

class Descriptor final : public InfiniopDescriptor {
    SwigluInfo _info;
    size_t _workspace_size;

    Descriptor(SwigluInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id},
                                                                                                    _info(info), _workspace_size(workspace_size) {}

public:
    ~Descriptor();
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t c_desc,
                                 std::vector<infiniopTensorDescriptor_t> input_descs);
    size_t workspaceSize() const { return _workspace_size; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::swiglu::ascend
#endif // __ACLNN_SWIGLU_H__
