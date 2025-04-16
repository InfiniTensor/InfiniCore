#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../../utils/check.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::swiglu::ascend {
struct SwigluInfo {
    infiniDtype_t dtype;
    std::vector<size_t> shape;
    int32_t ndim;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;
};

inline infiniStatus_t createSwigluInfo(SwigluInfo &info,
                                       infiniopTensorDescriptor_t c_desc,
                                       infiniopTensorDescriptor_t a_desc,
                                       infiniopTensorDescriptor_t b_desc) {
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
    info.ndim = ndim;
    info.dtype = c_desc->dtype();
    info.shape = std::move(c_desc->shape());
    info.c_strides = std::move(c_desc->strides());
    info.a_strides = std::move(a_desc->strides());
    info.b_strides = std::move(b_desc->strides());

    return INFINI_STATUS_SUCCESS;
}

class Descriptor final : public InfiniopDescriptor {
    SwigluInfo _info;

    Descriptor(SwigluInfo info, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id},
                                                                             _info(info) {}

public:
    ~Descriptor();
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t c_desc,
                                 infiniopTensorDescriptor_t a_desc,
                                 infiniopTensorDescriptor_t b_desc);

    infiniStatus_t calculate(
        void *c,
        const void *a,
        const void *b,
        void *stream) const;
};

} // namespace op::swiglu::ascend
#endif // __ACLNN_SWIGLU_H__
