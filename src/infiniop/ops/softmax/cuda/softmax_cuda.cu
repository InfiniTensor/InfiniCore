#include "../../../devices/cuda/cuda_common.cuh"
#include "softmax_cuda.cuh"
#include "softmax_kernel.cuh"

namespace op::softmax::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int axis) {
    auto dtype = y->dtype();
    auto handle = reinterpret_cast<device::cuda::Handle *>(handle_);
    auto result = SoftmaxInfo::create(y, x, axis);
    CHECK_RESULT(result);
    CHECK_SAME_SHAPE(y->shape(), x->shape());
    CHECK_DTYPE(y->dtype(), x->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_F32);
    *desc_ptr = new Descriptor(
        dtype,
        result.take(),
        0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream_) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return softmax_dispatch<half>(_info, y, x, stream_);
    case INFINI_DTYPE_F32:
        return softmax_dispatch<float>(_info, y, x, stream_);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::softmax::cuda