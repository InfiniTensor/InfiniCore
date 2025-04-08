#include "../../../devices/musa/common_musa.h"
#include "rms_norm_kernel.h"
#include "rms_norm_musa.h"

namespace op::rms_norm::musa {

struct Descriptor::Opaque {
    std::shared_ptr<device::musa::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    // only support contiguous last dimension
    if (info.x_strides[1] != 1 || info.y_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::musa::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// launch kernel with different data types
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t dim,
    void *y, infiniDtype_t atype, ptrdiff_t stride_y,
    const void *x, ptrdiff_t stride_x,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    musaStream_t musa_stream) {

#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute)                                                     \
    rmsnormBlock<BLOCK_SIZE, Tdata, Tweight, Tcompute><<<batch_size, BLOCK_SIZE, 0, musa_stream>>>( \
        reinterpret_cast<Tdata *>(y),                                                               \
        stride_y,                                                                                   \
        reinterpret_cast<const Tdata *>(x),                                                         \
        stride_x,                                                                                   \
        reinterpret_cast<const Tweight *>(w),                                                       \
        dim,                                                                                        \
        epsilon)

    if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float);
    } else if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_x = _info.x_strides[0];
    auto stride_y = _info.y_strides[0];
    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // launch kernel with different block sizes
    if (_opaque->internal->maxThreadsPerBlock() == MUSA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<MUSA_BLOCK_SIZE_1024>(batch_size, dim, y, _info.atype, stride_y, x, stride_x, w, _info.wtype, _info.epsilon, musa_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm::musa
