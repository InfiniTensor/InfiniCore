#include "../../../devices/moore/moore_common.h"
#include "softmax_moore.h"

#include <cub/block/block_reduce.cuh>
#include "../../../devices/moore/moore_kernel_common.h"

#include "../../../reduce/cuda/reduce.cuh"

#include "softmax_moore_kernel.h"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_MOORE_KERNEL softmax_kernel(
    Tdata *y, const Tdata *x,
    size_t othersize, size_t dimsize, ptrdiff_t stride) {
    softmaxKernel<BLOCK_SIZE, Tdata, Tcompute>(y, x, othersize, dimsize, stride);
}

namespace op::softmax::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis) {
    auto info = SoftmaxInfo::create(y_desc, x_desc, axis);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t othersize, size_t dimsize, ptrdiff_t stride,
                            musaStream_t stream) {
    dim3 grid(uint32_t(othersize), 1, 1);
    if (dtype == INFINI_DTYPE_F16) {
        softmax_kernel<BLOCK_SIZE, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x,
                                             othersize, dimsize, stride);
    } else if (dtype == INFINI_DTYPE_BF16) {
        softmax_kernel<BLOCK_SIZE, __mt_bfloat16, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((__mt_bfloat16 *)y, (const __mt_bfloat16 *)x,
                                             othersize, dimsize, stride);
    } else if (dtype == INFINI_DTYPE_F32) {
        softmax_kernel<BLOCK_SIZE, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x,
                                             othersize, dimsize, stride);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    musaStream_t stream = (musaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<MOORE_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<MOORE_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softmax::moore