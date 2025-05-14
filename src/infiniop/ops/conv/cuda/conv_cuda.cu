#include "../../../devices/cuda/cuda_handle.cuh"
#include "conv_cuda.cuh"

namespace op::conv::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
    auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
                                   pads, strides, dilations, n);

    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) const {
    const float alpha = 1.0f, beta = 0.0f;
    if (bias != nullptr) {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
                    handle,
                    &alpha,
                    _info.handler->x_desc,
                    x,
                    _info.handler->w_desc,
                    w,
                    _info.handler->conv_desc,
                    _info.handler->algo,
                    workspace, _info.handler->workspace_size,
                    &beta,
                    _info.handler->y_desc,
                    y,
                    _info.handler->b_desc,
                    bias,
                    _info.handler->act_desc,
                    _info.handler->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    } else {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionForward(
                    handle,
                    &alpha,
                    _info.handler->x_desc,
                    x,
                    _info.handler->w_desc,
                    w,
                    _info.handler->conv_desc,
                    _info.handler->algo,
                    workspace, _info.handler->workspace_size,
                    &beta,
                    _info.handler->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::conv::cuda
