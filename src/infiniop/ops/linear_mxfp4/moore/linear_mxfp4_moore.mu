#include "linear_mxfp4_moore.h"

#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"

#define INFINIOP_MXFP4_KERNEL INFINIOP_MOORE_KERNEL
#include "../../mxfp4_common/cuda/linear_mxfp4_kernel.cuh"
#undef INFINIOP_MXFP4_KERNEL

namespace op::linear_mxfp4::moore {
namespace {

template <typename T>
void launch(T *output,
            const T *input,
            const uint8_t *packed_weight,
            const uint8_t *weight_scale,
            const T *bias,
            const LinearMxfp4Info &info,
            musaStream_t stream) {
    op::mxfp4_common::cuda::launchLinearMxfp4(
        output, input, packed_weight, weight_scale, bias,
        info.M, info.N, info.K, info.alpha, stream);
}

} // namespace

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t packed_weight_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t bias_desc,
    float alpha) {
    auto info = LinearMxfp4Info::create(
        output_desc, input_desc, packed_weight_desc, weight_scale_desc, bias_desc, alpha);
    CHECK_RESULT(info);
    auto moore_handle = reinterpret_cast<device::moore::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{moore_handle->internal()}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t,
    void *output,
    const void *input,
    const void *packed_weight,
    const void *weight_scale,
    const void *bias,
    void *stream) const {
    auto moore_stream = reinterpret_cast<musaStream_t>(stream);
    const auto *packed_ptr = reinterpret_cast<const uint8_t *>(packed_weight);
    const auto *scale_ptr = reinterpret_cast<const uint8_t *>(weight_scale);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        launch(reinterpret_cast<half *>(output),
               reinterpret_cast<const half *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const half *>(bias),
               _info, moore_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        launch(reinterpret_cast<__nv_bfloat16 *>(output),
               reinterpret_cast<const __nv_bfloat16 *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const __nv_bfloat16 *>(bias),
               _info, moore_stream);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        launch(reinterpret_cast<float *>(output),
               reinterpret_cast<const float *>(input),
               packed_ptr, scale_ptr, reinterpret_cast<const float *>(bias),
               _info, moore_stream);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::linear_mxfp4::moore
