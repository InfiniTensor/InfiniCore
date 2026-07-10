#include "../../../devices/nvidia/nvidia_common.cuh"
#include "fused_moe_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>

namespace op::fused_moe::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t token_selected_experts_desc,
    infiniopTensorDescriptor_t token_final_scales_desc,
    infiniopTensorDescriptor_t w1_desc,
    infiniopTensorDescriptor_t w2_desc,
    infiniopTensorDescriptor_t b1_desc,
    infiniopTensorDescriptor_t b2_desc,
    infiniopFusedMoeActivation_t activation) {
    auto info = FusedMoeInfo::create(out_desc, input_desc, token_selected_experts_desc,
                                     token_final_scales_desc, w1_desc, w2_desc,
                                     b1_desc, b2_desc, activation);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *token_selected_experts,
    const void *token_final_scales,
    const void *w1,
    const void *w2,
    const void *b1,
    const void *b2,
    void *stream_) const {
    if (out == nullptr || input == nullptr || token_selected_experts == nullptr ||
        token_final_scales == nullptr || w1 == nullptr || w2 == nullptr ||
        (_info.has_b1 && b1 == nullptr) || (_info.has_b2 && b2 == nullptr)) {
        return INFINI_STATUS_NULL_POINTER;
    }

    cudaStream_t stream = (cudaStream_t)stream_;
    size_t total = _info.N * _info.hidden_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    if (_info.dtype == INFINI_DTYPE_F16) {
        fusedMoeKernel<half><<<blocks, threads, 0, stream>>>(
            static_cast<half *>(out), static_cast<const half *>(input),
            static_cast<const int32_t *>(token_selected_experts),
            static_cast<const float *>(token_final_scales),
            static_cast<const half *>(w1), static_cast<const half *>(w2),
            static_cast<const half *>(b1), static_cast<const half *>(b2),
            _info.N, _info.hidden_size, _info.inter_size, _info.num_experts,
            _info.topk, _info.w1_cols, static_cast<int>(_info.activation));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        fusedMoeKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(out), static_cast<const __nv_bfloat16 *>(input),
            static_cast<const int32_t *>(token_selected_experts),
            static_cast<const float *>(token_final_scales),
            static_cast<const __nv_bfloat16 *>(w1), static_cast<const __nv_bfloat16 *>(w2),
            static_cast<const __nv_bfloat16 *>(b1), static_cast<const __nv_bfloat16 *>(b2),
            _info.N, _info.hidden_size, _info.inter_size, _info.num_experts,
            _info.topk, _info.w1_cols, static_cast<int>(_info.activation));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        fusedMoeKernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<float *>(out), static_cast<const float *>(input),
            static_cast<const int32_t *>(token_selected_experts),
            static_cast<const float *>(token_final_scales),
            static_cast<const float *>(w1), static_cast<const float *>(w2),
            static_cast<const float *>(b1), static_cast<const float *>(b2),
            _info.N, _info.hidden_size, _info.inter_size, _info.num_experts,
            _info.topk, _info.w1_cols, static_cast<int>(_info.activation));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::fused_moe::nvidia
