// recurrent_gated_delta_rule_nvidia.cu

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "recurrent_gated_delta_rule_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"
#include <cuda_runtime.h>



// Kernel Launcher Wrapper
template <typename Tdata, typename Tcompute, size_t Dk, size_t Dv, size_t NUM_THREADS>
INFINIOP_CUDA_KERNEL recurrentGatedDeltaRule(
    Tdata* out, Tdata* final_state,
    const Tdata* q, const Tdata* k, const Tdata* v,
    const Tdata* g, const Tdata* beta, const Tdata* initial_state,
    bool use_qk_l2norm
) {
    recurrentGatedDeltaRuleKernel<Tdata, Tcompute, Dk, Dv, NUM_THREADS>(
        out, final_state, q, k, v, g, beta, initial_state, use_qk_l2norm
    );
}

namespace op {
namespace recurrent_gated_delta_rule {
namespace nvidia {


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
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    bool use_qk_l2norm
) {
    auto info = RecurrentGatedDeltaRuleInfo::create(
        out_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, initial_state_desc, use_qk_l2norm);
    CHECK_RESULT(info);

    // Calculate workspace size if needed, here it's 0
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

template <size_t Dk, size_t Dv, size_t NUM_THREADS>
infiniStatus_t launchKernel(
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    bool use_qk_l2norm,
    infiniDtype_t dtype,
    size_t B, size_t H,
    cudaStream_t stream
) {
    dim3 grid(uint32_t(B), uint32_t(H), 1);
    dim3 block(NUM_THREADS);
    // Shared memory for local Q, K, and one reduction value
    size_t shared_mem_size = (Dk + Dk + NUM_THREADS) * sizeof(float);

    if (dtype == INFINI_DTYPE_F16) {
        recurrentGatedDeltaRule<half, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (half*)out, (half*)final_state,
                (const half*)q, (const half*)k, (const half*)v,
                (const half*)g, (const half*)beta, (const half*)initial_state,
                use_qk_l2norm
            );
    } else if (dtype == INFINI_DTYPE_BF16) {
        recurrentGatedDeltaRule<__nv_bfloat16, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (__nv_bfloat16*)out, (__nv_bfloat16*)final_state,
                (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
                (const __nv_bfloat16*)g, (const __nv_bfloat16*)beta, (const __nv_bfloat16*)initial_state,
                use_qk_l2norm
            );
    } else if (dtype == INFINI_DTYPE_F32) {
        recurrentGatedDeltaRule<float, float, Dk, Dv, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (float*)out, (float*)final_state,
                (const float*)q, (const float*)k, (const float*)v,
                (const float*)g, (const float*)beta, (const float*)initial_state,
                use_qk_l2norm
            );
    } else {
        return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return infiniStatus_t::INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, void* final_state,
    const void *q, const void *k, const void *v,
    const void *g, const void* beta, const void* initial_state,
    void *stream_
) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    // Specialize for common shapes and thread counts
    if (_info.Dk == 128 && _info.Dv == 128) {
         if (_opaque->internal->maxThreadsPerBlock() >= 128) {
            return launchKernel<128, 128, 128>(
                out, final_state, q, k, v, g, beta, initial_state, _info.use_qk_l2norm,
                _info.dtype, _info.B, _info.H, stream);
        }
    } else if (_info.Dk == 64 && _info.Dv == 64) {
        if (_opaque->internal->maxThreadsPerBlock() >= 64) {
            return launchKernel<64, 64, 64>(
                out, final_state, q, k, v, g, beta, initial_state, _info.use_qk_l2norm,
                _info.dtype, _info.B, _info.H, stream);
        }
    }
    
    // Fallback or error for unsupported shapes
    // You can add more specializations for other shapes here.
    return infiniStatus_t::INFINI_STATUS_BAD_TENSOR_SHAPE;
}

} // namespace nvidia
} // namespace recurrent_gated_delta_rule
} // namespace op