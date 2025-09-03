#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "topkrouter_nvidia.cuh"
#include <cub/block/block_reduce.cuh>

namespace op::topkrouter::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t correction_bias_desc) {
    auto result = TopkrouterInfo::create(x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    if (info.x_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <int BLOCK_SIZE = 128>
infiniStatus_t launch_topkrouter(float *d_values_out, int *d_indices_out, void *d_input, float *d_correction_bias, float routed_scaling_factor,
                                 size_t N, size_t width, size_t topk, infiniDtype_t xtype, cudaStream_t stream) {

    const int block_threads = BLOCK_SIZE;
    dim3 blocks(N);
    dim3 threads(block_threads);

    if (xtype == INFINI_DTYPE_F32) {
        topkrouter_kernel<float, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (float *)d_input, d_correction_bias, routed_scaling_factor, N, width, topk);
    } else if (xtype == INFINI_DTYPE_F16) {
        topkrouter_kernel<half, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (half *)d_input, d_correction_bias, routed_scaling_factor, N, width, topk);
    } else if (xtype == INFINI_DTYPE_BF16) {
        topkrouter_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (__nv_bfloat16 *)d_input, d_correction_bias, routed_scaling_factor, N, width, topk);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

}; // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    float *values, int *indices, void *x, float *correction_bias, float routed_scaling_factor, size_t topk, void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    size_t N = _info.N;
    size_t width = _info.width; // 256

    // size_t n_routed_experts = 256;
    // size_t n_group = 8;
    // size_t topk_group = 4;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    if (256 == width) {
        launch_topkrouter<256>(values, indices, x, correction_bias, routed_scaling_factor, N, width, topk, _info.xtype, cuda_stream);
    } else {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topkrouter::nvidia
