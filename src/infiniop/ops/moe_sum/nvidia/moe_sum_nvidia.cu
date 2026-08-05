#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)

#include "moe_sum_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <algorithm>
#include <memory>
#include <utility>

namespace op::moe_sum::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {
    auto result = MoeSumInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t *__restrict__ output,
    const scalar_t *__restrict__ input,
    size_t hidden_size) {
    const size_t token_idx = blockIdx.x;
    for (size_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        scalar_t x = static_cast<scalar_t>(0.0f);
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            x += input[token_idx * TOPK * hidden_size + k * hidden_size + idx];
        }
        output[token_idx * hidden_size + idx] = x;
    }
}

template <typename scalar_t>
infiniStatus_t launch_moe_sum(
    void *output,
    const void *input,
    size_t num_tokens,
    size_t topk,
    size_t hidden_size,
    cudaStream_t stream) {
    dim3 grid(static_cast<unsigned int>(num_tokens));
    dim3 block(static_cast<unsigned int>(std::min<size_t>(hidden_size, 1024)));

    switch (topk) {
    case 2:
        moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            static_cast<scalar_t *>(output),
            static_cast<const scalar_t *>(input),
            hidden_size);
        return INFINI_STATUS_SUCCESS;
    case 3:
        moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            static_cast<scalar_t *>(output),
            static_cast<const scalar_t *>(input),
            hidden_size);
        return INFINI_STATUS_SUCCESS;
    case 4:
        moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            static_cast<scalar_t *>(output),
            static_cast<const scalar_t *>(input),
            hidden_size);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_PARAM;
    }
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch_moe_sum<half>(output, input, _info.num_tokens, _info.topk, _info.hidden_size, cuda_stream);
    case INFINI_DTYPE_BF16:
        return launch_moe_sum<__nv_bfloat16>(output, input, _info.num_tokens, _info.topk, _info.hidden_size, cuda_stream);
    case INFINI_DTYPE_F32:
        return launch_moe_sum<float>(output, input, _info.num_tokens, _info.topk, _info.hidden_size, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::moe_sum::nvidia

#endif // ENABLE_NVIDIA_API || ENABLE_ILUVATAR_API
