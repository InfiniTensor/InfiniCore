#include "w4a8_moe_shuffle_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::w4a8_moe_shuffle::nvidia {
namespace {

INFINIOP_CUDA_KERNEL w4a8MoeShuffleKernel(
    int8_t *output,
    const int8_t *input,
    size_t output_size,
    size_t packed_k,
    size_t n,
    size_t n_tile) {
    const size_t output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= output_size) {
        return;
    }

    size_t index = output_idx;
    const size_t k_lane = index % 8;
    index /= 8;
    const size_t n_lane = index % 32;
    index /= 32;
    const size_t k_group_lane = index % 4;
    index /= 4;
    const size_t n_group = index % (n_tile / 32);
    index /= n_tile / 32;
    const size_t n_block = index % (n / n_tile);
    const size_t k_block = index / (n / n_tile);

    const size_t source_n = n_block * n_tile + n_group * 32 + n_lane;
    const size_t blocked_k = k_block * 32 + k_group_lane * 8 + k_lane;
    const size_t source_group = blocked_k / 4;
    const size_t source_lane = blocked_k % 4;
    const size_t source_byte = source_group * 4 + source_lane / 2;
    const bool high_nibble = source_lane % 2 == 0;
    const auto first = static_cast<uint8_t>(input[source_n * packed_k + source_byte]);
    const auto second = static_cast<uint8_t>(input[source_n * packed_k + source_byte + 2]);
    const uint8_t first_nibble = high_nibble ? first >> 4 : first & 0x0f;
    const uint8_t second_nibble = high_nibble ? second >> 4 : second & 0x0f;
    output[output_idx] = static_cast<int8_t>((first_nibble << 4) | second_nibble);
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {
    auto result = W4A8MoeShuffleInfo::create(output_desc, input_desc);
    CHECK_RESULT(result);
    auto gpu_handle = reinterpret_cast<device::nvidia::Handle *>(handle);
    *desc_ptr = new Descriptor(
        result.take(), gpu_handle->device, gpu_handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output, const void *input, void *stream_) const {
    constexpr size_t block_size = 256;
    const size_t grid_size = (_info.output_size + block_size - 1) / block_size;
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    w4a8MoeShuffleKernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<int8_t *>(output), static_cast<const int8_t *>(input),
        _info.output_size, _info.packed_k, _info.n, _info.n_tile);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::w4a8_moe_shuffle::nvidia
