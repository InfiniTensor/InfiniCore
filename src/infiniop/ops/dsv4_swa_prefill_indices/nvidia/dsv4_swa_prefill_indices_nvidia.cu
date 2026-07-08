
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_swa_prefill_indices_nvidia.cuh"
namespace {
__global__ void kernel(int32_t *indices, int seq_len, int window_size, int batch) {
    int b = blockIdx.x;
    if (b >= batch) {
        return;
    }
    int tid = threadIdx.x;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        int start = max(0, i - window_size + 1);
        indices[b * seq_len + i] = start;
    }
}
} // namespace
namespace op::dsv4_swa_prefill_indices::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t indices_desc, int window_size) {
    Info info;
    CHECK_STATUS(createInfo(&info, indices_desc, window_size));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *indices, void *stream) const {
    kernel<<<_info.batch, 128, 0, (cudaStream_t)stream>>>(static_cast<int32_t *>(indices), (int)_info.seq_len, _info.window_size, (int)_info.batch);
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_swa_prefill_indices::nvidia
