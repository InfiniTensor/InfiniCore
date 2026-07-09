#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_mask_topk_ids_nvidia.cuh"

namespace {

__global__ void kernel(int32_t *topk_ids, const int32_t *num_token_non_padded, size_t batch, size_t topk) {
    int keep = num_token_non_padded[0];
    if (keep < 0) {
        keep = 0;
    }
    if (static_cast<size_t>(keep) > batch) {
        keep = static_cast<int>(batch);
    }
    size_t total = batch * topk;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        size_t row = idx / topk;
        if (row >= static_cast<size_t>(keep)) {
            topk_ids[idx] = -1;
        }
    }
}

} // namespace

namespace op::dsv4_mask_topk_ids::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_ids_desc,
    infiniopTensorDescriptor_t num_token_non_padded_desc) {
    Info info;
    CHECK_STATUS(createInfo(&info, topk_ids_desc, num_token_non_padded_desc));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *topk_ids, const void *num_token_non_padded, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    size_t total = _info.batch * _info.topk;
    int blocks = static_cast<int>((total + 255) / 256);
    if (blocks > 1024) {
        blocks = 1024;
    }
    kernel<<<blocks, 256, 0, s>>>(
        static_cast<int32_t *>(topk_ids),
        static_cast<const int32_t *>(num_token_non_padded),
        _info.batch,
        _info.topk);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_mask_topk_ids::nvidia
