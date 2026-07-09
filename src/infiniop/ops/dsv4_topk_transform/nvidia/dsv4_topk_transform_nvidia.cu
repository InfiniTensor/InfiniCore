#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "dsv4_topk_transform_nvidia.cuh"
namespace {
__global__ void kernel(int32_t *out, const int32_t *seq_lens, size_t batch, size_t index_topk) {
    size_t b = blockIdx.x;
    if (b >= batch) {
        return;
    }
    int valid = seq_lens[b];
    if (valid < 0) {
        valid = 0;
    }
    size_t n_valid = static_cast<size_t>(valid) < index_topk ? static_cast<size_t>(valid) : index_topk;
    for (size_t i = threadIdx.x; i < index_topk; i += blockDim.x) {
        out[b * index_topk + i] = i < n_valid ? static_cast<int32_t>(i) : -1;
    }
}
} // namespace
namespace op::dsv4_topk_transform::nvidia {
infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_tables_desc, int page_size) {
    Info info;
    CHECK_STATUS(createInfo(&info, out_desc, scores_desc, seq_lens_desc, page_tables_desc, page_size));
    *desc_ptr = new Descriptor(info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *, size_t, void *out, const void *, const void *seq_lens, const void *, void *stream) const {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    kernel<<<_info.batch, 128, 0, s>>>(static_cast<int32_t *>(out), static_cast<const int32_t *>(seq_lens), _info.batch, _info.index_topk);
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_topk_transform::nvidia
