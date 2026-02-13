#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../paged_attention_v2/utils/dtype_fp8.cuh"
#include "../cuda/reshape_and_cache_kernels.cuh"
#include "reshape_and_cache_nvidia.cuh"

namespace {
// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                            \
    op::reshape_and_cache::cuda::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE> \
        <<<grid, block, 0, stream>>>(                                              \
            reinterpret_cast<KV_T *>(key),                                         \
            reinterpret_cast<KV_T *>(value),                                       \
            reinterpret_cast<CACHE_T *>(key_cache),                                \
            reinterpret_cast<CACHE_T *>(value_cache),                              \
            reinterpret_cast<const int64_t *>(slot_mapping),                       \
            key_stride, value_stride, num_heads, head_size, block_size, x,         \
            reinterpret_cast<const float *>(k_scale),                              \
            reinterpret_cast<const float *>(v_scale));

}; // namespace

namespace op::reshape_and_cache::nvidia {

using Fp8KVCacheDataType = op::paged_attention_v2::vllm::Fp8KVCacheDataType;

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t key_desc,
    infiniopTensorDescriptor_t value_desc,
    infiniopTensorDescriptor_t key_cache_desc,
    infiniopTensorDescriptor_t value_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    const char *kv_cache_dtype) {

    (void)kv_cache_dtype; // reserved for fp8 / quantized cache
    auto info = ReshapeAndCacheInfo::create(
        key_desc, value_desc, key_cache_desc, value_cache_desc, slot_mapping_desc, kv_cache_dtype);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *key,
    void *value,
    void *key_cache,
    void *value_cache,
    const void *slot_mapping,
    const char *kv_cache_dtype,
    void *k_scale,
    void *v_scale,
    void *stream_) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);

    int num_tokens = _info.num_tokens;
    int num_heads = _info.num_kv_heads;
    int head_size = _info.head_size;
    int block_size = _info.block_size;
    int x = _info.x;
    int key_stride = _info.key_stride;
    int value_stride = _info.value_stride;
    int head_div_x = head_size / x;

    dim3 grid(num_tokens);
    dim3 block(std::min(num_heads * head_div_x, 512));

    auto key_dtype = _info.dtype;
    if (key_dtype == INFINI_DTYPE_F32) {
        CALL_RESHAPE_AND_CACHE(float, float, Fp8KVCacheDataType::kAuto);
    } else if (key_dtype == INFINI_DTYPE_F16) {
        CALL_RESHAPE_AND_CACHE(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);
    } else if (key_dtype == INFINI_DTYPE_BF16) {
        CALL_RESHAPE_AND_CACHE(__nv_bfloat16, __nv_bfloat16, Fp8KVCacheDataType::kAuto);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::reshape_and_cache::nvidia
