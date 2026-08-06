#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/concat_and_cache_mla.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

template <typename T, typename Index>
INFINIOP_CUDA_KERNEL concatAndCacheMlaKernel(
    const T *kv_c,
    const T *k_pe,
    T *kv_cache,
    const Index *slot_mapping,
    size_t tokens,
    size_t kv_dim,
    size_t rope_dim,
    size_t cache_slots) {
    const size_t head_dim = kv_dim + rope_dim;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * head_dim) {
        return;
    }
    const size_t token = index / head_dim;
    const size_t column = index % head_dim;
    const int64_t slot = static_cast<int64_t>(slot_mapping[token]);
    if (slot < 0 || static_cast<size_t>(slot) >= cache_slots) {
        return;
    }
    kv_cache[static_cast<size_t>(slot) * head_dim + column] = column < kv_dim ? kv_c[token * kv_dim + column]
                                                                              : k_pe[token * rope_dim + column - kv_dim];
}
} // namespace

__INFINI_C infiniStatus_t infiniopConcatAndCacheMla(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t kv_c_desc,
    infiniopTensorDescriptor_t k_pe_desc,
    infiniopTensorDescriptor_t kv_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    const void *kv_c,
    const void *k_pe,
    void *kv_cache,
    const void *slot_mapping,
    void *stream) {
    (void)handle;
    const auto kv_shape = kv_c_desc->shape();
    const auto rope_shape = k_pe_desc->shape();
    const auto cache_shape = kv_cache_desc->shape();
    const auto slots_shape = slot_mapping_desc->shape();
    CHECK_OR_RETURN(kv_shape.size() == 2 && rope_shape.size() == 2
                        && cache_shape.size() >= 3 && slots_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t tokens = kv_shape[0];
    const size_t kv_dim = kv_shape[1];
    const size_t rope_dim = rope_shape[1];
    const size_t head_dim = kv_dim + rope_dim;
    CHECK_OR_RETURN(rope_shape[0] == tokens && slots_shape[0] == tokens
                        && cache_shape.back() == head_dim,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto dtype = kv_c_desc->dtype();
    CHECK_OR_RETURN((dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)
                        && k_pe_desc->dtype() == dtype
                        && kv_cache_desc->dtype() == dtype
                        && (slot_mapping_desc->dtype() == INFINI_DTYPE_I32
                            || slot_mapping_desc->dtype() == INFINI_DTYPE_I64),
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(kv_c_desc->isContiguous() && k_pe_desc->isContiguous()
                        && kv_cache_desc->isContiguous()
                        && slot_mapping_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    const size_t total = tokens * head_dim;
    const size_t blocks = (total + THREADS - 1) / THREADS;
    const size_t cache_slots = kv_cache_desc->numel() / head_dim;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH(T, INDEX)                                                      \
    concatAndCacheMlaKernel<T, INDEX><<<blocks, THREADS, 0, cuda_stream>>>(   \
        static_cast<const T *>(kv_c), static_cast<const T *>(k_pe),           \
        static_cast<T *>(kv_cache), static_cast<const INDEX *>(slot_mapping), \
        tokens, kv_dim, rope_dim, cache_slots)
    if (dtype == INFINI_DTYPE_F16) {
        if (slot_mapping_desc->dtype() == INFINI_DTYPE_I64) {
            LAUNCH(half, int64_t);
        } else {
            LAUNCH(half, int32_t);
        }
    } else if (slot_mapping_desc->dtype() == INFINI_DTYPE_I64) {
        LAUNCH(__nv_bfloat16, int64_t);
    } else {
        LAUNCH(__nv_bfloat16, int32_t);
    }
#undef LAUNCH
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
