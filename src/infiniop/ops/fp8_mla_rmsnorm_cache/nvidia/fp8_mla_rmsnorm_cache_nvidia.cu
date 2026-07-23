#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "fp8_mla_rmsnorm_cache_nvidia.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr size_t LATENT_DIM = 512;
constexpr size_t GROUP_SIZE = 128;
constexpr size_t NUM_GROUPS = LATENT_DIM / GROUP_SIZE;
constexpr size_t ROPE_DIM = 64;
constexpr size_t VENDOR_CACHE_STRIDE = LATENT_DIM + ROPE_DIM;
constexpr size_t CACHE_STRIDE = LATENT_DIM + NUM_GROUPS * sizeof(float)
                              + ROPE_DIM * sizeof(cuda_bfloat16);

INFINIOP_CUDA_KERNEL fp8MlaRmsnormCacheKernel(
    uint8_t *__restrict__ cache,
    cuda_bfloat16 *__restrict__ vendor_cache,
    const cuda_bfloat16 *__restrict__ compressed_kv,
    const cuda_bfloat16 *__restrict__ norm_weight,
    const cuda_bfloat16 *__restrict__ rope,
    const int64_t *__restrict__ slot_mapping,
    size_t num_cache_tokens,
    float eps) {
    const size_t token = blockIdx.x;
    const size_t column = threadIdx.x;
    const int64_t slot_raw = slot_mapping[token];
    if (slot_raw < 0 || static_cast<size_t>(slot_raw) >= num_cache_tokens) {
        return;
    }

    extern __shared__ float scratch[];
    const float input = __bfloat162float(
        compressed_kv[token * LATENT_DIM + column]);
    scratch[column] = input * input;
    __syncthreads();
    for (size_t stride = LATENT_DIM / 2; stride > 0; stride >>= 1) {
        if (column < stride) {
            scratch[column] += scratch[column + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(LATENT_DIM) + eps);
    const cuda_bfloat16 normalized = __float2bfloat16_rn(
        input * inv_rms * __bfloat162float(norm_weight[column]));
    const float value = __bfloat162float(normalized);
    scratch[column] = fabsf(value);
    __syncthreads();

    const size_t group = column / GROUP_SIZE;
    const size_t lane = column % GROUP_SIZE;
    const size_t group_base = group * GROUP_SIZE;
    for (size_t stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            scratch[group_base + lane] = fmaxf(
                scratch[group_base + lane],
                scratch[group_base + lane + stride]);
        }
        __syncthreads();
    }
    const float abs_max = scratch[group_base];
    const float scale = abs_max > 0.0f ? abs_max / 448.0f : 0.0f;

    const size_t slot = static_cast<size_t>(slot_raw);
    uint8_t *cache_entry = cache + slot * CACHE_STRIDE;
    const float quantized = scale > 0.0f
                              ? fminf(448.0f, fmaxf(-448.0f, value / scale))
                              : 0.0f;
    const cuda_fp8_e4m3 fp8_value(quantized);
    reinterpret_cast<cuda_fp8_e4m3 *>(cache_entry)[column] = fp8_value;
    if (lane == 0) {
        reinterpret_cast<float *>(cache_entry + LATENT_DIM)[group] = scale;
    }
    const cuda_bfloat16 rope_value = column < ROPE_DIM
                                       ? rope[token * ROPE_DIM + column]
                                       : __float2bfloat16_rn(0.0f);
    if (column < ROPE_DIM) {
        reinterpret_cast<cuda_bfloat16 *>(
            cache_entry + LATENT_DIM + NUM_GROUPS * sizeof(float))[column]
            = rope_value;
    }
    if (vendor_cache != nullptr) {
        auto *vendor_entry = vendor_cache + slot * VENDOR_CACHE_STRIDE;
        vendor_entry[column] = __float2bfloat16_rn(
            static_cast<float>(fp8_value) * scale);
        if (column < ROPE_DIM) {
            vendor_entry[LATENT_DIM + column] = rope_value;
        }
    }
}
} // namespace

namespace op::fp8_mla_rmsnorm_cache::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t cache_desc,
    infiniopTensorDescriptor_t vendor_cache_desc,
    infiniopTensorDescriptor_t compressed_kv_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t rope_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    double eps) {
    const auto cache_shape = cache_desc->shape();
    const auto kv_shape = compressed_kv_desc->shape();
    const auto weight_shape = norm_weight_desc->shape();
    const auto rope_shape = rope_desc->shape();
    const auto slots_shape = slot_mapping_desc->shape();
    CHECK_OR_RETURN(cache_shape.size() == 3 && kv_shape.size() == 2
                        && weight_shape.size() == 1 && rope_shape.size() == 2
                        && slots_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(kv_shape[1] == LATENT_DIM
                        && weight_shape[0] == LATENT_DIM
                        && rope_shape[0] == kv_shape[0]
                        && rope_shape[1] == ROPE_DIM
                        && slots_shape[0] == kv_shape[0]
                        && cache_shape[1] > 0
                        && cache_shape[2] == CACHE_STRIDE,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(cache_desc->isContiguous()
                        && compressed_kv_desc->isContiguous()
                        && norm_weight_desc->isContiguous()
                        && rope_desc->isContiguous()
                        && slot_mapping_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(cache_desc->dtype() == INFINI_DTYPE_U8
                        && compressed_kv_desc->dtype() == INFINI_DTYPE_BF16
                        && norm_weight_desc->dtype() == INFINI_DTYPE_BF16
                        && rope_desc->dtype() == INFINI_DTYPE_BF16
                        && slot_mapping_desc->dtype() == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(eps > 0.0, INFINI_STATUS_BAD_PARAM);
    if (vendor_cache_desc != nullptr) {
        const auto vendor_shape = vendor_cache_desc->shape();
        CHECK_OR_RETURN(
            vendor_shape.size() == 3
                && vendor_shape[0] == cache_shape[0]
                && vendor_shape[1] == cache_shape[1]
                && vendor_shape[2] == VENDOR_CACHE_STRIDE,
            INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(
            vendor_cache_desc->isContiguous(),
            INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(
            vendor_cache_desc->dtype() == INFINI_DTYPE_BF16,
            INFINI_STATUS_BAD_TENSOR_DTYPE);
    }
    *desc_ptr = new Descriptor(
        kv_shape[0], cache_shape[0], cache_shape[1],
        vendor_cache_desc != nullptr, eps,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *cache,
    void *vendor_cache,
    const void *compressed_kv,
    const void *norm_weight,
    const void *rope,
    const void *slot_mapping,
    void *stream) const {
    fp8MlaRmsnormCacheKernel<<<
        _num_tokens, LATENT_DIM, LATENT_DIM * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t *>(cache),
        _write_vendor_cache
            ? reinterpret_cast<cuda_bfloat16 *>(vendor_cache)
            : nullptr,
        reinterpret_cast<const cuda_bfloat16 *>(compressed_kv),
        reinterpret_cast<const cuda_bfloat16 *>(norm_weight),
        reinterpret_cast<const cuda_bfloat16 *>(rope),
        reinterpret_cast<const int64_t *>(slot_mapping),
        _num_cache_blocks * _block_size,
        _eps);
    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::fp8_mla_rmsnorm_cache::nvidia
