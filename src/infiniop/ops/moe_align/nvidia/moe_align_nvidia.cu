/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_align_kernel.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_align_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

namespace op::moe_align::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t expert_ids_desc,
    infiniopTensorDescriptor_t num_tokens_post_padded_desc,
    infiniopTensorDescriptor_t topk_ids_desc,
    size_t num_experts,
    size_t block_size) {
    auto result = MoeAlignInfo::create(
        sorted_token_ids_desc,
        expert_ids_desc,
        num_tokens_post_padded_desc,
        topk_ids_desc,
        num_experts,
        block_size);
    CHECK_RESULT(result);
    auto info = result.take();
    const size_t workspace_size = (2 * info.num_experts + 1) * sizeof(int32_t);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

constexpr int VEC_SIZE = 4;
using Vec = int4;

size_t next_pow2(size_t value) {
    size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ sorted_token_ids,
    int32_t *__restrict__ cumsum_buffer,
    size_t numel) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_map != nullptr) {
            expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
            if (expert_id < 0) {
                continue;
            }
        }
        expert_id += 1;
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
    int original = v;
#pragma unroll
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        int n = __shfl_up_sync(mask, v, offset);
        if ((threadIdx.x & (warpSize - 1)) >= offset) {
            v += n;
        }
    }
    return v - original;
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ sorted_token_ids,
    int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t *__restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
    if (blockIdx.x == 1) {
        if (pad_sorted_token_ids) {
            Vec fill_vec;
            fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
            int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
            Vec *out_ptr = reinterpret_cast<Vec *>(sorted_token_ids);
            for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
                out_ptr[i] = fill_vec;
            }
        }
        return;
    }

    extern __shared__ int32_t smem[];
    int32_t *shared_counts = smem;
    int32_t *prefix = shared_counts + num_experts;
    int32_t *scan_buf = prefix + num_experts + 1;
    __shared__ int32_t s_total_tokens_post_pad;

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    if (tid < num_experts) {
        shared_counts[tid] = 0;
    }

    __syncthreads();

    for (size_t i = tid; i < numel; i += stride) {
        int expert_id = topk_ids[i];
        if (expert_map != nullptr) {
            expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
            if (expert_id < 0) {
                continue;
            }
        }
        expert_id += 1;
        atomicAdd(&shared_counts[expert_id], 1);
    }

    __syncthreads();

    int32_t padded_count = 0;
    if (tid < num_experts) {
        int32_t count = shared_counts[tid];
        padded_count = (count + block_size - 1) / block_size * block_size;
        scan_buf[tid] = padded_count;
    }

    int32_t *warp_sums = scan_buf + scan_size;
    const int warp_id = tid / warpSize;
    const int lane_id = tid & (warpSize - 1);
    const int num_warps_for_scan = (scan_size + warpSize - 1) / warpSize;
    const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
    if (lane_id == warpSize - 1) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (tid < warpSize) {
        int val = (tid < static_cast<size_t>(num_warps_for_scan)) ? warp_sums[tid] : 0;
        int incl = warp_exclusive_scan(val) + val;
        warp_sums[tid] = incl;
    }
    __syncthreads();

    if (tid == 0) {
        prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
        s_total_tokens_post_pad = prefix[num_experts];
        *total_tokens_post_pad = s_total_tokens_post_pad;
    }
    __syncthreads();

    if (tid >= num_experts && tid < static_cast<size_t>(scan_size)) {
        scan_buf[tid] = 0;
    }
    __syncthreads();

    int v = (tid < static_cast<size_t>(scan_size)) ? scan_buf[tid] : 0;
    int pre = warp_exclusive_scan(v);
    if (lane_id == warpSize - 1) {
        warp_sums[warp_id] = pre + v;
    }
    __syncthreads();

    if (warp_id == 0) {
        int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
        warp_sums[lane_id] = warp_exclusive_scan(val);
    }
    __syncthreads();

    int offset = warp_sums[warp_id];
    if (tid < static_cast<size_t>(scan_size)) {
        scan_buf[tid] = pre + offset;
    }
    __syncthreads();

    if (tid < num_experts) {
        prefix[tid] = scan_buf[tid];
    }

    if (tid <= num_experts) {
        cumsum[tid] = prefix[tid];
    }

    const int32_t num_blocks = s_total_tokens_post_pad / block_size;
    for (int32_t i = tid; i < num_blocks; i += stride) {
        int32_t block_start = i * block_size;
        int left = 0;
        int right = num_experts;
        while (left < right) {
            int mid = (left + right) >> 1;
            if (prefix[mid] <= block_start) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        expert_ids[i] = left - 2;
    }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t *__restrict__ topk_ids,
    const int32_t *__restrict__ expert_map,
    int32_t *__restrict__ sorted_token_ids,
    int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
    if (threadIdx.x < fill_threads) {
        if (pad_sorted_token_ids) {
            for (int32_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
                sorted_token_ids[it] = numel;
            }
        }
        __syncthreads();
        __syncthreads();
        __syncthreads();
        return;
    }

    const size_t tid = threadIdx.x - fill_threads;
    const size_t stride = blockDim.x - fill_threads;

    extern __shared__ int32_t shared_mem[];
    int32_t *cumsum = shared_mem;
    int32_t *tokens_cnts = reinterpret_cast<int32_t *>(shared_mem + num_experts + 1);

    for (int i = 0; i < num_experts; ++i) {
        tokens_cnts[(tid + 1) * num_experts + i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_map != nullptr) {
            expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
            if (expert_id < 0) {
                continue;
            }
        }
        expert_id += 1;
        ++tokens_cnts[(tid + 1) * num_experts + expert_id];
    }

    __syncthreads();

    if (tid < static_cast<size_t>(num_experts)) {
        tokens_cnts[tid] = 0;
        for (size_t i = 1; i <= stride; ++i) {
            tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
        }
    }

    __syncthreads();

    if (tid == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i) {
            cumsum[i] = cumsum[i - 1] + ceil_div(tokens_cnts[stride * num_experts + i - 1], block_size) * block_size;
        }
        *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }

    __syncthreads();

    if (tid < static_cast<size_t>(num_experts)) {
        for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
            expert_ids[i / block_size] = tid - 1;
        }
    }

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        if (expert_map != nullptr) {
            expert_id = expert_id >= 0 ? expert_map[expert_id] : -1;
            if (expert_id < 0) {
                continue;
            }
        }
        expert_id += 1;
        int32_t rank_post_pad = tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[tid * num_experts + expert_id];
    }
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *sorted_token_ids,
    void *expert_ids,
    void *num_tokens_post_padded,
    const void *topk_ids,
    const void *expert_map,
    bool pad_sorted_token_ids,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto *cumsum_buffer = static_cast<int32_t *>(workspace);

    constexpr int warp_size = 32;
    int threads = 1024;
    threads = ((threads + warp_size - 1) / warp_size) * warp_size;

    const int32_t num_experts = static_cast<int32_t>(_info.num_experts + 1);
    const int32_t block_size = static_cast<int32_t>(_info.block_size);
    const int32_t max_num_tokens_padded = static_cast<int32_t>(_info.max_num_tokens_padded);
    const bool small_batch_expert_mode = (_info.numel < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
        const int32_t expert_threads = std::max(num_experts, warp_size);
        constexpr int32_t fill_threads = 256;
        const int32_t shared_mem_size = ((expert_threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
        moe_align_block_size_small_batch_expert_kernel<int32_t, fill_threads>
            <<<1, fill_threads + expert_threads, shared_mem_size, cuda_stream>>>(
                static_cast<const int32_t *>(topk_ids),
                static_cast<const int32_t *>(expert_map),
                static_cast<int32_t *>(sorted_token_ids),
                static_cast<int32_t *>(expert_ids),
                static_cast<int32_t *>(num_tokens_post_padded),
                num_experts,
                block_size,
                _info.numel,
                pad_sorted_token_ids,
                max_num_tokens_padded);
    } else {
        const int32_t scan_size = static_cast<int32_t>(next_pow2(num_experts));
        const size_t shared_mem_size = (num_experts + (num_experts + 1) + scan_size + warp_size) * sizeof(int32_t);
        moe_align_block_size_kernel<int32_t><<<2, threads, shared_mem_size, cuda_stream>>>(
            static_cast<const int32_t *>(topk_ids),
            static_cast<const int32_t *>(expert_map),
            static_cast<int32_t *>(sorted_token_ids),
            static_cast<int32_t *>(expert_ids),
            static_cast<int32_t *>(num_tokens_post_padded),
            num_experts,
            block_size,
            _info.numel,
            cumsum_buffer,
            pad_sorted_token_ids,
            scan_size,
            max_num_tokens_padded);

        const int block_threads = std::min(256, threads);
        const int num_blocks = static_cast<int>((_info.numel + block_threads - 1) / block_threads);
        const int max_blocks = 65535;
        const int actual_blocks = std::min(num_blocks, max_blocks);

        count_and_sort_expert_tokens_kernel<int32_t><<<actual_blocks, block_threads, 0, cuda_stream>>>(
            static_cast<const int32_t *>(topk_ids),
            static_cast<const int32_t *>(expert_map),
            static_cast<int32_t *>(sorted_token_ids),
            cumsum_buffer,
            _info.numel);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::moe_align::nvidia
