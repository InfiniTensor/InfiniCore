#include "../../../devices/metax/metax_kernel_common.h"
#include "infinicore.h"
#include <hccub/device/device_radix_sort.cuh>
#include <hccub/device/device_reduce.cuh>
#include <hccub/device/device_scan.cuh>
#include <hcr/hc_runtime_api.h>
#include <vector>
#include <algorithm>
#include <cstdio>

namespace op::random_sample::metax {

// ↓↓↓ 重新封装 cub api，减少模板参数，方便调用

template <class T>
static hcError_t argMax_(
    cub::KeyValuePair<int, T> *kv_pair,
    const T *logits,
    int n,
    void *workspace_ptr,
    size_t &workspace_len,
    hcStream_t stream) {
    return cub::DeviceReduce::ArgMax(
        workspace_ptr, workspace_len,
        logits, kv_pair, n,
        stream);
}

template <class Tval, class Tidx>
static hcError_t radixSort(
    void *workspace_ptr, size_t &workspace_len,
    const Tval *key_in, Tval *key_out,
    const Tidx *val_in, Tidx *val_out,
    int n,
    hcStream_t stream) {
    return cub::DeviceRadixSort::SortPairsDescending(
        workspace_ptr, workspace_len,
        key_in, key_out,
        val_in, val_out,
        n,
        0, sizeof(Tval) * 8,
        stream);
}

template <class T>
static hcError_t inclusiveSum(
    void *workspace_ptr, size_t &workspace_len,
    T *data, int n,
    hcStream_t stream) {
    return cub::DeviceScan::InclusiveSum(
        workspace_ptr, workspace_len,
        data, data, n,
        stream);
}

// ↑↑↑ 重新封装 cub api，减少模板参数，方便调用
// ↓↓↓ 计算 workspace

// 地址对齐到 256
static constexpr size_t align256(size_t size) {
    return (size + 255) & (~255);
}

template <class Tidx, class Tval>
utils::Result<size_t> calculateWorkspace(size_t n_) {
    const auto n = static_cast<int>(n_);

    size_t argmax;
    CHECK_METAX(argMax_<Tval>(
        nullptr, nullptr, n,
        nullptr, argmax,
        nullptr));
    // 前 256 字节用于 kv pair
    argmax += 256;

    // indices
    size_t size_random = align256(sizeof(Tidx) * n);
    // sorted
    size_random += align256(sizeof(Tval) * n);
    // indices_out
    size_random += align256(sizeof(Tidx) * n);
    // sorted_out (needed when repetition_penalty != 1.0)
    size_random += align256(sizeof(Tval) * n);
    // cub device api
    size_t size_radix_sort;
    CHECK_METAX((radixSort<Tval, Tidx>(
        nullptr, size_radix_sort,
        nullptr, nullptr,
        nullptr, nullptr,
        n,
        nullptr)));

    size_t size_inclusive_sum;
    CHECK_METAX(inclusiveSum<Tval>(
        nullptr, size_inclusive_sum,
        nullptr, n,
        nullptr));
    size_random += cub::Max()(size_radix_sort, size_inclusive_sum);

    return utils::Result<size_t>(cub::Max()(argmax, size_random));
}

// ↑↑↑ 计算 workspace
// ↓↓↓ 通过特化将 fp16_t 转换为 half

template <class Tval>
struct CudaTval {
    using Type = Tval;
};

template <>
struct CudaTval<fp16_t> {
    using Type = half;
};

template <>
struct CudaTval<bf16_t> {
    using Type = __hpcc_bfloat16;
};

// ↑↑↑ 通过特化将 fp16_t 转换为 half
// ↓↓↓ 用于采样过程的小型 kernel

// maca toolkit 11.x 带的 cub::DeviceReduce::ArgMax 只接受 cub::KeyValuePair<int, Tval> 输出。
// 这个 kernel 用于取出序号
template <class Tidx, class Tval>
static __global__ void castIdx(Tidx *result, const cub::KeyValuePair<int, Tval> *kv_pair) {
    *result = kv_pair->key;
}

// 填充排序要求的序号数组
template <class Tidx>
static __global__ void fillIndices(Tidx *indices, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = i;
    }
}

// random sample 使用的 softmax 可以简化为一个基本的线性映射
// 由于已经排序，最大值就是第一个数字
// 第一个数字需要被多个 block 读取，不能写
template <class T>
static __global__ void partialSoftmaxKernel(
    T *__restrict__ data, int n,
    float temperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 < i && i < n) {
        float max = __ldg(data);
        data[i] = (T)expf(((float)data[i] - max) / temperature);
    }
}

// 将第一个数字写成 1，即 exp(0)
template <class T>
static __global__ void setSoftmaxMaxKernel(
    T *__restrict__ data) {
    *data = 1;
}

// 直接 for 循环遍历采样
// 这个 kernel 仅用于避免将数据拷贝到 cpu
template <class Tval, class Tidx>
static __global__ void randomSampleKernel(
    Tidx *__restrict__ result,
    const Tval *__restrict__ sorted,
    const Tidx *__restrict__ indices_out,
    size_t n,
    float random, float topp, size_t topk) {
    // topk should already be validated to be > 0 and <= n by the caller
    // (disabled topk 0/-1 is converted to n before calling this kernel)
    topk = cub::Min()(topk, n);
    auto p = (Tval)(random * cub::Min()(topp * (float)sorted[n - 1], (float)sorted[topk - 1]));
    for (size_t i = 0;; ++i) {
        if ((sorted[i]) >= p) {
            *result = indices_out[i];
            return;
        }
    }
}

// ↑↑↑ 用于采样过程的小型 kernel

struct Algo {
    int block_size;

    template <class Tidx, class Tval_>
    infiniStatus_t argmax(
        void *workspace, size_t workspace_size,
        void *result, const void *probs, size_t n,
        void *stream_) const {

        using Tval = typename CudaTval<Tval_>::Type;

        auto stream = (hcStream_t)stream_;
        auto logits = (Tval *)probs;
        auto kv_pair = (cub::KeyValuePair<int, Tval> *)workspace;
        workspace = (void *)((char *)workspace + 256);
        workspace_size -= 256;

        argMax_(
            kv_pair,
            logits,
            n,
            workspace,
            workspace_size, stream);
        castIdx<<<1, 1, 0, stream>>>((Tidx *)result, kv_pair);

        return INFINI_STATUS_SUCCESS;
    }

    template <class Tidx, class Tval_>
    infiniStatus_t random(
        void *workspace_, size_t workspace_size,
        void *result_, const void *probs, size_t n,
        float random_val, float topp, int topk, float temperature, float repetition_penalty,
        const uint32_t *previous_tokens, size_t previous_tokens_len,
        void *stream_) const {

        using Tval = typename CudaTval<Tval_>::Type;

        auto stream = (hcStream_t)stream_;
        auto logits = (Tval *)probs;
        auto result = (Tidx *)result_;

        auto workspace = reinterpret_cast<size_t>(workspace_);
        auto workspace_end = workspace + workspace_size;

        auto indices = reinterpret_cast<Tidx *>(workspace);
        workspace += align256(sizeof(Tidx) * n);

        auto sorted = reinterpret_cast<Tval *>(workspace);
        workspace += align256(sizeof(Tval) * n);

        auto indices_out = reinterpret_cast<Tidx *>(workspace);
        workspace += align256(sizeof(Tidx) * n);

        auto block = cub::Min()((size_t)block_size, n);
        auto grid = (n + block - 1) / block;

        // Apply repetition penalty if needed (penalize all tokens before sorting)
        if (repetition_penalty != 1.0f) {
            // Allocate temporary output buffer for radixSort from workspace (before CUB workspace)
            auto sorted_out = reinterpret_cast<Tval *>(workspace);
            workspace += align256(sizeof(Tval) * n);

            // Now set CUB workspace pointer and size
            workspace_ = reinterpret_cast<void *>(workspace);
            workspace_size = workspace_end - workspace;

            // Copy logits to host memory
            std::vector<Tval> host_logits(n);
            CHECK_METAX(hcMemcpyAsync(host_logits.data(), logits, n * sizeof(Tval), hcMemcpyDeviceToHost, stream));
            CHECK_METAX(hcStreamSynchronize(stream));

            // Apply penalty: if previous_tokens are provided, only penalize those tokens
            // Otherwise, penalize all tokens (full-history penalty for backward compatibility)
            if (previous_tokens != nullptr && previous_tokens_len > 0) {
                // Proper repetition penalty: only penalize previously generated tokens
                for (size_t i = 0; i < previous_tokens_len; i++) {
                    uint32_t token_id = previous_tokens[i];
                    if (token_id < n) {
                        float val = static_cast<float>(host_logits[token_id]);
                        if (val > 0) {
                            host_logits[token_id] = static_cast<Tval>(val / repetition_penalty);
                        } else {
                            host_logits[token_id] = static_cast<Tval>(val * repetition_penalty);
                        }
                    }
                }
            } else {
                // Full-history penalty: penalize all tokens (backward compatibility)
                for (size_t i = 0; i < n; i++) {
                    float val = static_cast<float>(host_logits[i]);
                    if (val > 0) {
                        host_logits[i] = static_cast<Tval>(val / repetition_penalty);
                    } else {
                        host_logits[i] = static_cast<Tval>(val * repetition_penalty);
                    }
                }
            }


            // Copy penalized logits to sorted buffer (will be used as input to radixSort)
            CHECK_METAX(hcMemcpyAsync(sorted, host_logits.data(), n * sizeof(Tval), hcMemcpyHostToDevice, stream));
            CHECK_METAX(hcStreamSynchronize(stream));

            // sort with penalized logits
            fillIndices<<<grid, block, 0, stream>>>(indices, n);
            CHECK_METAX(radixSort(
                workspace_, workspace_size,
                sorted, sorted_out,
                indices, indices_out,
                n,
                stream));

            // Copy sorted_out back to sorted for softmax
            CHECK_METAX(hcMemcpyAsync(sorted, sorted_out, n * sizeof(Tval), hcMemcpyDeviceToDevice, stream));
        } else {
            // Set CUB workspace pointer and size
            workspace_ = reinterpret_cast<void *>(workspace);
            workspace_size = workspace_end - workspace;

            // sort
            fillIndices<<<grid, block, 0, stream>>>(indices, n);
            CHECK_METAX(radixSort(
                workspace_, workspace_size,
                logits, sorted,
                indices, indices_out,
                n,
                stream));
        }
        // softmax
        partialSoftmaxKernel<<<grid, block, 0, stream>>>(sorted, n, temperature);
        setSoftmaxMaxKernel<<<1, 1, 0, stream>>>(sorted);
        // sum
        CHECK_METAX(inclusiveSum(
            workspace_, workspace,
            sorted, n,
            stream));
        // sample
        // Handle disabled topk (0 or -1 means consider all tokens, like vLLM)
        int effective_topk = (topk <= 0) ? static_cast<int>(n) : topk;
        randomSampleKernel<<<1, 1, 0, stream>>>(
            result,
            sorted, indices_out, n,
            random_val, topp, effective_topk);

        return INFINI_STATUS_SUCCESS;
    }
};

} // namespace op::random_sample::metax
