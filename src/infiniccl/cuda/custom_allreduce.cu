#include "custom_allreduce.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../utils.h"

namespace infiniccl::cuda {

namespace {

using FlagType = uint32_t;

constexpr size_t kMaxCustomAllReduceBytes = 8 * 1024 * 1024;
// NVIDIA custom allreduce is a small-message decode fast path. Large prefill
// collectives should stay on NCCL, which is better optimized for bandwidth.
constexpr size_t kMaxCustomAllReduceFastPathBytes = 512 * 1024;
constexpr int kMaxCustomAllReduceWorldSize = 8;
constexpr int kSignalReady = 0;
constexpr int kSignalDone = 1;
constexpr int kSignalCount = 2;
constexpr int kSignalSlotsPerBlock = kSignalCount * kMaxCustomAllReduceWorldSize + 1;
constexpr int kSignalFlagSlot = kSignalCount * kMaxCustomAllReduceWorldSize;
constexpr int kAllReduceBlockSize = 512;
constexpr int kMaxAllReduceBlocks = 36;
constexpr size_t kSignalBufferBytes =
    static_cast<size_t>(kMaxAllReduceBlocks) * kSignalSlotsPerBlock * sizeof(FlagType);
constexpr size_t kTwoStageWorldSize4ThresholdBytes = 512 * 1024;
constexpr size_t kTwoStageWorldSize8ThresholdBytes = 256 * 1024;

} // namespace

struct RegisteredAllReduceBuffer {
    std::string key;
    void *local_buffer = nullptr;
    size_t bytes = 0;
    void **rank_buffers = nullptr;
};

struct CustomAllReduceContext {
    uint64_t group_id = 0;
    int rank = 0;
    int world_size = 1;
    int device_id = 0;
    size_t max_bytes = kMaxCustomAllReduceBytes;
    std::vector<int> device_ids;

    bool initialized = false;
    const char *disabled_reason = "custom allreduce context is not initialized";

    void *scratch = nullptr;
    FlagType *signals = nullptr;
    void **rank_scratch = nullptr;
    FlagType **rank_signals = nullptr;
    std::vector<RegisteredAllReduceBuffer> registered_buffers;
    std::vector<void **> retired_rank_buffers;
    std::vector<const infinicclAllReduceBackend_t *> rank_backends;

    const infinicclAllReduceBackend_t *local_backend = nullptr;
};

namespace {

CustomAllReduceCheckResult unsupported(const char *reason) {
    return CustomAllReduceCheckResult{false, reason};
}

CustomAllReduceCheckResult supported() {
    return CustomAllReduceCheckResult{true, nullptr};
}

void retire_rank_buffer_table(CustomAllReduceContext *ctx, void **rank_buffers) {
    if (ctx == nullptr || rank_buffers == nullptr) {
        return;
    }
    ctx->retired_rank_buffers.push_back(rank_buffers);
}

void free_retired_rank_buffer_tables(CustomAllReduceContext *ctx) {
    if (ctx == nullptr) {
        return;
    }
    for (auto *rank_buffers : ctx->retired_rank_buffers) {
        if (rank_buffers != nullptr) {
            (void)cudaFree(rank_buffers);
        }
    }
    ctx->retired_rank_buffers.clear();
}

void erase_registered_buffer(CustomAllReduceContext *ctx, const void *local_buffer, const std::string *key = nullptr) {
    if (ctx == nullptr) {
        return;
    }
    for (auto it = ctx->registered_buffers.begin(); it != ctx->registered_buffers.end();) {
        const bool match_buffer = local_buffer != nullptr && it->local_buffer == local_buffer;
        const bool match_key = key != nullptr && it->key == *key;
        if (match_buffer || match_key) {
            retire_rank_buffer_table(ctx, it->rank_buffers);
            it = ctx->registered_buffers.erase(it);
        } else {
            ++it;
        }
    }
}

void clear_registered_buffers(CustomAllReduceContext *ctx) {
    if (ctx == nullptr) {
        return;
    }
    for (auto &entry : ctx->registered_buffers) {
        retire_rank_buffer_table(ctx, entry.rank_buffers);
        entry.rank_buffers = nullptr;
    }
    ctx->registered_buffers.clear();
}

RegisteredAllReduceBuffer *find_registered_buffer(CustomAllReduceContext *ctx, const void *local_buffer, size_t bytes) {
    if (ctx == nullptr || local_buffer == nullptr) {
        return nullptr;
    }
    for (auto &entry : ctx->registered_buffers) {
        if (entry.local_buffer == local_buffer && bytes <= entry.bytes) {
            return &entry;
        }
    }
    return nullptr;
}

bool is_supported_dtype(infiniDtype_t datatype) {
    return datatype == INFINI_DTYPE_F32 ||
           datatype == INFINI_DTYPE_F16 ||
           datatype == INFINI_DTYPE_BF16;
}

bool is_supported_world_size(int world_size) {
    return world_size >= 2 && world_size <= kMaxCustomAllReduceWorldSize;
}

bool is_aligned_to(const void *ptr, size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

void clear_cuda_error() {
    (void)cudaGetLastError();
}

struct PendingRegisteredAllReduceBuffer {
    int world_size = 0;
    size_t bytes = 0;
    std::vector<void *> buffers;
    std::vector<CustomAllReduceContext *> contexts;
    std::vector<char> arrived;
    int arrived_count = 0;
    bool installed = false;
    infiniStatus_t status = INFINI_STATUS_SUCCESS;
    std::condition_variable cv;
};

std::atomic<uint64_t> next_custom_allreduce_group_id{1};
std::mutex registered_buffer_registry_mutex;
std::unordered_map<std::string, std::shared_ptr<PendingRegisteredAllReduceBuffer>> registered_buffer_registry;

std::string make_registered_buffer_key(uint64_t group_id, const char *key) {
    return std::to_string(group_id) + ":" + std::string(key);
}

void erase_registered_buffer_registry(uint64_t group_id) {
    const auto prefix = std::to_string(group_id) + ":";
    std::lock_guard<std::mutex> lock(registered_buffer_registry_mutex);
    for (auto it = registered_buffer_registry.begin(); it != registered_buffer_registry.end();) {
        if (it->first.compare(0, prefix.size(), prefix) == 0) {
            it = registered_buffer_registry.erase(it);
        } else {
            ++it;
        }
    }
}

infiniStatus_t install_registered_buffers(
    const std::vector<CustomAllReduceContext *> &contexts,
    const std::vector<void *> &rank_buffers,
    size_t bytes,
    const std::string &key) {

    if (contexts.empty() || contexts.size() != rank_buffers.size() || bytes == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const int ndevice = static_cast<int>(contexts.size());
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;

    auto restore = [&]() {
        if (restore_device) {
            (void)cudaSetDevice(previous_device);
        }
    };
    auto rollback_key = [&]() {
        if (key.empty()) {
            return;
        }
        for (auto *ctx : contexts) {
            if (ctx != nullptr) {
                (void)cudaSetDevice(ctx->device_id);
                erase_registered_buffer(ctx, nullptr, &key);
            }
        }
    };

    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = contexts[static_cast<size_t>(i)];
        if (ctx == nullptr || rank_buffers[static_cast<size_t>(i)] == nullptr) {
            restore();
            return INFINI_STATUS_NULL_POINTER;
        }
        if (!ctx->initialized || ctx->world_size != ndevice) {
            restore();
            return INFINI_STATUS_BAD_PARAM;
        }
    }

    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = contexts[static_cast<size_t>(i)];
        auto *local_buffer = rank_buffers[static_cast<size_t>(i)];
        if (cudaSetDevice(ctx->device_id) != cudaSuccess) {
            clear_cuda_error();
            rollback_key();
            restore();
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        const std::string *key_to_erase = key.empty() ? nullptr : &key;
        erase_registered_buffer(ctx, local_buffer, key_to_erase);
        RegisteredAllReduceBuffer entry{};
        entry.key = key;
        entry.local_buffer = local_buffer;
        entry.bytes = bytes;
        if (cudaMalloc(reinterpret_cast<void **>(&entry.rank_buffers), rank_buffers.size() * sizeof(void *)) != cudaSuccess ||
            cudaMemcpy(entry.rank_buffers,
                       rank_buffers.data(),
                       rank_buffers.size() * sizeof(void *),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            clear_cuda_error();
            if (entry.rank_buffers != nullptr) {
                (void)cudaFree(entry.rank_buffers);
            }
            rollback_key();
            restore();
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        ctx->registered_buffers.push_back(entry);
    }

    restore();
    return INFINI_STATUS_SUCCESS;
}

void mark_disabled(const std::vector<CustomAllReduceContext *> &contexts, const char *reason) {
    for (auto *ctx : contexts) {
        if (ctx != nullptr) {
            ctx->initialized = false;
            ctx->disabled_reason = reason;
        }
    }
}

void release_buffers(const std::vector<CustomAllReduceContext *> &contexts) {
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    for (auto *ctx : contexts) {
        if (ctx == nullptr) {
            continue;
        }
        if (ctx->scratch != nullptr || ctx->signals != nullptr ||
            !ctx->registered_buffers.empty() || !ctx->retired_rank_buffers.empty()) {
            (void)cudaSetDevice(ctx->device_id);
            clear_registered_buffers(ctx);
            free_retired_rank_buffer_tables(ctx);
            if (ctx->scratch != nullptr) {
                (void)cudaFree(ctx->scratch);
                ctx->scratch = nullptr;
            }
            if (ctx->signals != nullptr) {
                (void)cudaFree(ctx->signals);
                ctx->signals = nullptr;
            }
            if (ctx->rank_scratch != nullptr) {
                (void)cudaFree(ctx->rank_scratch);
                ctx->rank_scratch = nullptr;
            }
            if (ctx->rank_signals != nullptr) {
                (void)cudaFree(ctx->rank_signals);
                ctx->rank_signals = nullptr;
            }
        }
        ctx->rank_backends.clear();
        ctx->initialized = false;
    }
    if (restore_device) {
        (void)cudaSetDevice(previous_device);
    }
}

bool device_supports_custom_allreduce(int device_id) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return false;
    }
    return prop.major >= 8;
}

bool enable_peer_access(int device_id, int peer_device_id) {
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return false;
    }

    int can_access = 0;
    if (cudaDeviceCanAccessPeer(&can_access, device_id, peer_device_id) != cudaSuccess) {
        clear_cuda_error();
        return false;
    }
    if (can_access == 0) {
        return false;
    }

    const cudaError_t err = cudaDeviceEnablePeerAccess(peer_device_id, 0);
    if (err == cudaSuccess) {
        return true;
    }
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        clear_cuda_error();
        return true;
    }
    clear_cuda_error();
    return false;
}

__device__ __forceinline__ FlagType load_signal_volatile(const FlagType *signal) {
    FlagType value;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(signal));
    return value;
}

__device__ __forceinline__ void store_signal_volatile(FlagType *signal, FlagType value) {
    asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(value), "l"(signal));
}

__device__ __forceinline__ FlagType load_signal_acquire(const FlagType *signal) {
    FlagType value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(signal));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(value) : "l"(signal));
#endif
    return value;
}

__device__ __forceinline__ void store_signal_release(FlagType *signal, FlagType value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(value), "l"(signal));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" :: "r"(value), "l"(signal));
#endif
}

__device__ __forceinline__ int signal_slot_index(int block, int slot, int rank) {
    return block * kSignalSlotsPerBlock + slot * kMaxCustomAllReduceWorldSize + rank;
}

__device__ __forceinline__ int signal_flag_index(int block) {
    return block * kSignalSlotsPerBlock + kSignalFlagSlot;
}

__device__ __forceinline__ void multi_gpu_barrier_start(
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank,
    int slot) {

    const FlagType flag = load_signal_volatile(local_signals + signal_flag_index(blockIdx.x)) + 1U;
    if (threadIdx.x < static_cast<unsigned int>(world_size)) {
        const int peer = static_cast<int>(threadIdx.x);
        auto *peer_signal = rank_signals[peer] + signal_slot_index(blockIdx.x, slot, rank);
        const auto *local_peer_signal = local_signals + signal_slot_index(blockIdx.x, slot, peer);
        store_signal_volatile(peer_signal, flag);
        while (load_signal_volatile(local_peer_signal) < flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        store_signal_volatile(local_signals + signal_flag_index(blockIdx.x), flag);
    }
}

template <bool final_sync>
__device__ __forceinline__ void multi_gpu_barrier_end(
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank,
    int slot) {

    __syncthreads();
    const FlagType flag = load_signal_volatile(local_signals + signal_flag_index(blockIdx.x)) + 1U;
    if (threadIdx.x < static_cast<unsigned int>(world_size)) {
        const int peer = static_cast<int>(threadIdx.x);
        auto *peer_signal = rank_signals[peer] + signal_slot_index(blockIdx.x, slot, rank);
        const auto *local_peer_signal = local_signals + signal_slot_index(blockIdx.x, slot, peer);
        if constexpr (final_sync) {
            store_signal_volatile(peer_signal, flag);
            while (load_signal_volatile(local_peer_signal) < flag) {
            }
        } else {
            store_signal_release(peer_signal, flag);
            while (load_signal_acquire(local_peer_signal) < flag) {
            }
        }
    }
    if constexpr (!final_sync) {
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        store_signal_volatile(local_signals + signal_flag_index(blockIdx.x), flag);
    }
}

__device__ __forceinline__ void multi_gpu_barrier(
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank,
    int slot) {
    if (slot == kSignalDone) {
        multi_gpu_barrier_end<true>(local_signals, rank_signals, world_size, rank, slot);
    } else {
        multi_gpu_barrier_start(local_signals, rank_signals, world_size, rank, slot);
    }
}

__device__ __forceinline__ void multi_gpu_barrier_middle(
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank,
    int slot) {
    multi_gpu_barrier_end<false>(local_signals, rank_signals, world_size, rank, slot);
}

__device__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ uint16_t float_to_bf16(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int lsb = (bits >> 16) & 1U;
    const unsigned int rounded = bits + 0x7fffU + lsb;
    return static_cast<uint16_t>(rounded >> 16);
}

__global__ void allreduce_1stage_f32_kernel(
    float *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += static_cast<const float *>(rank_scratch[rank_idx])[idx];
        }
        recvbuf[idx] = sum;
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_1stage_f16_kernel(
    __half *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += __half2float(static_cast<const __half *>(rank_scratch[rank_idx])[idx]);
        }
        recvbuf[idx] = __float2half(sum);
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_1stage_bf16_kernel(
    uint16_t *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t stride = blockDim.x * gridDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += bf16_to_float(static_cast<const uint16_t *>(rank_scratch[rank_idx])[idx]);
        }
        recvbuf[idx] = float_to_bf16(sum);
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_2stage_f32_kernel(
    float *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    auto *local_scratch = static_cast<float *>(rank_scratch[rank]);
    const size_t stride = blockDim.x * gridDim.x;

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t begin = count * static_cast<size_t>(rank) / static_cast<size_t>(world_size);
    const size_t end = count * static_cast<size_t>(rank + 1) / static_cast<size_t>(world_size);
    for (size_t idx = begin + blockIdx.x * blockDim.x + threadIdx.x; idx < end; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += static_cast<const float *>(rank_scratch[rank_idx])[idx];
        }
        local_scratch[idx] = sum;
    }

    multi_gpu_barrier_middle(local_signals, rank_signals, world_size, rank, kSignalReady);

    for (int owner = 0; owner < world_size; ++owner) {
        const size_t owner_begin = count * static_cast<size_t>(owner) / static_cast<size_t>(world_size);
        const size_t owner_end = count * static_cast<size_t>(owner + 1) / static_cast<size_t>(world_size);
        const auto *owner_scratch = static_cast<const float *>(rank_scratch[owner]);
        for (size_t idx = owner_begin + blockIdx.x * blockDim.x + threadIdx.x; idx < owner_end; idx += stride) {
            recvbuf[idx] = owner_scratch[idx];
        }
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_2stage_f16_kernel(
    __half *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    auto *local_scratch = static_cast<__half *>(rank_scratch[rank]);
    const size_t stride = blockDim.x * gridDim.x;

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t begin = count * static_cast<size_t>(rank) / static_cast<size_t>(world_size);
    const size_t end = count * static_cast<size_t>(rank + 1) / static_cast<size_t>(world_size);
    for (size_t idx = begin + blockIdx.x * blockDim.x + threadIdx.x; idx < end; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += __half2float(static_cast<const __half *>(rank_scratch[rank_idx])[idx]);
        }
        local_scratch[idx] = __float2half(sum);
    }

    multi_gpu_barrier_middle(local_signals, rank_signals, world_size, rank, kSignalReady);

    for (int owner = 0; owner < world_size; ++owner) {
        const size_t owner_begin = count * static_cast<size_t>(owner) / static_cast<size_t>(world_size);
        const size_t owner_end = count * static_cast<size_t>(owner + 1) / static_cast<size_t>(world_size);
        const auto *owner_scratch = static_cast<const __half *>(rank_scratch[owner]);
        for (size_t idx = owner_begin + blockIdx.x * blockDim.x + threadIdx.x; idx < owner_end; idx += stride) {
            recvbuf[idx] = owner_scratch[idx];
        }
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_2stage_bf16_kernel(
    uint16_t *recvbuf,
    void **rank_scratch,
    size_t count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    auto *local_scratch = static_cast<uint16_t *>(rank_scratch[rank]);
    const size_t stride = blockDim.x * gridDim.x;

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t begin = count * static_cast<size_t>(rank) / static_cast<size_t>(world_size);
    const size_t end = count * static_cast<size_t>(rank + 1) / static_cast<size_t>(world_size);
    for (size_t idx = begin + blockIdx.x * blockDim.x + threadIdx.x; idx < end; idx += stride) {
        float sum = 0.0f;
        for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
            sum += bf16_to_float(static_cast<const uint16_t *>(rank_scratch[rank_idx])[idx]);
        }
        local_scratch[idx] = float_to_bf16(sum);
    }

    multi_gpu_barrier_middle(local_signals, rank_signals, world_size, rank, kSignalReady);

    for (int owner = 0; owner < world_size; ++owner) {
        const size_t owner_begin = count * static_cast<size_t>(owner) / static_cast<size_t>(world_size);
        const size_t owner_end = count * static_cast<size_t>(owner + 1) / static_cast<size_t>(world_size);
        const auto *owner_scratch = static_cast<const uint16_t *>(rank_scratch[owner]);
        for (size_t idx = owner_begin + blockIdx.x * blockDim.x + threadIdx.x; idx < owner_end; idx += stride) {
            recvbuf[idx] = owner_scratch[idx];
        }
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__device__ uint4 reduce_packed_f32(
    void **rank_scratch,
    size_t vec_idx,
    int world_size) {

    float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
    for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
        const auto value = reinterpret_cast<const float4 *>(rank_scratch[rank_idx])[vec_idx];
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
    }
    union {
        float4 f;
        uint4 u;
    } out{sum};
    return out.u;
}

__device__ uint4 reduce_packed_f16(
    void **rank_scratch,
    size_t vec_idx,
    int world_size) {

    float sum[8]{};
    for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
        const uint4 packed = static_cast<const uint4 *>(rank_scratch[rank_idx])[vec_idx];
        const auto *value = reinterpret_cast<const __half *>(&packed);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            sum[item] += __half2float(value[item]);
        }
    }

    uint4 out{};
    auto *packed_out = reinterpret_cast<__half *>(&out);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        packed_out[item] = __float2half(sum[item]);
    }
    return out;
}

__device__ uint4 reduce_packed_bf16(
    void **rank_scratch,
    size_t vec_idx,
    int world_size) {

    float sum[8]{};
    for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
        const uint4 packed = static_cast<const uint4 *>(rank_scratch[rank_idx])[vec_idx];
        const auto *value = reinterpret_cast<const uint16_t *>(&packed);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            sum[item] += bf16_to_float(value[item]);
        }
    }

    uint4 out{};
    auto *packed_out = reinterpret_cast<uint16_t *>(&out);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        packed_out[item] = float_to_bf16(sum[item]);
    }
    return out;
}

__device__ uint4 reduce_packed_bf16_tp2(uint4 a, uint4 b) {
    const auto *av = reinterpret_cast<const uint16_t *>(&a);
    const auto *bv = reinterpret_cast<const uint16_t *>(&b);
    uint4 out{};
    auto *ov = reinterpret_cast<uint16_t *>(&out);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        ov[item] = float_to_bf16(bf16_to_float(av[item]) + bf16_to_float(bv[item]));
    }
    return out;
}

__device__ uint4 reduce_packed(
    void **rank_scratch,
    size_t vec_idx,
    int world_size,
    infiniDtype_t datatype) {

    switch (datatype) {
    case INFINI_DTYPE_F32:
        return reduce_packed_f32(rank_scratch, vec_idx, world_size);
    case INFINI_DTYPE_F16:
        return reduce_packed_f16(rank_scratch, vec_idx, world_size);
    case INFINI_DTYPE_BF16:
        return reduce_packed_bf16(rank_scratch, vec_idx, world_size);
    default:
        return uint4{};
    }
}

__global__ void allreduce_1stage_packed_bf16_tp2_kernel(
    uint4 *recvbuf,
    void **rank_scratch,
    size_t vec_count,
    FlagType *local_signals,
    FlagType **rank_signals,
    int rank) {

    multi_gpu_barrier(local_signals, rank_signals, 2, rank, kSignalReady);

    const auto *rank0 = static_cast<const uint4 *>(rank_scratch[0]);
    const auto *rank1 = static_cast<const uint4 *>(rank_scratch[1]);
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < vec_count; vec_idx += stride) {
        recvbuf[vec_idx] = reduce_packed_bf16_tp2(rank0[vec_idx], rank1[vec_idx]);
    }

    multi_gpu_barrier(local_signals, rank_signals, 2, rank, kSignalDone);
}

__global__ void allreduce_1stage_packed_kernel(
    uint4 *recvbuf,
    void **rank_scratch,
    size_t vec_count,
    infiniDtype_t datatype,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t stride = blockDim.x * gridDim.x;
    for (size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < vec_count; vec_idx += stride) {
        recvbuf[vec_idx] = reduce_packed(rank_scratch, vec_idx, world_size, datatype);
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}

__global__ void allreduce_2stage_packed_kernel(
    uint4 *recvbuf,
    void **rank_scratch,
    size_t vec_count,
    infiniDtype_t datatype,
    FlagType *local_signals,
    FlagType **rank_signals,
    int world_size,
    int rank) {

    auto *local_scratch = static_cast<uint4 *>(rank_scratch[rank]);
    const size_t stride = blockDim.x * gridDim.x;

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalReady);

    const size_t begin = vec_count * static_cast<size_t>(rank) / static_cast<size_t>(world_size);
    const size_t end = vec_count * static_cast<size_t>(rank + 1) / static_cast<size_t>(world_size);
    for (size_t vec_idx = begin + blockIdx.x * blockDim.x + threadIdx.x; vec_idx < end; vec_idx += stride) {
        local_scratch[vec_idx] = reduce_packed(rank_scratch, vec_idx, world_size, datatype);
    }

    multi_gpu_barrier_middle(local_signals, rank_signals, world_size, rank, kSignalReady);

    for (int owner = 0; owner < world_size; ++owner) {
        const size_t owner_begin = vec_count * static_cast<size_t>(owner) / static_cast<size_t>(world_size);
        const size_t owner_end = vec_count * static_cast<size_t>(owner + 1) / static_cast<size_t>(world_size);
        const auto *owner_scratch = static_cast<const uint4 *>(rank_scratch[owner]);
        for (size_t vec_idx = owner_begin + blockIdx.x * blockDim.x + threadIdx.x;
             vec_idx < owner_end;
             vec_idx += stride) {
            recvbuf[vec_idx] = owner_scratch[vec_idx];
        }
    }

    multi_gpu_barrier(local_signals, rank_signals, world_size, rank, kSignalDone);
}
infiniStatus_t check_last_cuda_launch() {
    return cudaPeekAtLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

int allreduce_block_count(size_t count) {
    const size_t blocks = (count + kAllReduceBlockSize - 1) / kAllReduceBlockSize;
    if (blocks == 0) {
        return 1;
    }
    return static_cast<int>(blocks > kMaxAllReduceBlocks ? kMaxAllReduceBlocks : blocks);
}

bool use_two_stage_allreduce(int world_size, size_t bytes) {
    if (world_size <= 2) {
        return false;
    }
    if (world_size <= 4) {
        return bytes > kTwoStageWorldSize4ThresholdBytes;
    }
    return bytes > kTwoStageWorldSize8ThresholdBytes;
}

bool should_register_custom_allreduce_buffer(size_t bytes) {
    return bytes > 0 && bytes <= kMaxCustomAllReduceFastPathBytes;
}

bool should_use_registered_custom_allreduce(int world_size, size_t bytes) {
    if (!is_supported_world_size(world_size)) {
        return false;
    }
    if (bytes == 0 || bytes > kMaxCustomAllReduceFastPathBytes) {
        return false;
    }
    return true;
}

infiniStatus_t launch_allreduce_kernel(
    void *recvbuf,
    void **rank_scratch,
    size_t count,
    size_t bytes,
    infiniDtype_t datatype,
    int world_size,
    int rank,
    bool use_packed,
    bool force_two_stage,
    FlagType *local_signals,
    FlagType **rank_signals,
    cudaStream_t stream) {

    const bool two_stage = force_two_stage || use_two_stage_allreduce(world_size, bytes);
    if (use_packed) {
        const size_t vec_count = bytes / sizeof(uint4);
        const int blocks = allreduce_block_count(vec_count);
        if (!two_stage && world_size == 2 && datatype == INFINI_DTYPE_BF16) {
            allreduce_1stage_packed_bf16_tp2_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<uint4 *>(recvbuf),
                rank_scratch,
                vec_count,
                local_signals,
                rank_signals,
                rank);
            return check_last_cuda_launch();
        }
        if (two_stage) {
            allreduce_2stage_packed_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<uint4 *>(recvbuf),
                rank_scratch,
                vec_count,
                datatype,
                local_signals,
                rank_signals,
                world_size,
                rank);
        } else {
            allreduce_1stage_packed_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<uint4 *>(recvbuf),
                rank_scratch,
                vec_count,
                datatype,
                local_signals,
                rank_signals,
                world_size,
                rank);
        }
        return check_last_cuda_launch();
    }

    const int blocks = allreduce_block_count(count);
    switch (datatype) {
    case INFINI_DTYPE_F32:
        if (two_stage) {
            allreduce_2stage_f32_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<float *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        } else {
            allreduce_1stage_f32_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<float *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        }
        return check_last_cuda_launch();
    case INFINI_DTYPE_F16:
        if (two_stage) {
            allreduce_2stage_f16_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<__half *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        } else {
            allreduce_1stage_f16_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<__half *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        }
        return check_last_cuda_launch();
    case INFINI_DTYPE_BF16:
        if (two_stage) {
            allreduce_2stage_bf16_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<uint16_t *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        } else {
            allreduce_1stage_bf16_kernel<<<blocks, kAllReduceBlockSize, 0, stream>>>(
                static_cast<uint16_t *>(recvbuf),
                rank_scratch,
                count,
                local_signals,
                rank_signals,
                world_size,
                rank);
        }
        return check_last_cuda_launch();
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

cudaStream_t get_cuda_stream(infinirtStream_t stream) {
    return stream == nullptr ? 0 : static_cast<cudaStream_t>(stream);
}

} // namespace

CustomAllReduceContext *createCustomAllReduceContext(
    int rank,
    int world_size,
    int device_id,
    const int *device_ids) {

    auto *ctx = new CustomAllReduceContext{};
    ctx->rank = rank;
    ctx->world_size = world_size;
    ctx->device_id = device_id;
    if (device_ids != nullptr && world_size > 0) {
        ctx->device_ids.assign(device_ids, device_ids + world_size);
    }
    return ctx;
}

void initializeCustomAllReduceContexts(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    if (comms == nullptr || device_ids == nullptr || ndevice <= 0) {
        return;
    }

    const auto group_id = next_custom_allreduce_group_id.fetch_add(1, std::memory_order_relaxed);
    std::vector<CustomAllReduceContext *> contexts;
    contexts.reserve(static_cast<size_t>(ndevice));
    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = static_cast<CustomAllReduceContext *>(comms[i]->custom_allreduce_context);
        if (ctx != nullptr) {
            ctx->group_id = group_id;
            ctx->local_backend = &comms[i]->allreduce_backend;
            ctx->rank_backends.assign(static_cast<size_t>(ndevice), nullptr);
            ctx->disabled_reason = "custom allreduce supports TP sizes from 2 to 8 in this phase";
        }
        contexts.push_back(ctx);
    }

    if (!is_supported_world_size(ndevice)) {
        mark_disabled(contexts, "custom allreduce supports TP sizes from 2 to 8 in this phase");
        return;
    }

    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;

    for (auto *ctx : contexts) {
        if (ctx == nullptr || !device_supports_custom_allreduce(ctx->device_id)) {
            release_buffers(contexts);
            mark_disabled(contexts, "custom allreduce requires an SM80 or newer CUDA device");
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return;
        }
    }

    for (int i = 0; i < ndevice; ++i) {
        for (int peer = 0; peer < ndevice; ++peer) {
            if (peer == i) {
                continue;
            }
            if (!enable_peer_access(device_ids[i], device_ids[peer])) {
                release_buffers(contexts);
                mark_disabled(contexts, "custom allreduce requires peer access between all TP ranks");
                if (restore_device) {
                    (void)cudaSetDevice(previous_device);
                }
                return;
            }
        }
    }

    for (auto *ctx : contexts) {
        if (ctx == nullptr || cudaSetDevice(ctx->device_id) != cudaSuccess) {
            release_buffers(contexts);
            mark_disabled(contexts, "custom allreduce failed to set the rank device during initialization");
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return;
        }
        if (cudaMalloc(&ctx->scratch, ctx->max_bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&ctx->signals), kSignalBufferBytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&ctx->rank_scratch), ndevice * sizeof(void *)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&ctx->rank_signals), ndevice * sizeof(FlagType *)) != cudaSuccess ||
            cudaMemset(ctx->signals, 0, kSignalBufferBytes) != cudaSuccess) {
            clear_cuda_error();
            release_buffers(contexts);
            mark_disabled(contexts, "custom allreduce failed to allocate scratch or signal buffers");
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return;
        }
    }

    std::vector<void *> scratch_ptrs(static_cast<size_t>(ndevice));
    std::vector<FlagType *> signal_ptrs(static_cast<size_t>(ndevice));
    for (int i = 0; i < ndevice; ++i) {
        scratch_ptrs[static_cast<size_t>(i)] = contexts[static_cast<size_t>(i)]->scratch;
        signal_ptrs[static_cast<size_t>(i)] = contexts[static_cast<size_t>(i)]->signals;
    }
    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = contexts[static_cast<size_t>(i)];
        for (int rank_idx = 0; rank_idx < ndevice; ++rank_idx) {
            ctx->rank_backends[static_cast<size_t>(rank_idx)] = &comms[rank_idx]->allreduce_backend;
        }
    }

    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = contexts[i];
        if (cudaSetDevice(ctx->device_id) != cudaSuccess ||
            cudaMemcpy(ctx->rank_scratch,
                       scratch_ptrs.data(),
                       scratch_ptrs.size() * sizeof(void *),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(ctx->rank_signals,
                       signal_ptrs.data(),
                       signal_ptrs.size() * sizeof(FlagType *),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            clear_cuda_error();
            release_buffers(contexts);
            mark_disabled(contexts, "custom allreduce failed to initialize rank pointer tables");
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return;
        }
        ctx->initialized = true;
        ctx->disabled_reason = nullptr;
    }

    if (restore_device) {
        (void)cudaSetDevice(previous_device);
    }
}

void destroyCustomAllReduceContext(CustomAllReduceContext *ctx) {
    if (ctx == nullptr) {
        return;
    }
    erase_registered_buffer_registry(ctx->group_id);
    std::vector<CustomAllReduceContext *> contexts{ctx};
    release_buffers(contexts);
    delete ctx;
}

infiniStatus_t clearCustomAllReduceBuffers(infinicclComm_t *comms, int ndevice) {
    if (comms == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (ndevice <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    for (int i = 0; i < ndevice; ++i) {
        if (comms[i] == nullptr || comms[i]->custom_allreduce_context == nullptr) {
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return INFINI_STATUS_NULL_POINTER;
        }
        auto *ctx = static_cast<CustomAllReduceContext *>(comms[i]->custom_allreduce_context);
        if (cudaSetDevice(ctx->device_id) != cudaSuccess) {
            clear_cuda_error();
            if (restore_device) {
                (void)cudaSetDevice(previous_device);
            }
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        clear_registered_buffers(ctx);
    }
    for (int i = 0; i < ndevice; ++i) {
        auto *ctx = static_cast<CustomAllReduceContext *>(comms[i]->custom_allreduce_context);
        erase_registered_buffer_registry(ctx->group_id);
    }
    if (restore_device) {
        (void)cudaSetDevice(previous_device);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t registerCustomAllReduceBuffers(
    infinicclComm_t *comms,
    int ndevice,
    void **buffers,
    size_t bytes) {

    if (comms == nullptr || buffers == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (ndevice <= 0 || bytes == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (!should_register_custom_allreduce_buffer(bytes)) {
        return INFINI_STATUS_SUCCESS;
    }

    std::vector<CustomAllReduceContext *> contexts(static_cast<size_t>(ndevice));
    std::vector<void *> rank_buffers(static_cast<size_t>(ndevice));
    for (int i = 0; i < ndevice; ++i) {
        if (comms[i] == nullptr || buffers[i] == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }
        if (comms[i]->allreduce_backend == INFINICCL_ALLREDUCE_BACKEND_NCCL) {
            return INFINI_STATUS_SUCCESS;
        }
        if (comms[i]->custom_allreduce_context == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }
        contexts[static_cast<size_t>(i)] = static_cast<CustomAllReduceContext *>(comms[i]->custom_allreduce_context);
        rank_buffers[static_cast<size_t>(i)] = buffers[i];
    }

    return install_registered_buffers(contexts, rank_buffers, bytes, std::string());
}

infiniStatus_t registerCustomAllReduceBuffer(
    infinicclComm_t comm,
    const char *key,
    void *buffer,
    size_t bytes) {

    if (comm == nullptr || key == nullptr || buffer == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (bytes == 0 || key[0] == '\0') {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (!should_register_custom_allreduce_buffer(bytes)) {
        return INFINI_STATUS_SUCCESS;
    }
    if (comm->allreduce_backend == INFINICCL_ALLREDUCE_BACKEND_NCCL) {
        return INFINI_STATUS_SUCCESS;
    }
    if (comm->custom_allreduce_context == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto *ctx = static_cast<CustomAllReduceContext *>(comm->custom_allreduce_context);
    if (!ctx->initialized || ctx->world_size <= 0 || ctx->rank < 0 || ctx->rank >= ctx->world_size) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const std::string user_key(key);
    if (auto *registered = find_registered_buffer(ctx, buffer, bytes);
        registered != nullptr && registered->key == user_key) {
        return INFINI_STATUS_SUCCESS;
    }

    const auto registry_key = make_registered_buffer_key(ctx->group_id, key);
    std::unique_lock<std::mutex> lock(registered_buffer_registry_mutex);
    auto &registry_slot = registered_buffer_registry[registry_key];
    const auto rank = static_cast<size_t>(ctx->rank);

    if (registry_slot != nullptr && registry_slot->installed) {
        if (registry_slot->world_size == ctx->world_size &&
            rank < registry_slot->buffers.size() &&
            registry_slot->contexts[rank] == ctx &&
            registry_slot->buffers[rank] == buffer &&
            bytes <= registry_slot->bytes) {
            return INFINI_STATUS_SUCCESS;
        }
        registry_slot.reset();
    }

    if (registry_slot == nullptr) {
        registry_slot = std::make_shared<PendingRegisteredAllReduceBuffer>();
        registry_slot->world_size = ctx->world_size;
        registry_slot->bytes = bytes;
        registry_slot->buffers.assign(static_cast<size_t>(ctx->world_size), nullptr);
        registry_slot->contexts.assign(static_cast<size_t>(ctx->world_size), nullptr);
        registry_slot->arrived.assign(static_cast<size_t>(ctx->world_size), 0);
    }

    auto slot = registry_slot;

    if (slot->world_size != ctx->world_size || slot->bytes != bytes) {
        slot->status = INFINI_STATUS_BAD_PARAM;
        slot->installed = true;
        slot->cv.notify_all();
        return INFINI_STATUS_BAD_PARAM;
    }

    if (slot->arrived[rank] != 0) {
        if (slot->contexts[rank] != ctx || slot->buffers[rank] != buffer) {
            slot->status = INFINI_STATUS_BAD_PARAM;
            slot->installed = true;
            slot->cv.notify_all();
            return INFINI_STATUS_BAD_PARAM;
        }
    } else {
        slot->contexts[rank] = ctx;
        slot->buffers[rank] = buffer;
        slot->arrived[rank] = 1;
        ++slot->arrived_count;
    }

    if (slot->arrived_count == slot->world_size) {
        slot->status = install_registered_buffers(slot->contexts, slot->buffers, slot->bytes, user_key);
        slot->installed = true;
        slot->cv.notify_all();
        return slot->status;
    }

    slot->cv.wait(lock, [slot]() { return slot->installed; });
    return slot->status;
}

CustomAllReduceCheckResult canUseCustomAllReduce(
    const CustomAllReduceContext *ctx,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op) {

    if (ctx == nullptr) {
        return unsupported("custom allreduce context is not initialized");
    }
    if (ctx->local_backend == nullptr || *ctx->local_backend == INFINICCL_ALLREDUCE_BACKEND_NCCL) {
        return unsupported("custom allreduce backend is not enabled on this rank");
    }
    if (!ctx->initialized) {
        return unsupported(ctx->disabled_reason != nullptr ? ctx->disabled_reason : "custom allreduce context is disabled");
    }
    if (!is_supported_world_size(ctx->world_size)) {
        return unsupported("custom allreduce supports TP sizes from 2 to 8 in this phase");
    }
    if (ctx->rank_backends.size() != static_cast<size_t>(ctx->world_size)) {
        return unsupported("custom allreduce rank backend table is not initialized");
    }
    for (int rank_idx = 0; rank_idx < ctx->world_size; ++rank_idx) {
        const auto *backend = ctx->rank_backends[static_cast<size_t>(rank_idx)];
        if (backend == nullptr || *backend == INFINICCL_ALLREDUCE_BACKEND_NCCL) {
            return unsupported("custom allreduce backend is not enabled on all TP ranks");
        }
    }
    if (!is_supported_dtype(datatype)) {
        return unsupported("custom allreduce supports only F32, F16, and BF16");
    }
    if (op != INFINICCL_SUM) {
        return unsupported("custom allreduce supports only SUM");
    }
    if (count == 0) {
        return unsupported("custom allreduce does not handle empty tensors");
    }

    const size_t dtype_size = infiniSizeOf(datatype);
    if (dtype_size == 0) {
        return unsupported("custom allreduce received an invalid dtype");
    }
    if (!is_aligned_to(ctx->scratch, dtype_size)) {
        return unsupported("custom allreduce scratch buffers are not dtype-aligned");
    }
    if (count > ctx->max_bytes / dtype_size) {
        return unsupported("custom allreduce message is larger than the configured fast-path limit");
    }

    if (ctx->scratch == nullptr || ctx->signals == nullptr ||
        ctx->rank_scratch == nullptr || ctx->rank_signals == nullptr) {
        return unsupported("custom allreduce scratch or signal buffers are not initialized");
    }

    return supported();
}

infiniStatus_t tryCustomAllReduce(
    CustomAllReduceContext *ctx,
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinirtStream_t stream,
    bool *handled) {

    if (handled == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *handled = false;

    const auto check = canUseCustomAllReduce(ctx, count, datatype, op);
    if (!check.supported) {
        return INFINI_STATUS_SUCCESS;
    }

    const size_t bytes = count * infiniSizeOf(datatype);
    auto cuda_stream = get_cuda_stream(stream);
    auto *registered = find_registered_buffer(ctx, sendbuf, bytes);
    if (registered == nullptr || !should_use_registered_custom_allreduce(ctx->world_size, bytes)) {
        return INFINI_STATUS_SUCCESS;
    }

    void **rank_buffers = registered->rank_buffers;
    bool force_two_stage = sendbuf == recvbuf;

    const bool use_packed = bytes % sizeof(uint4) == 0 &&
                            is_aligned_to(sendbuf, sizeof(uint4)) &&
                            is_aligned_to(recvbuf, sizeof(uint4)) &&
                            (registered != nullptr || is_aligned_to(ctx->scratch, sizeof(uint4)));
    auto status = launch_allreduce_kernel(
        recvbuf,
        rank_buffers,
        count,
        bytes,
        datatype,
        ctx->world_size,
        ctx->rank,
        use_packed,
        force_two_stage,
        ctx->signals,
        ctx->rank_signals,
        cuda_stream);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    *handled = true;
    return INFINI_STATUS_SUCCESS;
}

} // namespace infiniccl::cuda