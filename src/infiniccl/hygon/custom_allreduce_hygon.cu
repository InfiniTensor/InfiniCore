#include "custom_allreduce_hygon.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace infiniccl::hygon {
namespace {

extern "C" int hipExtMallocWithFlags(
    void **ptr, size_t size_bytes, unsigned int flags);

constexpr int MAX_BLOCKS = 80;
constexpr int MAX_RANKS = 8;
constexpr int THREADS = 512;
constexpr size_t MAX_BUFFER_BYTES = 4 * 1024 * 1024;

struct Signal {
    alignas(128) uint32_t start[MAX_BLOCKS][MAX_RANKS];
    alignas(128) uint32_t end[MAX_BLOCKS][MAX_RANKS];
    alignas(128) uint32_t flag[MAX_BLOCKS];
};

struct alignas(16) RankData {
    const void *ptrs[MAX_RANKS];
};

struct alignas(16) RankSignals {
    Signal *signals[MAX_RANKS];
};

struct State {
    int rank;
    int world_size;
    int device_id;
    void *allocation;
    void *local_buffer;
    RankData buffers;
    RankSignals signals;
    cudaEvent_t release_event;
    std::vector<void *> opened_ipc;
};

std::mutex state_mutex;
std::unordered_map<infinicclComm_t, State *> states;

bool enabledByEnvironment() {
    const char *value = std::getenv("INFINICCL_HYGON_CUSTOM_ALLREDUCE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

void cleanupState(State *state) {
    if (state == nullptr) {
        return;
    }
    cudaSetDevice(state->device_id);
    if (state->release_event != nullptr) {
        cudaEventDestroy(state->release_event);
    }
    for (void *ptr : state->opened_ipc) {
        cudaIpcCloseMemHandle(ptr);
    }
    if (state->allocation != nullptr) {
        cudaFree(state->allocation);
    }
    delete state;
}

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T values[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(
    __nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(
    float value) {
    return __float2bfloat16(value);
}

template <int NRANKS>
__device__ __forceinline__ void startSync(
    const RankSignals &signals,
    Signal *self,
    int rank) {
    const uint32_t flag = self->flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __scoped_atomic_store_n(
            &signals.signals[threadIdx.x]->start[blockIdx.x][rank],
            flag,
            __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_RELAXED,
                   __MEMORY_SCOPE_DEVICE)
               < flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = flag;
    }
}

template <int NRANKS>
__device__ __forceinline__ void endSync(
    const RankSignals &signals,
    Signal *self,
    int rank) {
    __syncthreads();
    const uint32_t flag = self->flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __scoped_atomic_store_n(
            &signals.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag,
            __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_RELAXED,
                   __MEMORY_SCOPE_DEVICE)
               < flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = flag;
    }
}

template <typename T, int NRANKS>
__global__ __launch_bounds__(THREADS, 1) void customAllReduceKernel(
    RankData rank_data,
    RankSignals signals,
    Signal *self,
    T *output,
    int rank,
    int packed_count) {
    constexpr int PACK_SIZE = 16 / sizeof(T);
    using Packed = Pack<T, PACK_SIZE>;

    startSync<NRANKS>(signals, self, rank);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < packed_count;
         index += gridDim.x * blockDim.x) {
        Packed result;
#pragma unroll
        for (int item = 0; item < PACK_SIZE; ++item) {
            if constexpr (std::is_same_v<T, int32_t>) {
                int32_t sum = 0;
#pragma unroll
                for (int peer = 0; peer < NRANKS; ++peer) {
                    sum += reinterpret_cast<const Packed *>(
                               rank_data.ptrs[peer])[index]
                               .values[item];
                }
                result.values[item] = sum;
            } else {
                float sum = 0.0f;
#pragma unroll
                for (int peer = 0; peer < NRANKS; ++peer) {
                    sum += toFloat(
                        reinterpret_cast<const Packed *>(
                            rank_data.ptrs[peer])[index]
                            .values[item]);
                }
                result.values[item] = fromFloat<T>(sum);
            }
        }
        reinterpret_cast<Packed *>(output)[index] = result;
    }
    endSync<NRANKS>(signals, self, rank);
}

template <typename T>
bool launch(
    State *state,
    void *sendbuf,
    void *recvbuf,
    size_t count,
    cudaStream_t stream) {
    constexpr size_t PACK_SIZE = 16 / sizeof(T);
    const size_t bytes = count * sizeof(T);
    const size_t packed_count = count / PACK_SIZE;
    if (count % PACK_SIZE != 0 || bytes > MAX_BUFFER_BYTES) {
        return false;
    }

    const int blocks = std::min<int>(
        MAX_BLOCKS,
        static_cast<int>(
            (packed_count + THREADS - 1) / THREADS));
    if (blocks <= 0) {
        return false;
    }

    if (cudaMemcpyAsync(
            state->local_buffer,
            sendbuf,
            bytes,
            cudaMemcpyDeviceToDevice,
            stream)
        != cudaSuccess) {
        return false;
    }
    if (cudaEventRecord(state->release_event, stream) != cudaSuccess) {
        return false;
    }

    customAllReduceKernel<T, MAX_RANKS>
        <<<blocks, THREADS, 0, stream>>>(
            state->buffers,
            state->signals,
            reinterpret_cast<Signal *>(state->allocation),
            reinterpret_cast<T *>(recvbuf),
            state->rank,
            static_cast<int>(packed_count));
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace

void customAllReduceInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {
    if (!enabledByEnvironment() || ndevice != MAX_RANKS) {
        return;
    }

    std::vector<State *> pending(ndevice, nullptr);
    bool ok = true;

    for (int rank = 0; rank < ndevice && ok; ++rank) {
        auto *state = new State{
            rank, ndevice, device_ids[rank], nullptr, nullptr, {}, {}, nullptr, {}};
        pending[rank] = state;
        ok = cudaSetDevice(state->device_id) == cudaSuccess;
        if (ok) {
            ok = hipExtMallocWithFlags(
                     &state->allocation,
                     sizeof(Signal) + MAX_BUFFER_BYTES,
                     3u)
              == cudaSuccess;
        }
        if (ok) {
            state->local_buffer = static_cast<char *>(state->allocation) + sizeof(Signal);
            ok = cudaMemset(
                     state->allocation,
                     0,
                     sizeof(Signal) + MAX_BUFFER_BYTES)
              == cudaSuccess;
        }
        if (ok) {
            ok = cudaEventCreateWithFlags(
                     &state->release_event,
                     cudaEventDisableTiming)
              == cudaSuccess;
        }
    }

    for (int rank = 0; rank < ndevice && ok; ++rank) {
        State *state = pending[rank];
        ok = cudaSetDevice(state->device_id) == cudaSuccess;
        for (int peer = 0; peer < ndevice && ok; ++peer) {
            if (peer != rank) {
                int can_access_peer = 0;
                ok = cudaDeviceCanAccessPeer(
                         &can_access_peer,
                         state->device_id,
                         device_ids[peer])
                      == cudaSuccess
                  && can_access_peer != 0;
                if (ok) {
                    const cudaError_t status = cudaDeviceEnablePeerAccess(device_ids[peer], 0);
                    ok = status == cudaSuccess
                      || status == cudaErrorPeerAccessAlreadyEnabled;
                    if (status == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError();
                    }
                }
            }
            if (ok) {
                void *base = pending[peer]->allocation;
                state->signals.signals[peer] = reinterpret_cast<Signal *>(base);
                state->buffers.ptrs[peer] = static_cast<char *>(base) + sizeof(Signal);
            }
        }
    }

    if (!ok) {
        const cudaError_t error = cudaGetLastError();
        std::cerr
            << "Hygon custom all-reduce initialization failed; "
               "falling back to RCCL. Last CUDA error: "
            << cudaGetErrorName(error) << " (" << cudaGetErrorString(error) << ")."
            << std::endl;
        for (State *state : pending) {
            cleanupState(state);
        }
        return;
    }

    std::lock_guard<std::mutex> guard(state_mutex);
    for (int rank = 0; rank < ndevice; ++rank) {
        states[comms[rank]] = pending[rank];
    }
}

void customAllReduceDestroy(infinicclComm_t comm) {
    State *state = nullptr;
    {
        std::lock_guard<std::mutex> guard(state_mutex);
        const auto it = states.find(comm);
        if (it == states.end()) {
            return;
        }
        state = it->second;
        states.erase(it);
    }
    cleanupState(state);
}

bool customAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {
    if (op != INFINICCL_SUM || sendbuf == nullptr || recvbuf == nullptr
        || stream == nullptr) {
        return false;
    }

    State *state = nullptr;
    {
        std::lock_guard<std::mutex> guard(state_mutex);
        const auto it = states.find(comm);
        if (it == states.end()) {
            return false;
        }
        state = it->second;
    }

    if (cudaSetDevice(state->device_id) != cudaSuccess) {
        return false;
    }
    const cudaStream_t hip_stream = static_cast<cudaStream_t>(stream);
    switch (datatype) {
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(
            state, sendbuf, recvbuf, count, hip_stream);
    case INFINI_DTYPE_F32:
        return launch<float>(
            state, sendbuf, recvbuf, count, hip_stream);
    case INFINI_DTYPE_I32:
        return launch<int32_t>(
            state, sendbuf, recvbuf, count, hip_stream);
    default:
        return false;
    }
}

} // namespace infiniccl::hygon
