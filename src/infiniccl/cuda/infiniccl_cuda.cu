#include "infiniccl_cuda.h"

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>

#if defined(ENABLE_HYGON_API)
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <cuda.h>
#include <cuda_bf16.h>
#endif

#include "../../utils.h"

#define CHECK_NCCL(API__) CHECK_INTERNAL(API__, ncclSuccess)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream);
}

inline ncclDataType_t getNcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_I32:
        return ncclInt32;
    case INFINI_DTYPE_I64:
        return ncclInt64;
    case INFINI_DTYPE_U32:
        return ncclUint32;
    case INFINI_DTYPE_U64:
        return ncclUint64;
    case INFINI_DTYPE_F32:
        return ncclFloat;
    case INFINI_DTYPE_F16:
        return ncclHalf;
    case INFINI_DTYPE_BF16:
        return ncclBfloat16;
    default:
        std::abort();
        return ncclHalf;
    }
}

inline ncclRedOp_t getNcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return ncclSum;
    case INFINICCL_PROD:
        return ncclProd;
    case INFINICCL_MAX:
        return ncclMax;
    case INFINICCL_MIN:
        return ncclMin;
    case INFINICCL_AVG:
        return ncclAvg;
    default:
        std::abort();
        return ncclSum;
    }
}

inline ncclComm_t getNcclComm(infinicclComm_t comm) {
    return static_cast<ncclComm_t>(comm->comm);
}

namespace infiniccl::cuda {

infiniStatus_t getCommName(
    infinicclComm_t comm,
    char *comm_name,
    size_t comm_name_size) {
    return getCommNameFromHandle(comm, comm_name, comm_name_size);
}

#if defined(ENABLE_HYGON_API)
namespace {

constexpr int kHygonTp2MaxBlocks = 80;
constexpr size_t kHygonTp2StageCapacityElements = 1u << 22;
constexpr int kHygonTp8WorldSize = 8;
constexpr int kHygonTp8Threads = 512;
constexpr int kHygonTp8MaxBlocks = 80;
constexpr int kHygonTp8OneStageMaxBlocks = 16;
constexpr size_t kHygonTp8OneStageMaxBytes = 80u * 1024u;
constexpr size_t kHygonTp8TwoStageMaxBytes = 512u * 1024u;
constexpr int kHygonHipSuccess = 0;
constexpr unsigned int kHygonHipDeviceMallocUncached = 0x3;
constexpr unsigned int kHygonHipEventDisableTiming = 0x2;
constexpr unsigned int kHygonHipEventReleaseToSystem = 0x80000000u;

struct HygonTp2Signal {
    alignas(128) uint32_t start[kHygonTp2MaxBlocks][8];
    alignas(128) uint32_t end[kHygonTp2MaxBlocks][8];
    alignas(128) uint32_t flag[kHygonTp2MaxBlocks];
};

struct alignas(16) HygonTp2RankData {
    const void *ptrs[2];
};

struct alignas(16) HygonTp2RankSignals {
    HygonTp2Signal *signals[2];
};

struct alignas(16) HygonBf16Pack {
    __nv_bfloat16 values[8];
};

static_assert(sizeof(HygonBf16Pack) == 16);

struct HygonTp8Signal {
    alignas(128) uint32_t start[kHygonTp8MaxBlocks][kHygonTp8WorldSize];
    alignas(128) uint32_t end[kHygonTp8MaxBlocks][kHygonTp8WorldSize];
    alignas(128) uint32_t flag[kHygonTp8MaxBlocks];
};

constexpr size_t kHygonTp8TwoStageScratchBytes = kHygonTp8TwoStageMaxBytes / kHygonTp8WorldSize + (kHygonTp8WorldSize - 1) * sizeof(HygonBf16Pack);
constexpr size_t kHygonTp8SignalAllocationBytes = sizeof(HygonTp8Signal) + kHygonTp8TwoStageScratchBytes;

struct alignas(16) HygonTp8RankData {
    const void *ptrs[kHygonTp8WorldSize];
};

struct alignas(16) HygonTp8RankSignals {
    HygonTp8Signal *signals[kHygonTp8WorldSize];
};

static_assert(sizeof(HygonTp8RankData) == 64 && alignof(HygonTp8RankData) == 16);
static_assert(sizeof(HygonTp8RankSignals) == 64 && alignof(HygonTp8RankSignals) == 16);

struct HygonCudaDriverApi {
    void *library = nullptr;
    decltype(&cuMemGetAllocationGranularity) mem_get_allocation_granularity = nullptr;
    decltype(&cuMemAddressReserve) mem_address_reserve = nullptr;
    decltype(&cuMemCreate) mem_create = nullptr;
    decltype(&cuMemMap) mem_map = nullptr;
    decltype(&cuMemSetAccess) mem_set_access = nullptr;
    decltype(&cuMemUnmap) mem_unmap = nullptr;
    decltype(&cuMemRelease) mem_release = nullptr;
    decltype(&cuMemAddressFree) mem_address_free = nullptr;
    bool available = false;

    HygonCudaDriverApi() {
        library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            library = dlopen("/opt/dtk/cuda/cuda/lib64/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        }
        available = library != nullptr && load(mem_get_allocation_granularity, "cuMemGetAllocationGranularity") && load(mem_address_reserve, "cuMemAddressReserve") && load(mem_create, "cuMemCreate") && load(mem_map, "cuMemMap") && load(mem_set_access, "cuMemSetAccess") && load(mem_unmap, "cuMemUnmap") && load(mem_release, "cuMemRelease") && load(mem_address_free, "cuMemAddressFree");
    }

private:
    template <typename T>
    bool load(T &symbol, const char *name) {
        symbol = reinterpret_cast<T>(dlsym(library, name));
        return symbol != nullptr;
    }
};

HygonCudaDriverApi &hygon_cuda_driver_api() {
    static HygonCudaDriverApi api;
    return api;
}

struct HygonHipExtApi {
    void *library = nullptr;
    int (*ext_malloc_with_flags)(void **, size_t, unsigned int) = nullptr;
    int (*memset)(void *, int, size_t) = nullptr;
    int (*free)(void *) = nullptr;
    int (*event_create_with_flags)(void **, unsigned int) = nullptr;
    int (*event_destroy)(void *) = nullptr;
    int (*ext_launch_kernel)(
        const void *, dim3, dim3, void **, size_t,
        void *, void *, void *, int)
        = nullptr;
    bool available = false;

    HygonHipExtApi() {
        constexpr const char *candidates[] = {
            "libgalaxyhip.so.5",
            "libgalaxyhip.so",
            "/opt/dtk/hip/lib/libgalaxyhip.so.5",
            "/opt/dtk/lib/libgalaxyhip.so.5",
        };
        for (const char *candidate : candidates) {
            library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            if (library != nullptr) {
                break;
            }
        }
        available = library != nullptr && load(ext_malloc_with_flags, "hipExtMallocWithFlags") && load(memset, "hipMemset") && load(free, "hipFree") && load(event_create_with_flags, "hipEventCreateWithFlags") && load(event_destroy, "hipEventDestroy") && load(ext_launch_kernel, "hipExtLaunchKernel");
    }

private:
    template <typename T>
    bool load(T &symbol, const char *name) {
        symbol = reinterpret_cast<T>(dlsym(library, name));
        return symbol != nullptr;
    }
};

HygonHipExtApi &hygon_hip_ext_api() {
    static HygonHipExtApi *api = new HygonHipExtApi();
    return *api;
}

struct HygonVmmAllocation {
    void *ptr = nullptr;
    size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool hip_uncached = false;
};

struct HygonTp2AllReduceState {
    int device_ids[2] = {0, 1};
    HygonVmmAllocation stages[2];
    HygonTp2Signal *signal_hosts[2] = {nullptr, nullptr};
    HygonTp2Signal *signals[2] = {nullptr, nullptr};
    HygonTp2RankData *rank_data[2] = {nullptr, nullptr};
    HygonTp2RankSignals rank_signals{};
    struct CaptureCursor {
        unsigned long long id = 0;
        size_t next_element = 0;
        bool initialized = false;
    };
    std::mutex capture_mutex;
    CaptureCursor capture_cursors[2];

    ~HygonTp2AllReduceState() {
        int previous_device = 0;
        const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
        for (int rank = 0; rank < 2; ++rank) {
            cudaSetDevice(device_ids[rank]);
            if (rank_data[rank] != nullptr) {
                cudaFree(rank_data[rank]);
            }
            if (signal_hosts[rank] != nullptr) {
                cudaFreeHost(signal_hosts[rank]);
            }
            auto &driver = hygon_cuda_driver_api();
            if (driver.available && stages[rank].handle != 0) {
                const auto address = reinterpret_cast<CUdeviceptr>(stages[rank].ptr);
                driver.mem_unmap(address, stages[rank].size);
                driver.mem_address_free(address, stages[rank].size);
                driver.mem_release(stages[rank].handle);
            } else if (stages[rank].hip_uncached && stages[rank].ptr != nullptr) {
                auto &hip = hygon_hip_ext_api();
                if (hip.available) {
                    hip.free(stages[rank].ptr);
                }
            } else if (stages[rank].ptr != nullptr) {
                cudaFree(stages[rank].ptr);
            }
        }
        if (restore_device) {
            cudaSetDevice(previous_device);
        }
    }
};

std::mutex hygon_tp2_states_mutex;
std::unordered_map<infinicclComm_t, std::shared_ptr<HygonTp2AllReduceState>> hygon_tp2_states;

struct HygonTp8AllReduceState {
    int device_ids[kHygonTp8WorldSize]{};
    HygonTp8Signal *signals[kHygonTp8WorldSize]{};
    void *release_events[kHygonTp8WorldSize]{};
    HygonTp8RankSignals rank_signals{};

    struct CaptureRendezvous {
        std::mutex mutex;
        std::condition_variable condition;
        uint64_t generation = 0;
        uint32_t arrived_mask = 0;
        int departed = 0;
        bool ready = false;
        bool use_custom = false;
        bool metadata_error = false;
        const void *sendbufs[kHygonTp8WorldSize]{};
        void *recvbufs[kHygonTp8WorldSize]{};
        size_t counts[kHygonTp8WorldSize]{};
        infiniDtype_t datatypes[kHygonTp8WorldSize]{};
        infinicclReduceOp_t ops[kHygonTp8WorldSize]{};
        bool eligible[kHygonTp8WorldSize]{};
    } rendezvous;

    ~HygonTp8AllReduceState() {
        int previous_device = 0;
        const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
        auto &hip = hygon_hip_ext_api();
        for (int rank = 0; rank < kHygonTp8WorldSize; ++rank) {
            cudaSetDevice(device_ids[rank]);
            if (hip.available && release_events[rank] != nullptr) {
                hip.event_destroy(release_events[rank]);
            }
            if (hip.available && signals[rank] != nullptr) {
                hip.free(signals[rank]);
            }
        }
        if (restore_device) {
            cudaSetDevice(previous_device);
        }
    }
};

std::mutex hygon_tp8_states_mutex;
std::unordered_map<infinicclComm_t, std::shared_ptr<HygonTp8AllReduceState>> hygon_tp8_states;

bool allocate_hygon_vmm(HygonVmmAllocation &allocation,
                        int owner_device,
                        const int device_ids[2],
                        size_t requested_size) {
    auto &driver = hygon_cuda_driver_api();
    if (!driver.available) {
        return false;
    }
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = owner_device;
    size_t granularity = 0;
    if (driver.mem_get_allocation_granularity(
            &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM)
            != CUDA_SUCCESS
        || granularity == 0) {
        return false;
    }
    allocation.size = (requested_size + granularity - 1) / granularity * granularity;
    CUdeviceptr address = 0;
    if (driver.mem_address_reserve(
            &address, allocation.size, granularity, 0, 0)
        != CUDA_SUCCESS) {
        allocation = {};
        return false;
    }
    allocation.ptr = reinterpret_cast<void *>(address);
    if (driver.mem_create(
            &allocation.handle, allocation.size, &properties, 0)
        != CUDA_SUCCESS) {
        driver.mem_address_free(address, allocation.size);
        allocation = {};
        return false;
    }
    if (driver.mem_map(
            address, allocation.size, 0, allocation.handle, 0)
        != CUDA_SUCCESS) {
        driver.mem_release(allocation.handle);
        driver.mem_address_free(address, allocation.size);
        allocation = {};
        return false;
    }
    CUmemAccessDesc access[2]{};
    for (int rank = 0; rank < 2; ++rank) {
        access[rank].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[rank].location.id = device_ids[rank];
        access[rank].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    if (driver.mem_set_access(address, allocation.size, access, 2) != CUDA_SUCCESS) {
        driver.mem_unmap(address, allocation.size);
        driver.mem_release(allocation.handle);
        driver.mem_address_free(address, allocation.size);
        allocation = {};
        return false;
    }
    return true;
}

template <int NumRanks>
__device__ __forceinline__ uint32_t hygon_tp2_start_sync(
    const HygonTp2RankSignals &rank_signals,
    HygonTp2Signal *self_signal,
    int rank) {
    const uint32_t next_flag = self_signal->flag[blockIdx.x] + 1;
    if (threadIdx.x < NumRanks) {
        __scoped_atomic_store_n(
            &rank_signals.signals[threadIdx.x]->start[blockIdx.x][rank],
            next_flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self_signal->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
               < next_flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self_signal->flag[blockIdx.x] = next_flag;
    }
    return next_flag;
}

template <int NumRanks>
__device__ __forceinline__ void hygon_tp2_end_sync(
    const HygonTp2RankSignals &rank_signals,
    HygonTp2Signal *self_signal,
    int rank,
    uint32_t flag) {
    __syncthreads();
    if (threadIdx.x < NumRanks) {
        __scoped_atomic_store_n(
            &rank_signals.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self_signal->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)
               < flag) {
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void hygon_tp2_bf16_allreduce_kernel(
    const HygonTp2RankData *rank_data,
    HygonTp2RankSignals rank_signals,
    HygonTp2Signal *self_signal,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    int rank,
    size_t pack_count,
    size_t pack_offset) {
    constexpr int num_ranks = 2;
    constexpr int threads_per_rank = 512 / num_ranks;
    constexpr int pack_size = 8;
    __shared__ __nv_bfloat16 shared[threads_per_rank * num_ranks * pack_size];
    const HygonTp2RankData data = *rank_data;
    const int source_rank = threadIdx.x / threads_per_rank;
    const int lane = threadIdx.x % threads_per_rank;
    if (threadIdx.x < threads_per_rank) {
        auto *local_stage = reinterpret_cast<HygonBf16Pack *>(
            const_cast<void *>(data.ptrs[rank]));
        const auto *local_input = reinterpret_cast<const HygonBf16Pack *>(input);
        for (size_t index = blockIdx.x * threads_per_rank + threadIdx.x;
             index < pack_count;
             index += gridDim.x * threads_per_rank) {
            local_stage[pack_offset + index] = local_input[index];
        }
        __threadfence_system();
    }
    __syncthreads();
    const uint32_t sync_flag = hygon_tp2_start_sync<num_ranks>(rank_signals, self_signal, rank);
    for (size_t index = blockIdx.x * threads_per_rank + lane;
         index < pack_count;
         index += gridDim.x * threads_per_rank) {
        auto *shared_packs = reinterpret_cast<HygonBf16Pack *>(shared);
        const auto *source = reinterpret_cast<const HygonBf16Pack *>(data.ptrs[source_rank]);
        shared_packs[threadIdx.x] = source[pack_offset + index];
        __syncthreads();
        if (source_rank == 0) {
            HygonBf16Pack reduced;
#pragma unroll
            for (int element = 0; element < pack_size; ++element) {
                const float value = __bfloat162float(shared[threadIdx.x * pack_size + element]) + __bfloat162float(shared[(threads_per_rank + threadIdx.x) * pack_size + element]);
                reduced.values[element] = __float2bfloat16(value);
            }
            reinterpret_cast<HygonBf16Pack *>(output)[index] = reduced;
        }
        __syncthreads();
    }
    hygon_tp2_end_sync<num_ranks>(
        rank_signals, self_signal, rank, sync_flag);
}

std::shared_ptr<HygonTp2AllReduceState> create_hygon_tp2_state(
    int ndevice, const int *device_ids) {
    if (ndevice != 2 || device_ids == nullptr) {
        return nullptr;
    }
    auto state = std::make_shared<HygonTp2AllReduceState>();
    state->device_ids[0] = device_ids[0];
    state->device_ids[1] = device_ids[1];
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    auto fail = [&]() -> std::shared_ptr<HygonTp2AllReduceState> {
        if (restore_device) {
            cudaSetDevice(previous_device);
        }
        return nullptr;
    };
    const size_t stage_bytes = kHygonTp2StageCapacityElements * sizeof(__nv_bfloat16);
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(device_ids[rank]) != cudaSuccess) {
            return fail();
        }
        const int peer = 1 - rank;
        int can_access_peer = 0;
        if (cudaDeviceCanAccessPeer(
                &can_access_peer,
                device_ids[rank],
                device_ids[peer])
                != cudaSuccess
            || can_access_peer == 0) {
            return fail();
        }
        const cudaError_t enable_status = cudaDeviceEnablePeerAccess(device_ids[peer], 0);
        if (enable_status == cudaErrorPeerAccessAlreadyEnabled) {
            (void)cudaGetLastError();
        } else if (enable_status != cudaSuccess) {
            return fail();
        }
        auto &hip = hygon_hip_ext_api();
        if (!hip.available) {
            return fail();
        }
        state->stages[rank].size = stage_bytes;
        if (hip.ext_malloc_with_flags(
                &state->stages[rank].ptr,
                stage_bytes,
                kHygonHipDeviceMallocUncached)
            != kHygonHipSuccess) {
            return fail();
        }
        state->stages[rank].hip_uncached = true;
        if (hip.memset(
                state->stages[rank].ptr,
                0,
                stage_bytes)
            != kHygonHipSuccess) {
            return fail();
        }
        void *signal_host = nullptr;
        if (cudaHostAlloc(&signal_host, sizeof(HygonTp2Signal),
                          cudaHostAllocMapped)
            != cudaSuccess) {
            return fail();
        }
        state->signal_hosts[rank] = static_cast<HygonTp2Signal *>(signal_host);
        std::memset(state->signal_hosts[rank], 0, sizeof(HygonTp2Signal));
        void *signal_device = nullptr;
        if (cudaHostGetDevicePointer(&signal_device, signal_host, 0) != cudaSuccess) {
            return fail();
        }
        state->signals[rank] = static_cast<HygonTp2Signal *>(signal_device);
        if (cudaMalloc(reinterpret_cast<void **>(&state->rank_data[rank]),
                       sizeof(HygonTp2RankData))
            != cudaSuccess) {
            return fail();
        }
    }
    HygonTp2RankData host_rank_data{{state->stages[0].ptr, state->stages[1].ptr}};
    state->rank_signals.signals[0] = state->signals[0];
    state->rank_signals.signals[1] = state->signals[1];
    for (int rank = 0; rank < 2; ++rank) {
        cudaSetDevice(device_ids[rank]);
        if (cudaMemcpy(state->rank_data[rank], &host_rank_data,
                       sizeof(host_rank_data), cudaMemcpyHostToDevice)
            != cudaSuccess) {
            return fail();
        }
    }
    if (restore_device) {
        cudaSetDevice(previous_device);
    }
    return state;
}

void register_hygon_tp2_state(infinicclComm_t *comms,
                              int ndevice,
                              const int *device_ids) {
    auto state = create_hygon_tp2_state(ndevice, device_ids);
    if (state == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(hygon_tp2_states_mutex);
    for (int rank = 0; rank < 2; ++rank) {
        hygon_tp2_states.emplace(comms[rank], state);
    }
}

void erase_hygon_tp2_state(infinicclComm_t comm) {
    std::lock_guard<std::mutex> lock(hygon_tp2_states_mutex);
    hygon_tp2_states.erase(comm);
}

std::shared_ptr<HygonTp2AllReduceState> get_hygon_tp2_state(infinicclComm_t comm) {
    std::lock_guard<std::mutex> lock(hygon_tp2_states_mutex);
    auto found = hygon_tp2_states.find(comm);
    return found == hygon_tp2_states.end() ? nullptr : found->second;
}

bool reserve_hygon_tp2_stage(
    const std::shared_ptr<HygonTp2AllReduceState> &state,
    int rank,
    unsigned long long capture_id,
    size_t count,
    size_t *element_offset) {
    std::lock_guard<std::mutex> lock(state->capture_mutex);
    auto &cursor = state->capture_cursors[rank];
    if (!cursor.initialized || cursor.id != capture_id) {
        cursor.id = capture_id;
        cursor.next_element = 0;
        cursor.initialized = true;
    }
    const size_t aligned_offset = (cursor.next_element + 7) & ~size_t{7};
    if (aligned_offset > kHygonTp2StageCapacityElements - count) {
        return false;
    }
    *element_offset = aligned_offset;
    cursor.next_element = aligned_offset + count;
    return true;
}

bool try_hygon_tp2_graph_allreduce(
    void *sendbuf, void *recvbuf, size_t count,
    infiniDtype_t datatype, infinicclReduceOp_t op,
    infinicclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || comm->world_size != 2 || datatype != INFINI_DTYPE_BF16 || op != INFINICCL_SUM || count == 0 || count > kHygonTp2StageCapacityElements || (count % 8) != 0) {
        return false;
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id) != cudaSuccess || capture_status != cudaStreamCaptureStatusActive) {
        return false;
    }
    auto state = get_hygon_tp2_state(comm);
    if (state == nullptr || comm->rank < 0 || comm->rank >= 2) {
        return false;
    }
    const int rank = comm->rank;
    size_t element_offset = 0;
    if (!reserve_hygon_tp2_stage(state, rank, capture_id, count, &element_offset)) {
        return false;
    }
    const size_t pack_count = count / 8;
    int blocks = static_cast<int>(
        std::min<size_t>(kHygonTp2MaxBlocks, (pack_count + 255) / 256));
    blocks = std::max(blocks, 1);
    hygon_tp2_bf16_allreduce_kernel<<<blocks, 512, 0, stream>>>(
        state->rank_data[rank], state->rank_signals, state->signals[rank],
        static_cast<const __nv_bfloat16 *>(sendbuf),
        static_cast<__nv_bfloat16 *>(recvbuf), rank, pack_count, element_offset / 8);
    return cudaGetLastError() == cudaSuccess;
}

template <int NumRanks>
__device__ __forceinline__ uint32_t hygon_tp8_start_sync(
    const HygonTp8RankSignals &rank_signals,
    HygonTp8Signal *self_signal,
    int rank,
    uint32_t *block_flag) {
    if (threadIdx.x == 0) {
        *block_flag = __scoped_atomic_load_n(
                          &self_signal->flag[blockIdx.x],
                          __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM)
                    + 1;
    }
    __syncthreads();
    const uint32_t next_flag = *block_flag;
    if (threadIdx.x < NumRanks) {
        __scoped_atomic_store_n(
            &rank_signals.signals[threadIdx.x]->start[blockIdx.x][rank],
            next_flag, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self_signal->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM)
               != next_flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __scoped_atomic_store_n(
            &self_signal->flag[blockIdx.x], next_flag,
            __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    return next_flag;
}

template <int NumRanks>
__device__ __forceinline__ void hygon_tp8_end_sync(
    const HygonTp8RankSignals &rank_signals,
    HygonTp8Signal *self_signal,
    int rank,
    uint32_t flag) {
    __syncthreads();
    if (threadIdx.x < NumRanks) {
        __scoped_atomic_store_n(
            &rank_signals.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        while (__scoped_atomic_load_n(
                   &self_signal->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM)
               != flag) {
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(kHygonTp8Threads, 1) void hygon_tp8_bf16_allreduce_kernel(
    HygonTp8RankData rank_data,
    HygonTp8RankSignals rank_signals,
    HygonTp8Signal *self_signal,
    __nv_bfloat16 *output,
    int rank,
    size_t pack_count) {
    constexpr int num_ranks = kHygonTp8WorldSize;
    __shared__ uint32_t block_flag;

    const uint32_t sync_flag = hygon_tp8_start_sync<num_ranks>(
        rank_signals, self_signal, rank, &block_flag);

    const size_t thread_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t thread_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    auto *packed_output = reinterpret_cast<HygonBf16Pack *>(output);
    for (size_t index = thread_index;
         index < pack_count;
         index += thread_stride) {
        const auto *source = reinterpret_cast<const HygonBf16Pack *>(
            rank_data.ptrs[0]);
        HygonBf16Pack source_pack = source[index];
        float reduced[8];
#pragma unroll
        for (int element = 0; element < 8; ++element) {
            reduced[element] = __bfloat162float(source_pack.values[element]);
        }
#pragma unroll
        for (int peer = 1; peer < num_ranks; ++peer) {
            source = reinterpret_cast<const HygonBf16Pack *>(
                rank_data.ptrs[peer]);
            source_pack = source[index];
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                reduced[element] += __bfloat162float(source_pack.values[element]);
            }
        }
        HygonBf16Pack result;
#pragma unroll
        for (int element = 0; element < 8; ++element) {
            result.values[element] = __float2bfloat16(reduced[element]);
        }
        packed_output[index] = result;
    }

    hygon_tp8_end_sync<num_ranks>(
        rank_signals, self_signal, rank, sync_flag);
}

__device__ __forceinline__ HygonBf16Pack *hygon_tp8_two_stage_scratch(
    HygonTp8Signal *signal) {
    return reinterpret_cast<HygonBf16Pack *>(signal + 1);
}

__global__ __launch_bounds__(kHygonTp8Threads, 1) void hygon_tp8_bf16_allreduce_2stage_kernel(
    HygonTp8RankData rank_data,
    HygonTp8RankSignals rank_signals,
    HygonTp8Signal *self_signal,
    __nv_bfloat16 *output,
    int rank,
    size_t pack_count) {
    constexpr int num_ranks = kHygonTp8WorldSize;
    constexpr int threads_per_rank = kHygonTp8Threads / num_ranks;
    __shared__ HygonBf16Pack shared_packs[kHygonTp8Threads];
    __shared__ uint32_t block_flag;

    const int source_rank = threadIdx.x / threads_per_rank;
    const int lane = threadIdx.x % threads_per_rank;
    const size_t thread_index = blockIdx.x * threads_per_rank + lane;
    const size_t thread_stride = gridDim.x * threads_per_rank;
    const size_t part = pack_count / num_ranks;
    const size_t remainder = pack_count % num_ranks;
    const size_t slice_begin = static_cast<size_t>(rank) * part;
    const size_t slice_end = rank == num_ranks - 1
                               ? pack_count
                               : slice_begin + part;
    const size_t largest_part = part + remainder;
    HygonBf16Pack *local_scratch = hygon_tp8_two_stage_scratch(self_signal);

    const uint32_t sync_flag = hygon_tp8_start_sync<num_ranks>(
        rank_signals, self_signal, rank, &block_flag);

    // Stage 1: each rank reduces one disjoint slice into its peer-visible
    // uncached scratch buffer.
    for (size_t base = slice_begin + blockIdx.x * threads_per_rank;
         base < slice_end;
         base += gridDim.x * threads_per_rank) {
        const size_t index = base + lane;
        HygonBf16Pack source_pack{};
        if (index < slice_end) {
            const auto *source = reinterpret_cast<const HygonBf16Pack *>(
                rank_data.ptrs[source_rank]);
            source_pack = source[index];
        }
        shared_packs[threadIdx.x] = source_pack;
        __syncthreads();

        if (source_rank == 0 && index < slice_end) {
            float reduced[8];
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                reduced[element] = __bfloat162float(
                    shared_packs[threadIdx.x].values[element]);
            }
#pragma unroll
            for (int peer = 1; peer < num_ranks; ++peer) {
#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    reduced[element] += __bfloat162float(
                        shared_packs[peer * threads_per_rank + threadIdx.x]
                            .values[element]);
                }
            }
            HygonBf16Pack result;
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                result.values[element] = __float2bfloat16(reduced[element]);
            }
            local_scratch[index - slice_begin] = result;
        }
        __syncthreads();
    }

    // Release makes the reduced slices visible before peers gather them.
    hygon_tp8_end_sync<num_ranks>(
        rank_signals, self_signal, rank, sync_flag);

    // Stage 2: the eight thread groups gather one source rank each.
    const HygonBf16Pack *source_scratch = hygon_tp8_two_stage_scratch(rank_signals.signals[source_rank]);
    auto *packed_output = reinterpret_cast<HygonBf16Pack *>(output);
    for (size_t offset = thread_index;
         offset < largest_part;
         offset += thread_stride) {
        if (source_rank == num_ranks - 1 || offset < part) {
            packed_output[static_cast<size_t>(source_rank) * part + offset] = source_scratch[offset];
        }
    }
    return;
}

std::shared_ptr<HygonTp8AllReduceState> create_hygon_tp8_state(
    int ndevice, const int *device_ids) {
    if (ndevice != kHygonTp8WorldSize || device_ids == nullptr) {
        return nullptr;
    }
    auto &hip = hygon_hip_ext_api();
    if (!hip.available) {
        return nullptr;
    }
    auto state = std::make_shared<HygonTp8AllReduceState>();
    std::memcpy(state->device_ids, device_ids, sizeof(state->device_ids));

    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    auto fail = [&]() -> std::shared_ptr<HygonTp8AllReduceState> {
        if (restore_device) {
            cudaSetDevice(previous_device);
        }
        return nullptr;
    };

    for (int rank = 0; rank < kHygonTp8WorldSize; ++rank) {
        if (cudaSetDevice(device_ids[rank]) != cudaSuccess) {
            return fail();
        }
        for (int peer = 0; peer < kHygonTp8WorldSize; ++peer) {
            if (peer == rank) {
                continue;
            }
            int can_access = 0;
            if (cudaDeviceCanAccessPeer(
                    &can_access, device_ids[rank], device_ids[peer])
                    != cudaSuccess
                || can_access == 0) {
                return fail();
            }
            const cudaError_t enable_status = cudaDeviceEnablePeerAccess(device_ids[peer], 0);
            if (enable_status == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
            } else if (enable_status != cudaSuccess) {
                return fail();
            }
        }
        if (hip.ext_malloc_with_flags(
                reinterpret_cast<void **>(&state->signals[rank]),
                kHygonTp8SignalAllocationBytes,
                kHygonHipDeviceMallocUncached)
                != kHygonHipSuccess
            || hip.memset(state->signals[rank], 0,
                          kHygonTp8SignalAllocationBytes)
                   != kHygonHipSuccess) {
            return fail();
        }
        if (hip.event_create_with_flags(
                &state->release_events[rank],
                kHygonHipEventReleaseToSystem | kHygonHipEventDisableTiming)
            != kHygonHipSuccess) {
            return fail();
        }
        state->rank_signals.signals[rank] = state->signals[rank];
    }
    if (restore_device) {
        cudaSetDevice(previous_device);
    }
    return state;
}

void register_hygon_tp8_state(infinicclComm_t *comms,
                              int ndevice,
                              const int *device_ids) {
    auto state = create_hygon_tp8_state(ndevice, device_ids);
    if (state == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(hygon_tp8_states_mutex);
    for (int rank = 0; rank < kHygonTp8WorldSize; ++rank) {
        hygon_tp8_states.emplace(comms[rank], state);
    }
}

void erase_hygon_tp8_state(infinicclComm_t comm) {
    std::lock_guard<std::mutex> lock(hygon_tp8_states_mutex);
    hygon_tp8_states.erase(comm);
}

std::shared_ptr<HygonTp8AllReduceState> get_hygon_tp8_state(infinicclComm_t comm) {
    std::lock_guard<std::mutex> lock(hygon_tp8_states_mutex);
    auto found = hygon_tp8_states.find(comm);
    return found == hygon_tp8_states.end() ? nullptr : found->second;
}

bool rendezvous_hygon_tp8_graph_inputs(
    const std::shared_ptr<HygonTp8AllReduceState> &state,
    int rank,
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    HygonTp8RankData *rank_data,
    bool *metadata_error) {
    const bool local_eligible = sendbuf != nullptr && recvbuf != nullptr && sendbuf != recvbuf && datatype == INFINI_DTYPE_BF16 && op == INFINICCL_SUM && count != 0 && (count % 8) == 0 && count <= kHygonTp8TwoStageMaxBytes / sizeof(__nv_bfloat16) && (reinterpret_cast<uintptr_t>(sendbuf) % alignof(HygonBf16Pack)) == 0 && (reinterpret_cast<uintptr_t>(recvbuf) % alignof(HygonBf16Pack)) == 0;

    auto &rendezvous = state->rendezvous;
    std::unique_lock<std::mutex> lock(rendezvous.mutex);
    const uint64_t generation = rendezvous.generation;
    const uint32_t rank_bit = uint32_t{1} << rank;
    if ((rendezvous.arrived_mask & rank_bit) != 0) {
        std::abort();
    }

    rendezvous.arrived_mask |= rank_bit;
    rendezvous.sendbufs[rank] = sendbuf;
    rendezvous.recvbufs[rank] = recvbuf;
    rendezvous.counts[rank] = count;
    rendezvous.datatypes[rank] = datatype;
    rendezvous.ops[rank] = op;
    rendezvous.eligible[rank] = local_eligible;

    constexpr uint32_t all_ranks_mask = (uint32_t{1} << kHygonTp8WorldSize) - 1;
    if (rendezvous.arrived_mask == all_ranks_mask) {
        bool signatures_match = true;
        bool use_custom = true;
        for (int peer = 0; peer < kHygonTp8WorldSize; ++peer) {
            signatures_match = signatures_match && rendezvous.counts[peer] == rendezvous.counts[0] && rendezvous.datatypes[peer] == rendezvous.datatypes[0] && rendezvous.ops[peer] == rendezvous.ops[0];
            use_custom = use_custom && rendezvous.eligible[peer];
        }
        rendezvous.metadata_error = !signatures_match;
        rendezvous.use_custom = signatures_match && use_custom;
        rendezvous.ready = true;
        rendezvous.condition.notify_all();
    } else {
        rendezvous.condition.wait(lock, [&] {
            return rendezvous.ready && rendezvous.generation == generation;
        });
    }

    const bool use_custom = rendezvous.use_custom;
    *metadata_error = rendezvous.metadata_error;
    if (use_custom) {
        for (int peer = 0; peer < kHygonTp8WorldSize; ++peer) {
            rank_data->ptrs[peer] = rendezvous.sendbufs[peer];
        }
    }

    if (++rendezvous.departed == kHygonTp8WorldSize) {
        rendezvous.arrived_mask = 0;
        rendezvous.departed = 0;
        rendezvous.ready = false;
        rendezvous.use_custom = false;
        rendezvous.metadata_error = false;
        ++rendezvous.generation;
        rendezvous.condition.notify_all();
    } else {
        rendezvous.condition.wait(lock, [&] {
            return rendezvous.generation != generation;
        });
    }
    return use_custom;
}

enum class HygonTp8AllReduceResult {
    Fallback,
    Success,
    Error,
};

HygonTp8AllReduceResult try_hygon_tp8_graph_allreduce(
    void *sendbuf, void *recvbuf, size_t count,
    infiniDtype_t datatype, infinicclReduceOp_t op,
    infinicclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || comm->world_size != kHygonTp8WorldSize || comm->rank < 0 || comm->rank >= kHygonTp8WorldSize) {
        return HygonTp8AllReduceResult::Fallback;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id) != cudaSuccess) {
        return HygonTp8AllReduceResult::Error;
    }
    (void)capture_id;
    if (capture_status != cudaStreamCaptureStatusActive) {
        return HygonTp8AllReduceResult::Fallback;
    }

    auto state = get_hygon_tp8_state(comm);
    if (state == nullptr) {
        return HygonTp8AllReduceResult::Fallback;
    }
    const int rank = comm->rank;
    HygonTp8RankData rank_data{};
    bool metadata_error = false;
    if (!rendezvous_hygon_tp8_graph_inputs(
            state, rank, sendbuf, recvbuf, count, datatype, op,
            &rank_data, &metadata_error)) {
        if (metadata_error) {
            return HygonTp8AllReduceResult::Error;
        }
        return HygonTp8AllReduceResult::Fallback;
    }

    size_t pack_count = count / 8;
    const bool use_two_stage = count * sizeof(__nv_bfloat16) >= kHygonTp8OneStageMaxBytes;
    const size_t work_pack_count = use_two_stage
                                     ? pack_count / kHygonTp8WorldSize + pack_count % kHygonTp8WorldSize
                                     : pack_count;
    const size_t work_threads = use_two_stage
                                  ? kHygonTp8Threads / kHygonTp8WorldSize
                                  : kHygonTp8Threads;
    const int max_blocks = use_two_stage ? kHygonTp8MaxBlocks : kHygonTp8OneStageMaxBlocks;
    int blocks = static_cast<int>(std::min<size_t>(
        max_blocks,
        (work_pack_count + work_threads - 1) / work_threads));
    blocks = std::max(blocks, 1);
    HygonTp8RankSignals rank_signals = state->rank_signals;
    HygonTp8Signal *self_signal = state->signals[rank];
    auto *output = static_cast<__nv_bfloat16 *>(recvbuf);
    int kernel_rank = rank;
    void *args[] = {
        &rank_data,
        &rank_signals,
        &self_signal,
        &output,
        &kernel_rank,
        &pack_count,
    };
    auto &hip = hygon_hip_ext_api();
    const void *kernel = use_two_stage
                           ? reinterpret_cast<const void *>(
                               hygon_tp8_bf16_allreduce_2stage_kernel)
                           : reinterpret_cast<const void *>(
                               hygon_tp8_bf16_allreduce_kernel);
    const int launch_status = hip.ext_launch_kernel(
        kernel,
        dim3(blocks), dim3(kHygonTp8Threads), args, 0,
        reinterpret_cast<void *>(stream), nullptr,
        state->release_events[rank], 0);
    return launch_status == kHygonHipSuccess
             ? HygonTp8AllReduceResult::Success
             : HygonTp8AllReduceResult::Error;
}

} // namespace
#endif

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<ncclComm_t> nccl_comms(ndevice);
    CHECK_NCCL(ncclCommInitAll(nccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_NVIDIA, device_ids[i], (void *)(nccl_comms[i]), i, ndevice};
    }

#if defined(ENABLE_HYGON_API)
    register_hygon_tp2_state(comms, ndevice, device_ids);
    register_hygon_tp8_state(comms, ndevice, device_ids);
#endif

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getUniqueId(infinicclUniqueId_t *unique_id) {
    if (unique_id == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    CHECK_NCCL(ncclGetUniqueId(reinterpret_cast<ncclUniqueId *>(unique_id)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commInitRank(
    infinicclComm_t *comm,
    int nranks,
    infinicclUniqueId_t comm_id,
    int rank) {
    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (nranks <= 0 || rank < 0 || rank >= nranks) {
        return INFINI_STATUS_BAD_PARAM;
    }

    infiniDevice_t device_type;
    int device_id;
    CHECK_STATUS(infinirtGetDevice(&device_type, &device_id));

    ncclUniqueId nccl_id;
    std::memcpy(&nccl_id, &comm_id, sizeof(nccl_id));
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, nranks, nccl_id, rank));
    *comm = new InfinicclComm{device_type, device_id, (void *)nccl_comm, rank, nranks};
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
#if defined(ENABLE_HYGON_API)
    erase_hygon_tp2_state(comm);
    erase_hygon_tp8_state(comm);
#endif
    CHECK_NCCL(ncclCommDestroy(getNcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t groupStart(infinicclComm_t) {
    CHECK_NCCL(ncclGroupStart());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t groupEnd(infinicclComm_t) {
    CHECK_NCCL(ncclGroupEnd());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16,
                INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U32, INFINI_DTYPE_U64);

#if defined(ENABLE_HYGON_API)
    const auto tp8_result = try_hygon_tp8_graph_allreduce(
        sendbuf, recvbuf, count, datatype, op, comm,
        getCudaStream(stream));
    if (tp8_result == HygonTp8AllReduceResult::Success) {
        return INFINI_STATUS_SUCCESS;
    }
    if (tp8_result == HygonTp8AllReduceResult::Error) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    if (try_hygon_tp2_graph_allreduce(
            sendbuf, recvbuf, count, datatype, op, comm,
            getCudaStream(stream))) {
        return INFINI_STATUS_SUCCESS;
    }
#endif

    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, getNcclDtype(datatype),
                             getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t broadcast(
    const void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    int root,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (sendbuf == nullptr || recvbuf == nullptr || comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (root < 0 || root >= comm->world_size || count == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclBroadcast(sendbuf, recvbuf, count, getNcclDtype(datatype), root,
                             getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t send(
    const void *sendbuf,
    size_t count,
    infiniDtype_t datatype,
    int peer,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (sendbuf == nullptr || comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (peer < 0 || peer >= comm->world_size || count == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclSend(sendbuf, count, getNcclDtype(datatype), peer,
                        getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t recv(
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    int peer,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (recvbuf == nullptr || comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (peer < 0 || peer >= comm->world_size || count == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }
    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclRecv(recvbuf, count, getNcclDtype(datatype), peer,
                        getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allGather(
    void *sendbuf,
    void *recvbuf,
    size_t send_count,
    infiniDtype_t datatype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclAllGather(sendbuf, recvbuf, send_count, getNcclDtype(datatype),
                             getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allGatherV(
    void *sendbuf,
    void *recvbuf,
    const size_t *recv_counts,
    int nranks,
    infiniDtype_t datatype,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);
    CHECK_OR_DO(nranks == comm->world_size, return INFINI_STATUS_BAD_PARAM);

    auto cuda_stream = getCudaStream(stream);
    ncclComm_t nccl_comm = getNcclComm(comm);
    ncclDataType_t nccl_dtype = getNcclDtype(datatype);
    size_t offset = 0;

    CHECK_NCCL(ncclGroupStart());
    for (int root = 0; root < nranks; ++root) {
        CHECK_NCCL(ncclBroadcast(
            sendbuf,
            static_cast<char *>(recvbuf) + offset,
            recv_counts[root],
            nccl_dtype,
            root,
            nccl_comm,
            cuda_stream));
        offset += recv_counts[root] * infiniSizeOf(datatype);
    }
    CHECK_NCCL(ncclGroupEnd());

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t reduceScatter(
    void *sendbuf,
    void *recvbuf,
    size_t recv_count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    CHECK_NCCL(ncclReduceScatter(sendbuf, recvbuf, recv_count, getNcclDtype(datatype),
                                 getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t reduceScatterV(
    void *sendbuf,
    void *recvbuf,
    const size_t *send_counts,
    int nranks,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);
    CHECK_OR_DO(nranks == comm->world_size, return INFINI_STATUS_BAD_PARAM);

    auto cuda_stream = getCudaStream(stream);
    ncclComm_t nccl_comm = getNcclComm(comm);
    ncclDataType_t nccl_dtype = getNcclDtype(datatype);
    ncclRedOp_t nccl_op = getNcclRedOp(op);
    size_t offset = 0;

    CHECK_NCCL(ncclGroupStart());
    for (int root = 0; root < nranks; ++root) {
        CHECK_NCCL(ncclReduce(
            static_cast<char *>(sendbuf) + offset,
            recvbuf,
            send_counts[root],
            nccl_dtype,
            nccl_op,
            root,
            nccl_comm,
            cuda_stream));
        offset += send_counts[root] * infiniSizeOf(datatype);
    }
    CHECK_NCCL(ncclGroupEnd());

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::cuda
