#include "infiniccl_cuda.h"

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>

#if defined(ENABLE_HYGON_API)
#include <algorithm>
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

#if defined(ENABLE_HYGON_API)
namespace {

constexpr int kHygonTp2MaxBlocks = 80;
constexpr size_t kHygonTp2StageCapacityElements = 1u << 22;

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
        available = library != nullptr &&
                    load(mem_get_allocation_granularity, "cuMemGetAllocationGranularity") &&
                    load(mem_address_reserve, "cuMemAddressReserve") &&
                    load(mem_create, "cuMemCreate") &&
                    load(mem_map, "cuMemMap") &&
                    load(mem_set_access, "cuMemSetAccess") &&
                    load(mem_unmap, "cuMemUnmap") &&
                    load(mem_release, "cuMemRelease") &&
                    load(mem_address_free, "cuMemAddressFree");
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

struct HygonVmmAllocation {
    void *ptr = nullptr;
    size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
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
            if (rank_data[rank] != nullptr) cudaFree(rank_data[rank]);
            if (signal_hosts[rank] != nullptr) cudaFreeHost(signal_hosts[rank]);
            auto &driver = hygon_cuda_driver_api();
            if (driver.available && stages[rank].ptr != nullptr) {
                const auto address = reinterpret_cast<CUdeviceptr>(stages[rank].ptr);
                driver.mem_unmap(address, stages[rank].size);
                driver.mem_address_free(address, stages[rank].size);
            }
            if (driver.available && stages[rank].handle != 0) {
                driver.mem_release(stages[rank].handle);
            }
        }
        if (restore_device) cudaSetDevice(previous_device);
    }
};

std::mutex hygon_tp2_states_mutex;
std::unordered_map<infinicclComm_t, std::shared_ptr<HygonTp2AllReduceState>> hygon_tp2_states;

bool allocate_hygon_vmm(HygonVmmAllocation &allocation,
                        int owner_device,
                        const int device_ids[2],
                        size_t requested_size) {
    auto &driver = hygon_cuda_driver_api();
    if (!driver.available) return false;
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = owner_device;
    size_t granularity = 0;
    if (driver.mem_get_allocation_granularity(
            &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
        granularity == 0) return false;
    allocation.size = (requested_size + granularity - 1) / granularity * granularity;
    CUdeviceptr address = 0;
    if (driver.mem_address_reserve(
            &address, allocation.size, granularity, 0, 0) != CUDA_SUCCESS) {
        allocation = {};
        return false;
    }
    allocation.ptr = reinterpret_cast<void *>(address);
    if (driver.mem_create(
            &allocation.handle, allocation.size, &properties, 0) != CUDA_SUCCESS) {
        driver.mem_address_free(address, allocation.size);
        allocation = {};
        return false;
    }
    if (driver.mem_map(
            address, allocation.size, 0, allocation.handle, 0) != CUDA_SUCCESS) {
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
                   __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < next_flag) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) self_signal->flag[blockIdx.x] = next_flag;
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
                   __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag) {
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
    const uint32_t sync_flag =
        hygon_tp2_start_sync<num_ranks>(rank_signals, self_signal, rank);
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
                const float value =
                    __bfloat162float(shared[threadIdx.x * pack_size + element]) +
                    __bfloat162float(shared[(threads_per_rank + threadIdx.x) * pack_size + element]);
                reduced.values[element] = __float2bfloat16(value);
            }
            reinterpret_cast<HygonBf16Pack *>(output)[index] = reduced;
        }
        __syncthreads();
    }
    if (pack_offset == 0) {
        hygon_tp2_end_sync<num_ranks>(
            rank_signals, self_signal, rank, sync_flag);
    }
}

std::shared_ptr<HygonTp2AllReduceState> create_hygon_tp2_state(
    int ndevice, const int *device_ids) {
    if (ndevice != 2 || device_ids == nullptr) return nullptr;
    auto state = std::make_shared<HygonTp2AllReduceState>();
    state->device_ids[0] = device_ids[0];
    state->device_ids[1] = device_ids[1];
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    auto fail = [&]() -> std::shared_ptr<HygonTp2AllReduceState> {
        if (restore_device) cudaSetDevice(previous_device);
        return nullptr;
    };
    const size_t stage_bytes = kHygonTp2StageCapacityElements * sizeof(__nv_bfloat16);
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaSetDevice(device_ids[rank]) != cudaSuccess ||
            !allocate_hygon_vmm(state->stages[rank], device_ids[rank],
                                state->device_ids, stage_bytes)) return fail();
        void *signal_host = nullptr;
        if (cudaHostAlloc(&signal_host, sizeof(HygonTp2Signal),
                          cudaHostAllocMapped) != cudaSuccess) return fail();
        state->signal_hosts[rank] = static_cast<HygonTp2Signal *>(signal_host);
        std::memset(state->signal_hosts[rank], 0, sizeof(HygonTp2Signal));
        void *signal_device = nullptr;
        if (cudaHostGetDevicePointer(&signal_device, signal_host, 0) != cudaSuccess) return fail();
        state->signals[rank] = static_cast<HygonTp2Signal *>(signal_device);
        if (cudaMalloc(reinterpret_cast<void **>(&state->rank_data[rank]),
                      sizeof(HygonTp2RankData)) != cudaSuccess) return fail();
    }
    HygonTp2RankData host_rank_data{{state->stages[0].ptr, state->stages[1].ptr}};
    state->rank_signals.signals[0] = state->signals[0];
    state->rank_signals.signals[1] = state->signals[1];
    for (int rank = 0; rank < 2; ++rank) {
        cudaSetDevice(device_ids[rank]);
        if (cudaMemcpy(state->rank_data[rank], &host_rank_data,
                      sizeof(host_rank_data), cudaMemcpyHostToDevice) != cudaSuccess) return fail();
    }
    if (restore_device) cudaSetDevice(previous_device);
    return state;
}

void register_hygon_tp2_state(infinicclComm_t *comms,
                              int ndevice,
                              const int *device_ids) {
    auto state = create_hygon_tp2_state(ndevice, device_ids);
    if (state == nullptr) return;
    std::lock_guard<std::mutex> lock(hygon_tp2_states_mutex);
    for (int rank = 0; rank < 2; ++rank) hygon_tp2_states.emplace(comms[rank], state);
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
    if (aligned_offset > kHygonTp2StageCapacityElements - count) return false;
    *element_offset = aligned_offset;
    cursor.next_element = aligned_offset + count;
    return true;
}

bool try_hygon_tp2_graph_allreduce(
    void *sendbuf, void *recvbuf, size_t count,
    infiniDtype_t datatype, infinicclReduceOp_t op,
    infinicclComm_t comm, cudaStream_t stream) {
    if (comm == nullptr || comm->world_size != 2 ||
        datatype != INFINI_DTYPE_BF16 || op != INFINICCL_SUM ||
        count == 0 || count > kHygonTp2StageCapacityElements || (count % 8) != 0) return false;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id) != cudaSuccess ||
        capture_status != cudaStreamCaptureStatusActive) return false;
    auto state = get_hygon_tp2_state(comm);
    if (state == nullptr || comm->rank < 0 || comm->rank >= 2) return false;
    const int rank = comm->rank;
    size_t element_offset = 0;
    if (!reserve_hygon_tp2_stage(state, rank, capture_id, count, &element_offset)) return false;
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
