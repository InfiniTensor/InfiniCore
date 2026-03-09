#include "infiniccl_moore.h"
#include "custom_all_reduce.h"

#include "../../utils.h"

#include <mccl.h>
#include <musa_runtime.h>

#include <iostream>
#include <vector>
#include <cstddef>
#include <cstring>
#include <future>

#define CHECK_MCCL(API__) CHECK_INTERNAL(API__, mcclSuccess)

inline musaStream_t getMusaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<musaStream_t>(stream);
}

inline mcclDataType_t getMcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return mcclFloat;
    case INFINI_DTYPE_F16:
        return mcclHalf;

#if MARCH_TYPE == 310
    case INFINI_DTYPE_BF16:
        return mcclBfloat16;
#endif

    default:
        std::abort();
        return mcclHalf;
    }
}

inline mcclRedOp_t getMcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return mcclSum;
    case INFINICCL_PROD:
        return mcclProd;
    case INFINICCL_MAX:
        return mcclMax;
    case INFINICCL_MIN:
        return mcclMin;
    case INFINICCL_AVG:
        return mcclAvg;
    default:
        std::abort();
        return mcclSum;
    }
}

inline mcclComm_t getMcclComm(infinicclComm_t comm) {
    CustomAllReduceComm* customComm = static_cast<CustomAllReduceComm*>(comm->comm);
    return static_cast<mcclComm_t>(customComm->acomm);
}

namespace infiniccl::moore {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<mcclComm_t> mccl_comms(ndevice);
    CHECK_MCCL(mcclCommInitAll(mccl_comms.data(), ndevice, (int const *)device_ids));

    std::vector<std::future<CustomAllReduceComm*>> futures;
    futures.reserve(ndevice);

    for (int i = 0; i < ndevice; i++) {
        futures.emplace_back(std::async(std::launch::async, [i, ndevice, &mccl_comms]() {
            return new CustomAllReduceComm(i, ndevice, mccl_comms[i]);
        }));
    }

    for (int i = 0; i < ndevice; i++) {
        auto ca = futures[i].get();
        comms[i] = new InfinicclComm{INFINI_DEVICE_MOORE, device_ids[i], (void *)(ca)};
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_MCCL(mcclCommDestroy(getMcclComm(comm)));
    // CustomAllReduceComm* customComm = static_cast<CustomAllReduceComm*>(comm->comm);
    // delete customComm;
    delete comm;
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

    if (datatype != INFINI_DTYPE_F32 && datatype != INFINI_DTYPE_F16 && datatype != INFINI_DTYPE_BF16) {
        return INFINI_STATUS_BAD_PARAM;
    }

    CustomAllReduceComm* customComm = static_cast<CustomAllReduceComm*>(comm->comm);
    if (customComm->should_custom_ar(sendbuf, count, datatype) == false) {
        // std::cout << "using mccl" << std::endl;
        // std::cout << "using mccl count: " << count << ", datatype: " << getMcclDtype(datatype) << std::endl;
        CHECK_MCCL(mcclAllReduce(sendbuf, recvbuf, count, getMcclDtype(datatype),
                             getMcclRedOp(op), getMcclComm(comm), getMusaStream(stream)));
        // std::cout << "end cmccl" << std::endl;
    } else {
        // std::cout << "using custom all reduce" << std::endl;
        // std::cout << "using mccl count: " << count << ", datatype: " << getMcclDtype(datatype) << std::endl;
        auto rank = customComm->crank;
        all_reduce(customComm->custom_ptr, sendbuf, recvbuf, count, getMcclDtype(datatype), customComm->buffer_ptrs[rank], customComm->max_size, getMusaStream(stream));
        musaError_t e = musaStreamSynchronize(getMusaStream(stream));
        if (e != musaSuccess) {
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
                   musaGetErrorString(e));
            exit(EXIT_FAILURE);
        }
        // std::cout << "end custom all reduce" << std::endl;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::moore

#define CHECK_MUSA_SUCCESS(cmd)                                              \
    do {                                                              \
      musaError_t e = cmd;                                            \
      if (e != musaSuccess) {                                         \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
               musaGetErrorString(e));                                \
        exit(EXIT_FAILURE);                                           \
      }                                                               \
    } while (0)



std::vector<int64_t> create_shared_buffer(int64_t size_in_bytes, int ndev, int rank, void* comm) {
    musaStream_t stream;
    CHECK_MUSA_SUCCESS(musaStreamCreateWithFlags(&stream, musaStreamNonBlocking));

    void* pointer = nullptr;
    CHECK_MUSA_SUCCESS(musaMalloc(&pointer, size_in_bytes));
    CHECK_MUSA_SUCCESS(musaMemset(pointer, 0, size_in_bytes));

    musaIpcMemHandle_t handle;
    CHECK_MUSA_SUCCESS(musaIpcGetMemHandle(&handle, pointer));

    size_t handle_size = sizeof(musaIpcMemHandle_t);

    void* input_tensor = nullptr;
    void* recv_buffer = nullptr;
    CHECK_MUSA_SUCCESS(musaMalloc(&input_tensor, handle_size));
    CHECK_MUSA_SUCCESS(musaMalloc(&recv_buffer, handle_size * ndev));
    CHECK_MUSA_SUCCESS(musaMemcpyAsync(input_tensor, &handle, handle_size, musaMemcpyHostToDevice, stream));

    mcclResult_t  e = mcclAllGather(input_tensor, recv_buffer, handle_size, mcclUint8, static_cast<mcclComm_t>(comm), stream);
    if (e != mcclSuccess) {
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,
               mcclGetErrorString(e));
        exit(EXIT_FAILURE);
    }

    CHECK_MUSA_SUCCESS(musaStreamSynchronize(stream));

    musaIpcMemHandle_t* handles = new musaIpcMemHandle_t[ndev];
    musaMemcpy(handles, recv_buffer, handle_size * ndev, musaMemcpyDeviceToHost);

    std::vector<void*> pointers;
    for (int i = 0; i < ndev; ++i) {
        if (i == rank) {
            pointers.push_back(pointer);
        } else {
            void* remote_ptr = nullptr;
            CHECK_MUSA_SUCCESS(musaIpcOpenMemHandle(&remote_ptr, handles[i], musaIpcMemLazyEnablePeerAccess));
            pointers.push_back(remote_ptr);
        }
    }

    std::vector<int64_t> int_pointers;

    int_pointers.resize(pointers.size());
    std::transform(pointers.begin(), pointers.end(), int_pointers.begin(),
        [](void* ptr) -> int64_t {
            return reinterpret_cast<int64_t>(ptr);
        }
    );

    musaStreamDestroy(stream);
    // musaFree(pointer);
    // musaFree(input_tensor);
    // musaFree(recv_buffer);
    return int_pointers;
}

void free_shared_buffer(std::vector<int64_t> pointers, int rank) {
    void* pointer = reinterpret_cast<void*>(pointers[rank]);
    musaFree(pointer);
}

bool CustomAllReduceComm::should_custom_ar(void* inp, size_t count, infiniDtype_t datatype) {
    if (!use_custom_all_reduce) {
        return false;
    }

    if (count % 16 != 0) {
        return false;
    }

    return (count * 2) < max_size;
}

CustomAllReduceComm::CustomAllReduceComm(int64_t rank, int ndev, void* comm) {
    CHECK_MUSA_SUCCESS(musaSetDevice(rank));

    devices = ndev;
    crank = rank;
    acomm = comm;
    auto it = std::find(
        support_world_sizes.begin(),
        support_world_sizes.end(),
        devices
    );

    if (it == support_world_sizes.end()) {
        use_custom_all_reduce = false;
        return;
    }

    int64_t metasize = meta_size();
    meta_ptrs = create_shared_buffer(
        metasize + max_size, devices, crank, acomm
    );

    size_t num_elements = 8 * 1024 * 1024;
    size_t element_size = sizeof(uint8_t);
    size_t total_bytes = num_elements * element_size;
    void* rank_data = nullptr;
    CHECK_MUSA_SUCCESS(musaMalloc(&rank_data, total_bytes));
    buffer_ptrs = create_shared_buffer(max_size, devices, crank, acomm);

    custom_ptr = init_custom_ar(meta_ptrs, rank_data, num_elements, crank, true);
    register_buffer(custom_ptr, buffer_ptrs);
}

// CustomAllReduceComm::~CustomAllReduceComm() {
//     musaFree(rank_data);
//     for (int i = 0; i < devices; ++i) {
//         free_shared_buffer(meta_ptrs, i);
//         free_shared_buffer(buffer_ptrs, i);
//     }
// }

