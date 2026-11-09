#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <infiniccl/infiniccl.h>

#define OLD_STATUS_SUCCESS 0
#define OLD_DEVICE_ASCEND 3
#define OLD_DTYPE_I8 3
#define OLD_DTYPE_I16 4
#define OLD_DTYPE_I32 5
#define OLD_DTYPE_I64 6
#define OLD_DTYPE_U8 7
#define OLD_DTYPE_U16 8
#define OLD_DTYPE_U32 9
#define OLD_DTYPE_U64 10
#define OLD_DTYPE_F16 12
#define OLD_DTYPE_F32 13
#define OLD_DTYPE_F64 14
#define OLD_DTYPE_BF16 19

typedef struct {
    char internal[128];
} old_unique_id_t;

struct OldApi {
    void *handle;
    int (*comm_init_all)(int, infinicclComm_t *, int, const int *);
    int (*get_unique_id)(old_unique_id_t *);
    int (*comm_init_rank)(infinicclComm_t *, int, old_unique_id_t, int);
    int (*comm_destroy)(infinicclComm_t);
    int (*all_reduce)(const void *, void *, size_t, int, int,
                      infinicclComm_t, void *);
    int (*all_gather)(const void *, void *, size_t, int, infinicclComm_t,
                      void *);
    int (*reduce_scatter)(const void *, void *, size_t, int, int,
                          infinicclComm_t, void *);
    int (*send)(const void *, size_t, int, int, infinicclComm_t, void *);
    int (*recv)(void *, size_t, int, int, infinicclComm_t, void *);
};

static OldApi load_old_api() {
    OldApi api{};
    api.handle = dlopen("libinfiniccl_hccl.so", RTLD_NOW | RTLD_LOCAL);
    if (api.handle == NULL) {
        return api;
    }
    api.comm_init_all = reinterpret_cast<decltype(api.comm_init_all)>(
        dlsym(api.handle, "infinicclCommInitAll"));
    api.get_unique_id = reinterpret_cast<decltype(api.get_unique_id)>(
        dlsym(api.handle, "infinicclGetUniqueId"));
    api.comm_init_rank = reinterpret_cast<decltype(api.comm_init_rank)>(
        dlsym(api.handle, "infinicclCommInitRank"));
    api.comm_destroy = reinterpret_cast<decltype(api.comm_destroy)>(
        dlsym(api.handle, "infinicclCommDestroy"));
    api.all_reduce = reinterpret_cast<decltype(api.all_reduce)>(
        dlsym(api.handle, "infinicclAllReduce"));
    api.all_gather = reinterpret_cast<decltype(api.all_gather)>(
        dlsym(api.handle, "infinicclAllGather"));
    api.reduce_scatter = reinterpret_cast<decltype(api.reduce_scatter)>(
        dlsym(api.handle, "infinicclReduceScatter"));
    api.send = reinterpret_cast<decltype(api.send)>(dlsym(api.handle, "infinicclSend"));
    api.recv = reinterpret_cast<decltype(api.recv)>(dlsym(api.handle, "infinicclRecv"));
    return api;
}

static const OldApi &old_api() {
    static const OldApi api = load_old_api();
    return api;
}

static infinicclResult_t map_status(int status) {
    if (status == OLD_STATUS_SUCCESS) {
        return infinicclSuccess;
    }
    if (status == 3 || status == 4) {
        return infinicclInvalidArgument;
    }
    if (status == 6 || status == 7) {
        return infinicclSystemError;
    }
    return infinicclInternalError;
}

static int map_datatype(infinicclDataType_t datatype) {
    switch (datatype) {
    case infinicclChar:
        return OLD_DTYPE_I8;
    case infinicclInt16:
        return OLD_DTYPE_I16;
    case infinicclInt:
        return OLD_DTYPE_I32;
    case infinicclInt64:
        return OLD_DTYPE_I64;
    case infinicclUInt8:
        return OLD_DTYPE_U8;
    case infinicclUInt16:
        return OLD_DTYPE_U16;
    case infinicclUInt32:
        return OLD_DTYPE_U32;
    case infinicclUInt64:
        return OLD_DTYPE_U64;
    case infinicclHalf:
        return OLD_DTYPE_F16;
    case infinicclBFloat16:
        return OLD_DTYPE_BF16;
    case infinicclFloat:
        return OLD_DTYPE_F32;
    case infinicclDouble:
        return OLD_DTYPE_F64;
    default:
        return -1;
    }
}

static int valid_datatype(infinicclDataType_t datatype) {
    return map_datatype(datatype) >= 0;
}

infinicclResult_t infinicclInit(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
    return infinicclSuccess;
}

infinicclResult_t infinicclFinalize(void) {
    return infinicclSuccess;
}

infinicclResult_t infinicclGetUniqueId(infinicclUniqueId *id) {
    if (id == NULL) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.get_unique_id == NULL) {
        return infinicclInternalError;
    }
    old_unique_id_t old_id;
    memset(&old_id, 0, sizeof(old_id));
    const int status = old.get_unique_id(&old_id);
    if (status != OLD_STATUS_SUCCESS) {
        return map_status(status);
    }
    memset(id, 0, sizeof(*id));
    memcpy(id->internal, old_id.internal, sizeof(old_id.internal));
    return infinicclSuccess;
}

infinicclResult_t infinicclCommInitAll(infinicclComm_t *comm, int ndev,
                                       const int *devlist) {
    if (comm == NULL || devlist == NULL || ndev <= 0) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.comm_init_all == NULL) {
        return infinicclInternalError;
    }
    return map_status(
        old.comm_init_all(OLD_DEVICE_ASCEND, comm, ndev, devlist));
}

infinicclResult_t infinicclCommInitRank(infinicclComm_t *comm, int nranks,
                                        infinicclUniqueId id, int rank) {
    if (comm == NULL || nranks <= 0 || rank < 0 || rank >= nranks) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.comm_init_rank == NULL) {
        return infinicclInternalError;
    }
    old_unique_id_t old_id;
    memcpy(old_id.internal, id.internal, sizeof(old_id.internal));
    return map_status(old.comm_init_rank(comm, nranks, old_id, rank));
}

infinicclResult_t infinicclCommDestroy(infinicclComm_t comm) {
    const OldApi &old = old_api();
    if (old.comm_destroy == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.comm_destroy(comm));
}

infinicclResult_t infinicclAllReduce(const void *sendbuff, void *recvbuff,
                                     size_t count,
                                     infinicclDataType_t datatype,
                                     infinicclRedOp_t op,
                                     infinicclComm_t comm, void *stream) {
    if (!valid_datatype(datatype)) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.all_reduce == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.all_reduce(sendbuff, recvbuff, count,
                                     map_datatype(datatype), (int)op, comm,
                                     stream));
}

infinicclResult_t infinicclAllGather(const void *sendbuff, void *recvbuff,
                                     size_t count,
                                     infinicclDataType_t datatype,
                                     infinicclComm_t comm, void *stream) {
    if (!valid_datatype(datatype)) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.all_gather == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.all_gather(sendbuff, recvbuff, count,
                                     map_datatype(datatype), comm, stream));
}

infinicclResult_t infinicclReduceScatter(const void *sendbuff, void *recvbuff,
                                         size_t recvcount,
                                         infinicclDataType_t datatype,
                                         infinicclRedOp_t op,
                                         infinicclComm_t comm, void *stream) {
    if (!valid_datatype(datatype)) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.reduce_scatter == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.reduce_scatter(sendbuff, recvbuff, recvcount,
                                         map_datatype(datatype), (int)op,
                                         comm, stream));
}

infinicclResult_t infinicclSend(const void *sendbuff, size_t count,
                                infinicclDataType_t datatype, int peer,
                                infinicclComm_t comm, void *stream) {
    if (!valid_datatype(datatype)) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.send == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.send(sendbuff, count, map_datatype(datatype), peer,
                               comm, stream));
}

infinicclResult_t infinicclRecv(void *recvbuff, size_t count,
                                infinicclDataType_t datatype, int peer,
                                infinicclComm_t comm, void *stream) {
    if (!valid_datatype(datatype)) {
        return infinicclInvalidArgument;
    }
    const OldApi &old = old_api();
    if (old.recv == NULL) {
        return infinicclInternalError;
    }
    return map_status(old.recv(recvbuff, count, map_datatype(datatype), peer,
                               comm, stream));
}
