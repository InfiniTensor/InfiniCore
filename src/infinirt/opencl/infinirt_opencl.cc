#include "infinirt_opencl.h"
#include "../../utils.h"
#include <CL/cl.h>
#include <mutex>
#include <vector>

#define CHECK_CLRT(RT_API) CHECK_INTERNAL(RT_API, CL_SUCCESS)

static std::mutex init_mutex;
static cl_platform_id platform = nullptr;
static cl_context context = nullptr;
static std::vector<cl_device_id> devices;
static std::vector<std::vector<cl_command_queue>> queues;
static cl_uint device_count = 0;
static bool initialized = false;
thread_local int CUR_DEV_ID = 0;
struct InfinirtEventStruct {
    cl_event ev = nullptr;
    bool bound = false;
};

namespace infinirt::opencl {
static void cleanupResources() {
    if (!queues.empty()) {
        for (auto &qvec : queues) {
            for (auto q : qvec) {
                if (q) {
                    clReleaseCommandQueue(q);
                }
            }
            qvec.clear();
        }
        queues.clear();
    }
    if (context) {
        clReleaseContext(context);
        context = nullptr;
    }
    devices.clear();
    device_count = 0;
    platform = nullptr;
    initialized = false;
}
infiniStatus_t init() {
    std::lock_guard<std::mutex> lk(init_mutex);
    if (initialized) {
        return INFINI_STATUS_SUCCESS;
    }
    cl_int err = CL_SUCCESS;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(1, nullptr, &num_platforms);
    if (err != CL_SUCCESS) {
        cleanupResources();
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    if (num_platforms == 0) {
        return INFINI_STATUS_DEVICE_NOT_FOUND;
    }
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        cleanupResources();
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
    if (err != CL_SUCCESS) {
        cleanupResources();
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    if (device_count == 0) {
        return INFINI_STATUS_DEVICE_NOT_FOUND;
    }
    devices.resize(static_cast<size_t>(device_count));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        cleanupResources();
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    context = clCreateContext(nullptr, device_count, devices.data(), nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        cleanupResources();
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }

    queues.resize(static_cast<size_t>(device_count)); 
    for (cl_uint i = 0; i < device_count; ++i) {
        cl_command_queue q = clCreateCommandQueueWithProperties(context, devices[i], nullptr, &err);
        if (err != CL_SUCCESS) {
            // 清理已创建的队列和 context
            cleanupResources();
            return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
        }
        queues[i].push_back(q); // 初始队列作为默认队列在 index 0
    }
    initialized = true;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getDeviceCount(int *count) { 
    if (!count) {
        return INFINI_STATUS_BAD_PARAM;
    }
    std::lock_guard<std::mutex> lk(init_mutex);
    if (!initialized) {
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    *count = static_cast<int>(device_count);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    std::lock_guard<std::mutex> lk(init_mutex);
    if (!initialized) {
        return INFINI_STATUS_DEVICE_NOT_INITIALIZED;
    }
    if (device_id < 0 || device_id >= static_cast<int>(device_count)) {
        return INFINI_STATUS_DEVICE_NOT_FOUND;
    }
    CUR_DEV_ID = device_id;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    for (auto &q : queues[CUR_DEV_ID]) {
        if (q) {
            CHECK_CLRT(clFinish(q));
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[CUR_DEV_ID], nullptr, &err);
    CHECK_CLRT(err);
    {
        std::lock_guard<std::mutex> lk(init_mutex);
        queues[CUR_DEV_ID].push_back(queue);
    }
    *stream_ptr = queue;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    {
        std::lock_guard<std::mutex> lk(init_mutex);
        auto &qvec = queues[CUR_DEV_ID];
        auto it = std::find(qvec.begin(), qvec.end(), (cl_command_queue)stream);
        if (it != qvec.end()) {
            qvec.erase(it);
        }
    }
    CHECK_CLRT(clReleaseCommandQueue((cl_command_queue)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_CLRT(clFinish((cl_command_queue)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    InfinirtEventStruct *evs = static_cast<InfinirtEventStruct *>(event);
    CHECK_CLRT(clWaitForEvents(1, &evs->ev));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    InfinirtEventStruct *event = new InfinirtEventStruct();
    *event_ptr = static_cast<infinirtEvent_t>(event);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    InfinirtEventStruct *evs = static_cast<InfinirtEventStruct *>(event);
    CHECK_CLRT(clEnqueueMarkerWithWaitList((cl_command_queue)stream, 0, nullptr, &evs->ev));
    evs->bound = true;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    InfinirtEventStruct *evs = static_cast<InfinirtEventStruct *>(event);
    if (!evs->ev) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    cl_int status;
    CHECK_CLRT(clGetEventInfo(evs->ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr));
    if (status == CL_COMPLETE) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    InfinirtEventStruct *evs = static_cast<InfinirtEventStruct *>(event);
    if (!evs->ev) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    CHECK_CLRT(clWaitForEvents(1, &evs->ev));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    InfinirtEventStruct *evs = static_cast<InfinirtEventStruct *>(event);
    if (!evs->ev) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    CHECK_CLRT(clReleaseEvent(evs->ev));
    evs->ev = nullptr;
    delete evs;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    void *p = clSVMAlloc(context, CL_MEM_READ_WRITE, size, 0);
    if (!p) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *p_ptr = p;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    void *p = clSVMAlloc(context, CL_MEM_READ_WRITE, size, 0);
    if (!p) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *p_ptr = p;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    clSVMFree(context, ptr);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    clSVMFree(context, ptr);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    cl_command_queue default_queue = queues[CUR_DEV_ID][0];
    CHECK_CLRT(clEnqueueSVMMemcpy(default_queue, CL_TRUE, dst, src, size, 0, nullptr, nullptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_CLRT(clEnqueueSVMMemcpy((cl_command_queue)stream, CL_FALSE, dst, src, size, 0, nullptr, nullptr));
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return INFINI_STATUS_INTERNAL_ERROR;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return INFINI_STATUS_INTERNAL_ERROR;
}
infiniStatus_t getOpenclDevice(infinirtOpenclDevice_t *cl_device) {
    if (cl_device == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *cl_device = static_cast<infinirtOpenclDevice_t>(devices[CUR_DEV_ID]);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t getOpenclContext(infinirtOpenclContext_t *cl_context) {
    if (cl_context == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *cl_context = static_cast<infinirtOpenclContext_t>(context);
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t getOpenclStream(infinirtOpenclStream_t *cl_queue) {
    if (cl_queue == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    *cl_queue = static_cast<infinirtOpenclStream_t>(queues[CUR_DEV_ID][0]);
    return INFINI_STATUS_SUCCESS;
}
} // namespace infinirt::opencl

__C infiniStatus_t infinirtGetOpenclDevice(infinirtOpenclDevice_t *cl_device) {
    return infinirt::opencl::getOpenclDevice(cl_device);
}
__C infiniStatus_t infinirtGetOpenclContext(infinirtOpenclContext_t *cl_context) {
    return infinirt::opencl::getOpenclContext(cl_context);
}
__C infiniStatus_t infinirtGetOpenclStream(infinirtOpenclStream_t *cl_queue) {
    return infinirt::opencl::getOpenclStream(cl_queue);
}
