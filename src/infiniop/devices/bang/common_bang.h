#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include <functional>

#define CHECK_BANG(API) CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)

#define NRAM_MAX_SIZE 1024 * 240
constexpr size_t ALIGN_SIZE = 128;

namespace device::bang {

// CNRT TaskTopo capture cannot record a sync on the queue being captured.
inline bool isQueueCapturing(cnrtQueue_t queue) {
    cnrtQueueCaptureStatus_t status = cnrtQueueCaptureStatusNone;
    uint64_t capture_id = 0;
    cnrtTaskTopo_t task_topo = nullptr;
    const cnrtTaskTopoNode_t *dependencies = nullptr;
    size_t num_dependencies = 0;
    auto ret = cnrtQueueGetCaptureInfo(
        queue,
        &status,
        &capture_id,
        &task_topo,
        &dependencies,
        &num_dependencies);
    return ret == cnrtSuccess && status == cnrtQueueCaptureStatusActive;
}

inline infiniStatus_t syncQueueIfNotCapturing(cnrtQueue_t queue) {
    if (!isQueueCapturing(queue)) {
        CHECK_INTERNAL(cnrtQueueSync(queue), cnrtSuccess);
    }
    return INFINI_STATUS_SUCCESS;
}

class Handle::Internal {
    Pool<cnnlHandle_t> cnnl_handles;

    int _core_per_cluster;
    int _cluster_count;

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useCnnl(cnrtQueue_t queue, const Fn<cnnlHandle_t> &f) const;

    int getCorePerCluster() const;
    int getClusterCount() const;
};

cnnlDataType_t getCnnlDtype(infiniDtype_t dt);

// set cnnl tensor descriptor without strides
infiniStatus_t setCnnlTensor(cnnlTensorDescriptor_t desc,
                             const InfiniopTensorDescriptor *layout);

// set cnnl tensor descriptor with strides
infiniStatus_t setCnnlTensorEx(cnnlTensorDescriptor_t desc,
                               const InfiniopTensorDescriptor *layout);

} // namespace device::bang

#endif // __COMMON_BANG_H__
