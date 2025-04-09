#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "../pool.h"
#include "bang_handle.h"
#include "cnnl.h"
#include "cnrt.h"
#include "infinicore.h"
#include <functional>

#define REM_FOR_STACK 8 // 防止内存溢出以及其他来源的NRAM占用
#define MAX_NRAM_SIZE (__MLU_NRAM_SIZE__ - REM_FOR_STACK)

#define CHECK_BANG(API) CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)

namespace device::bang {

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
