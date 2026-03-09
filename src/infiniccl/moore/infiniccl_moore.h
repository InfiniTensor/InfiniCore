#ifndef INFINICCL_MOORE_H_
#define INFINICCL_MOORE_H_

#include "../infiniccl_impl.h"
#include <vector>
#include <cstdint>

#if defined(ENABLE_MOORE_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(moore)
#else
INFINICCL_DEVICE_API_NOOP(moore)
#endif

#endif /* INFINICCL_MOORE_H_ */

struct CustomAllReduceComm {
    const size_t max_size = 128 * 1024 * 1024;
    const std::vector<int> support_world_sizes = {2,8};
    
    void *acomm;
    int devices;
    int64_t crank;
    bool use_custom_all_reduce = true;
    std::vector<int64_t> meta_ptrs;
    std::vector<int64_t> buffer_ptrs;
    void* rank_data;
    int64_t custom_ptr;
    
    CustomAllReduceComm(int64_t rank, int ndev, void* group);
    ~CustomAllReduceComm();
    bool should_custom_ar(void* inp, size_t count, infiniDtype_t datatype);
};