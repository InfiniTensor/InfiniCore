#ifndef __INFINIOP_REDUCE_KUNLUN_H__
#define __INFINIOP_REDUCE_KUNLUN_H__

#include "../../devices/kunlun/kunlun_kernel_common.h"

namespace op::common_kunlun::reduce_op {

using namespace device::kunlun::kernel;

// Sum(x^2) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sumSquared(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        Tdata xi = loadsm(data_ptr + i);
        ss += to<Tcompute>(xi) * to<Tcompute>(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = 0;
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return loadsm(&temp_storage);
}

// Sum(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sum(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        Tdata xi = loadsm(data_ptr + i);
        ss += to<Tcompute>(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = 0;
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    return loadsm(&temp_storage);
}

// Max(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ inline Tdata max(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tdata max_val = loadsm(data_ptr);

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        // Tdata xi = loadsm(data_ptr + i);
        Tdata xi = loadsm(data_ptr + i);
        max_val = fmax(max_val, to<Tdata>(xi));
    }

    __shared__ Tdata temp_storage;
    if (core_id() == 0) {
        temp_storage = loadsm(data_ptr);
    }
    sync_cluster();

    atomicMax(&temp_storage, max_val);
    sync_cluster();

    return loadsm(&temp_storage);
}

} // namespace op::common_kunlun::reduce_op

#endif
