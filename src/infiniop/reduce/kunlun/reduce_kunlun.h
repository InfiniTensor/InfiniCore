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
        // Fix: Load from shared memory to register first
        Tdata xi;
        __builtin_memcpy(&xi, &data_ptr[i], sizeof(Tdata));
        ss += Tcompute(xi) * Tcompute(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = Tcompute(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    // Fix: Load result from shared memory
    Tcompute result;
    __builtin_memcpy(&result, &temp_storage, sizeof(Tcompute));
    return result;
}

// Sum(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ inline Tcompute sum(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    Tcompute ss = 0;

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        // Fix: Load from shared memory to register first
        Tdata xi;
        __builtin_memcpy(&xi, &data_ptr[i], sizeof(Tdata));
        ss += Tcompute(xi);
    }

    __shared__ Tcompute temp_storage;
    if (core_id() == 0) {
        temp_storage = Tcompute(0.f);
    }
    sync_cluster();

    atomicAdd(&temp_storage, ss);
    sync_cluster();

    // Fix: Load result from shared memory
    Tcompute result;
    __builtin_memcpy(&result, &temp_storage, sizeof(Tcompute));
    return result;
}

// Max(x) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ inline Tdata max(__shared_ptr__ const Tdata *data_ptr, size_t count) {
    // Fix: Load from shared memory to register first
    Tdata max_val;
    __builtin_memcpy(&max_val, &data_ptr[0], sizeof(Tdata));

    for (size_t i = core_id(); i < count; i += BLOCK_SIZE) {
        // Fix: Load from shared memory to register first
        Tdata xi;
        __builtin_memcpy(&xi, &data_ptr[i], sizeof(Tdata));
        max_val = fmax(max_val, xi);
    }

    __shared__ Tdata temp_storage;
    if (core_id() == 0) {
        // Fix: Load initial value from shared memory
        Tdata initial_val;
        __builtin_memcpy(&initial_val, &data_ptr[0], sizeof(Tdata));
        __builtin_memcpy(&temp_storage, &initial_val, sizeof(Tdata));
    }
    sync_cluster();

    atomicMax(&temp_storage, max_val);
    sync_cluster();

    // Fix: Load result from shared memory
    Tdata result;
    __builtin_memcpy(&result, &temp_storage, sizeof(Tdata));
    return result;
}

} // namespace op::common_kunlun::reduce_op

#endif
