#ifndef __INFINIOP_REDUCE_METAX_H__
#define __INFINIOP_REDUCE_METAX_H__

#include <hccub/block/block_reduce.cuh>
#include <hc_runtime.h>

/*
 * Device functions for reduction operations on Metax.
 *
 * Note: Only local result on thread 0 is guranteed to be correct.
 *       A manual broadcast is needed for other threads.
 *
 * Important Note: This is a device-independent header file containing reduce kernels
 *                 for all metax-supporting platforms. Include device-specific headers
 *                 (such as <hccub/block/block_reduce.cuh> for metax) in your source file
 *                 and then include this file for proper usage.
 */
namespace op::common_metax::reduce_op {

// Sum(x^2) on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename T>
__device__ T sumSquare(const T *data, size_t count) {
    typedef hccub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T thread_data = 0;
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        T val = data[i];
        thread_data += val * val;
    }

    T block_sum = BlockReduce(temp_storage).Sum(thread_data);
    return block_sum;
}

// Sum on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename T>
__device__ T sum(const T *data, size_t count) {
    typedef hccub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T thread_data = 0;
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        thread_data += data[i];
    }

    T block_sum = BlockReduce(temp_storage).Sum(thread_data);
    return block_sum;
}

// Max on contiguous data of length count
template <unsigned int BLOCK_SIZE, typename T>
__device__ T max(const T *data, size_t count) {
    typedef hccub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    T thread_data = (count > 0) ? data[0] : T(0);
    for (size_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        if (data[i] > thread_data) {
            thread_data = data[i];
        }
    }

    T block_max = BlockReduce(temp_storage).Reduce(thread_data, hccub::Max());
    return block_max;
}

} // namespace op::common_metax::reduce_op

#endif // __INFINIOP_REDUCE_METAX_H__