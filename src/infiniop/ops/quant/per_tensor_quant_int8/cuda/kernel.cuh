#ifndef __PER_TENSOR_QUANT_INT8_KERNEL_CUH__
#define __PER_TENSOR_QUANT_INT8_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

__device__ inline int round_half_away_from_zero(float x) {
    float ax = fabsf(x);
    float r = floorf(ax + 0.5f);
    return (x >= 0.0f) ? (int)r : -(int)r;
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void perTensorQuantI8SymKernel(
    int8_t *x_packed, float *x_scale, const Tdata *x,
    int num_elements) {

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < num_elements; ind += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, fabs((float)x[ind]));
    }

#if CUDART_VERSION >= 12090
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, ::cuda::maximum());
#else
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
#endif
    __shared__ float global_max;
    if (threadIdx.x == 0) {
        global_max = local_max;
    }
    __syncthreads();
    float scale = global_max / 127.0f;
    if (scale < 1e-8f) {
        scale = 1e-8f;
    }
    if (gid == 0) {
        x_scale[0] = scale;
    }
    __syncthreads();
    float scale_val = 1.0f / scale;

    for (int tid = gid; tid < num_elements; tid += grid_size) {

        float qf = (float)x[tid] * scale_val;
        int q = round_half_away_from_zero(qf);

        if (q > 127) {
            q = 127;
        }
        if (q < -127) {
            q = -127;
        }

        x_packed[tid] = (int8_t)q;
    }
}

#endif // __PER_TENSOR_QUANT_INT8_KERNEL_CUH__
