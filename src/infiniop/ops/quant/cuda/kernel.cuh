#ifndef __QUANT_KERNEL_CUH__
#define __QUANT_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>
__device__ inline int round_half_away_from_zero(float x) {
    float ax = fabsf(x);
    float r = floorf(ax + 0.5f);
    return (x >= 0.0f) ? (int)r : -(int)r;
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockQuantKernel(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x,
    int M, int K) {
    int row = blockIdx.x;
    int tid = row * K;

    // ---- 1. reduce max ----
    float local_max = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(
        x + tid, K);

    __shared__ float global_max_f;
    if (threadIdx.x == 0) {
        global_max_f = local_max;
    }
    __syncthreads();

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_min = __FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {
        thread_min = fminf(thread_min, (float)x[tid + ind]);
    }
    float local_min = BlockReduce(temp_storage).Reduce(thread_min, cub::Min());

    __shared__ float global_min_f;
    if (threadIdx.x == 0) {
        global_min_f = local_min;
    }
    __syncthreads();

    // ---- 3. 使用 float（匹配 python）计算 scale/zero ----
    float global_max = global_max_f;
    float global_min = global_min_f;

    float scale = (global_max - global_min) / 255.0f;
    if (scale < 1e-8f) {
        scale = 1e-8f;
    }

    float inv_scale = 1.0f / scale;
    float zero = -global_min * inv_scale - 128.0f;

    // 写回 scale, zero
    x_scale[row] = (Tdata)scale;
    x_zero[row] = (Tdata)zero;

    // ---- 4. 使用 float + half-away-from-zero（与 Python 完全一致）----
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {

        float v = (float)x[tid + ind];
        float qf = v * inv_scale + zero;

        int q = round_half_away_from_zero(qf);

        if (q > 127) {
            q = 127;
        }
        if (q < -128) {
            q = -128;
        }

        x_packed[tid + ind] = (int8_t)q;
    }
}

template <typename Tdata, unsigned int BLOCK_SIZE>
__device__ void blockQuantSymKernel(
    int8_t *x_packed, Tdata *x_scale, const Tdata *x,
    int M, int K) {
    int row = blockIdx.x;
    int tid = row * K;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // ---- 2. reduce min ----
    float thread_max = -__FLT_MAX__;
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, fabs((float)x[tid + ind]));
    }
    float local_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());

    __shared__ float global_max_f;
    if (threadIdx.x == 0) {
        global_max_f = local_max;
    }
    __syncthreads();

    // ---- 3. 使用 float（匹配 python）计算 scale/zero ----
    float global_max = global_max_f;

    float scale = global_max / 127.0f;
    if (scale < 1e-8f) {
        scale = 1e-8f;
    }

    float inv_scale = 1.0f / scale;

    // 写回 scale, zero
    x_scale[row] = (Tdata)scale;

    // ---- 4. 使用 float + half-away-from-zero（与 Python 完全一致）----
    for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE) {

        float v = (float)x[tid + ind];
        float qf = v * inv_scale;

        int q = round_half_away_from_zero(qf);

        if (q > 127) {
            q = 127;
        }
        if (q < -128) {
            q = -128;
        }

        x_packed[tid + ind] = (int8_t)q;
    }
}

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
};
template <typename T>
struct MinOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return min(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpQuantKernel(
    int8_t *x_packed, Tdata *x_scale, Tdata *x_zero, const Tdata *x,
    int M, int K) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * K;

    if (otherIdx < M) {

        __shared__ float max_total[BLOCK_SIZE_y];
        __shared__ float min_total[BLOCK_SIZE_y];

        float max_data = -__FLT_MAX__;
        float min_data = __FLT_MAX__;

        // ---- reduce max/min ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            float v = (float)x[tid + ind];
            max_data = fmaxf(max_data, v);
            min_data = fminf(min_data, v);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);
        min_data = WarpAllReduce<MinOp, float, BLOCK_SIZE_x>(min_data);

        if (threadIdx.x == 0) {
            max_total[threadIdx.y] = max_data;
            min_total[threadIdx.y] = min_data;
        }
        __syncthreads();

        // ---- float scale/zero（与 Python float32 匹配）----
        float max_f = max_total[threadIdx.y];
        float min_f = min_total[threadIdx.y];

        float scale = (max_f - min_f) / 255.0f;
        if (scale < 1e-8f) {
            scale = 1e-8f;
        }

        float inv_scale = 1.0f / scale;
        float zero = -min_f * inv_scale - 128.0f;

        x_scale[otherIdx] = (Tdata)scale;
        x_zero[otherIdx] = (Tdata)zero;

        // ---- float + half-away-from-zero 量化 ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            float v = (float)x[tid + ind];
            float qf = v * inv_scale + zero;

            int q = round_half_away_from_zero(qf);

            if (q > 127) {
                q = 127;
            }
            if (q < -128) {
                q = -128;
            }

            x_packed[tid + ind] = (int8_t)q;
        }
    }
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
__device__ void warpQuantSymKernel(
    int8_t *x_packed, Tdata *x_scale, const Tdata *x,
    int M, int K) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = otherIdx * K;

    if (otherIdx < M) {

        __shared__ float max_total[BLOCK_SIZE_y];

        float max_data = -__FLT_MAX__;

        // ---- reduce max/min ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            float v = fabs((float)x[tid + ind]);
            max_data = fmaxf(max_data, v);
        }

        max_data = WarpAllReduce<MaxOp, float, BLOCK_SIZE_x>(max_data);

        if (threadIdx.x == 0) {
            max_total[threadIdx.y] = max_data;
        }
        __syncthreads();

        // ---- float scale/zero（与 Python float32 匹配）----
        float max_f = max_total[threadIdx.y];

        float scale = max_f / 127.0f;
        if (scale < 1e-8f) {
            scale = 1e-8f;
        }

        float inv_scale = 1.0f / scale;

        x_scale[otherIdx] = (Tdata)scale;

        // ---- float + half-away-from-zero 量化 ----
        for (int ind = threadIdx.x; ind < K; ind += BLOCK_SIZE_x) {
            float v = (float)x[tid + ind];
            float qf = v * inv_scale;

            int q = round_half_away_from_zero(qf);

            if (q > 127) {
                q = 127;
            }
            if (q < -128) {
                q = -128;
            }

            x_packed[tid + ind] = (int8_t)q;
        }
    }
}

#endif // __QUANT_KERNEL_CUH__
