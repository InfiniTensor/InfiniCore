#ifndef __SOFTMAX_CUDA_KERNEL_H__
#define __SOFTMAX_CUDA_KERNEL_H__
#include "../../../devices/cuda/cuda_kernel_common.cuh"
#include "softmax_cuda.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

struct __align__(8) MD {
    float max;
    float sum;
};

__device__ __forceinline__ MD reduce_for_md(MD a, MD b) {
    bool is_a_bigger = a.max > b.max;
    MD bigger = is_a_bigger ? a : b;
    MD smaller = is_a_bigger ? b : a;
    bigger.sum = bigger.sum + __expf(smaller.max - bigger.max) * smaller.sum;
    return bigger;
}

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half &a, const half &b) const {
        return __hadd(a, b);
    }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return max(a, b);
    }
};

template <>
struct MaxOp<half> {
    __device__ __forceinline__ half operator()(const half &a, const half &b) const {
        return __hmax(a, b);
    }
};

template <typename T, template <typename> class ReduceOp, int thread_group_width = 32>
__device__ __forceinline__ T warpReduce(T value) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        value = ReduceOp<T>()(value, __shfl_xor_sync(0xffffffff, value, mask));
    }
    return value;
}

// 高维度softmax，例如已知axis=1，输入为shape为[a1, a2, a3, a4]
// 在a1 * a3 * a4组tensor，每个tensor的shape为[a2]，进行求和和求max
// 所以我在算子desc创建的时候需要计算出规约轴中访问每个元素的步长stride以及总共多少组长度为a2的tensor
// 当规约轴元素较少的时候可以一个warp处理一组tensor，以1024为界限，当规约轴元素少于1024时使用一个warp处理一组tensor
// 1024 / 32 = 32
// blockDim.x = 32 -> warp_size
// blockDim.y = 32, 也就是一个block 32个warp
// dim3 block(32, 32);
// threadIdx.y表示block内的warp_id
// BLOCK_DIM_Y代表每个block的warp数目
/*
第0个元素：[i, 0, j]位置
第1个元素：[i, 1, j]位置
第19个元素：[i, 19, j]位置
(tid + idx * BLOCK_DIM_x) * stride得到的是在axis索引的线性offset
也就是我们还需要i 和 j
i 也就是 (blockIdx.x * blockDim.y + threadIdx.y) / stride
j 也就是 (blockIdx.x * blockDim.y + threadIdx.y) % stride
然后i转化为线性也就是 i * stride * dimsize
j直接加上就好

*/
template <int elemPerThread, int BLOCK_DIM_Y, int BLOCK_DIM_X, typename T>
__global__ void Softmax_warp_impl(const T *x, T *y, int stride, int dimsize, int otherdim_size) {
    float dataPerThread[elemPerThread];
    int global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int group_offset = global_warp_id % stride + (global_warp_id - global_warp_id % stride) * dimsize;
    int tid = threadIdx.x;
    if (global_warp_id >= otherdim_size) {
        return;
    }
    __shared__ float group_max[BLOCK_DIM_X];
    __shared__ float group_sum[BLOCK_DIM_X];
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    for (int i = 0; tid + i * BLOCK_DIM_X < dimsize; i++) {
        dataPerThread[i] = static_cast<float>(x[(tid + i * BLOCK_DIM_X) * stride + group_offset]);
        thread_max = max(thread_max, dataPerThread[i]);
    }

    thread_max = warpReduce<float, MaxOp>(thread_max);
    if (tid == 0) {
        group_max[threadIdx.y] = thread_max;
    }

    for (int i = 0; tid + i * BLOCK_DIM_X < dimsize; i++) {
        dataPerThread[i] = __expf(dataPerThread[i] - group_max[threadIdx.y]);
        thread_sum += dataPerThread[i];
    }

    thread_sum = warpReduce<float, SumOp>(thread_sum);
    if (tid == 0) {
        group_sum[threadIdx.y] = thread_sum;
    }

    for (int i = 0; tid + i * BLOCK_DIM_X < dimsize; i++) {
        y[(tid + i * BLOCK_DIM_X) * stride + group_offset] = static_cast<T>(dataPerThread[i] * __fdividef(1.0f, group_sum[threadIdx.y]));
    }
}

template <int elemPerThread, int BLOCK_DIM, typename T>
__launch_bounds__(BLOCK_DIM)
    __global__ void Softmax_block_impl(const T *x, T *y, int stride, int dimsize, int otherdim_size) {
    //  remain = dimsize - BLOCK_DIM * elemPerThread
    int tid = threadIdx.x;
    int block_offset = (blockIdx.x - blockIdx.x % stride) * dimsize + blockIdx.x % stride;
    int remain = dimsize - (BLOCK_DIM - 1) * elemPerThread; // 🔧 修正：最后线程处理的元素数

    MD md_partial;
    md_partial.max = -INFINITY;
    md_partial.sum = 0.0f;
    MD input;
    // tid = [0, BLOCK_DIM - 1], 所以最后一个线程处理余数部分
    if (tid < BLOCK_DIM - 1) {
#pragma unroll
        for (int i = 0; i < elemPerThread; i++) {
            int index = (tid * elemPerThread + i) * stride + block_offset;
            input.max = static_cast<float>(x[index]);
            input.sum = 1.0f;
            md_partial = reduce_for_md(md_partial, input);
        }
    } else {
#pragma unroll
        for (int i = 0; i < remain; i++) {
            int index = ((BLOCK_DIM - 1) * elemPerThread + i) * stride + block_offset;
            input.max = static_cast<float>(x[index]);
            input.sum = 1.0f;
            md_partial = reduce_for_md(md_partial, input);
        }
    }
    typedef cub::BlockReduce<MD, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;
    MD md_block = BlockReduce(temp_storage).Reduce(md_partial, reduce_for_md);
    if (threadIdx.x == 0) {
        md_total = md_block;
    }
    __syncthreads();
    if (tid < BLOCK_DIM - 1) {
        for (int i = 0; i < elemPerThread; i++) {
            int index = (tid * elemPerThread + i) * stride + block_offset;
            y[index] = static_cast<T>(__expf(static_cast<float>(x[index]) - md_total.max) * __fdividef(1.0f, md_total.sum));
        }
    } else {
        for (int i = 0; i < remain; i++) {
            int index = ((BLOCK_DIM - 1) * elemPerThread + i) * stride + block_offset;
            y[index] = static_cast<T>(__expf(static_cast<float>(x[index]) - md_total.max) * __fdividef(1.0f, md_total.sum));
        }
    }
}

template <typename T>
infiniStatus_t softmax_dispatch(const op::softmax::SoftmaxInfo &info, void *y, const void *x, void *stream) {
    int dimsize = info.dimsize;
    int stride = info.stride;
    int otherdim_size = info.otherdim_size;
    if (dimsize <= 1024) {
        dim3 block(32, 32); // BLOCK_DIM_X=32, BLOCK_DIM_Y=4
        int num_blocks = (otherdim_size + block.y - 1) / block.y;
        dim3 grid(num_blocks, 1, 1);
        int elemPerThread = (dimsize + 31) / 32; // 计算每个线程需要处理的元素数
        elemPerThread = min(elemPerThread, 32);  // 限制最大值
        if (elemPerThread <= 1) {
            Softmax_warp_impl<1, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 2) {
            Softmax_warp_impl<2, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 4) {
            Softmax_warp_impl<4, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 8) {
            Softmax_warp_impl<8, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 16) {
            Softmax_warp_impl<16, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else {
            Softmax_warp_impl<32, 32, 32, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        }
    } else if (dimsize > 1024) {
        int block_size = 1024;
        int elemPerThread = (dimsize + block_size - 1) / block_size; // 每个线程需要处理的元素数
        elemPerThread = min(elemPerThread, 32);                      // 限制最大值为32
        dim3 block(block_size);
        dim3 grid(otherdim_size);
        if (elemPerThread <= 1) {
            Softmax_block_impl<1, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 2) {
            Softmax_block_impl<2, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 4) {
            Softmax_block_impl<4, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 8) {
            Softmax_block_impl<8, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else if (elemPerThread <= 16) {
            Softmax_block_impl<16, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        } else {
            Softmax_block_impl<32, 1024, T>
                <<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    reinterpret_cast<const T *>(x), reinterpret_cast<T *>(y), stride, dimsize, otherdim_size);
        }
    }
    return INFINI_STATUS_SUCCESS;
}

#endif // __SOFTMAX_CUDA_KERNEL_H__