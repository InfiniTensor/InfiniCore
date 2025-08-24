#ifndef __RMS_NORM_BACKWARD_KERNEL_CUH__
#define __RMS_NORM_BACKWARD_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"

namespace op::rms_norm_backward::cuda {

/**
 * @brief RMS Norm backward kernel for computing gradients
 * 
 * This kernel computes the backward pass for RMS normalization, calculating
 * gradients for both input (grad_x) and weight (grad_w) tensors.
 * 
 * @tparam BLOCK_SIZE Number of threads per block
 * @tparam Tdata Data type for input/output tensors
 * @tparam Tcompute Computation type for intermediate calculations
 * @tparam Tweight Weight tensor data type
 */
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute, typename Tweight = Tdata>
__global__ void rmsNormBackwardKernel(
    Tdata * grad_x,
    Tcompute * grad_w_cuda,
    const Tdata * grad_y,
    const Tdata * x,
    const Tweight * w,
    size_t ndim,
    size_t batch_size,
    size_t norm_size,
    const ptrdiff_t *__restrict__ grad_x_strides,
    const size_t *__restrict__ shape,
    const ptrdiff_t *__restrict__ grad_y_strides,
    const ptrdiff_t *__restrict__ x_strides,
    ptrdiff_t w_stride,
    float epsilon
) {
    // Calculate tensor pointers for current batch - 完全按照CPU算法
    size_t batch_index = blockIdx.x;
    
    // 计算多维索引 - 与CPU实现完全一致
    size_t batch_indices[8]; // 假设最大8维
    size_t remaining = batch_index;
    for (int dim = ndim - 2; dim >= 0; --dim) {
        batch_indices[dim] = remaining % shape[dim];
        remaining /= shape[dim];
    }
    
    // 计算各个张量的batch偏移 - 与CPU实现完全一致
    size_t x_batch_offset = 0;
    size_t grad_y_batch_offset = 0;
    size_t grad_x_batch_offset = 0;
    for (size_t dim = 0; dim < ndim - 1; ++dim) {
        x_batch_offset += batch_indices[dim] * x_strides[dim];
        grad_y_batch_offset += batch_indices[dim] * grad_y_strides[dim];
        grad_x_batch_offset += batch_indices[dim] * grad_x_strides[dim];
    }
    
    auto grad_w_ptr = grad_w_cuda + batch_index;

    // 按照CPU算法计算RMS - 使用mean_square而不是rsqrt
    Tcompute sum_squares = Tcompute(0);
    for (size_t i = threadIdx.x; i < norm_size; i += BLOCK_SIZE) {
        size_t x_idx = x_batch_offset + i * x_strides[ndim - 1];
        Tcompute x_val = Tcompute(x[x_idx]);
        sum_squares += x_val * x_val;
    }
    
    // Reduce sum of squares across threads
    typedef cub::BlockReduce<Tcompute, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum_squares = BlockReduce(temp_storage).Sum(sum_squares);
    
    Tcompute norm_size_f = Tcompute(norm_size);

    // Shared memory for RMS and gradient accumulation
    __shared__ Tcompute rms;
    __shared__ Tcompute sum_grad_y_w_x;
    
    if (threadIdx.x == 0) {
        // 按照CPU算法计算RMS: sqrt(mean_square + epsilon)
        Tcompute mean_square = sum_squares / norm_size_f;
        rms = sqrtf(mean_square + Tcompute(epsilon));
        
        // 按照CPU算法计算sum(grad_y * w * x)
        sum_grad_y_w_x = Tcompute(0);
        for (size_t i = 0; i < norm_size; i++) {
            size_t x_idx = x_batch_offset + i * x_strides[ndim - 1];
            size_t grad_y_idx = grad_y_batch_offset + i * grad_y_strides[ndim - 1];
            size_t w_idx = i * w_stride;
            
            Tcompute gy = Tcompute(grad_y[grad_y_idx]);
            Tcompute w_val = Tcompute(w[w_idx]);
            Tcompute x_val = Tcompute(x[x_idx]);
            sum_grad_y_w_x += gy * w_val * x_val;
        }
    }
    __syncthreads();

    Tcompute rms_val = rms;
    Tcompute rms_squared = rms_val * rms_val;
    Tcompute rms_cubed = rms_squared * rms_val;
    Tcompute sum_val = sum_grad_y_w_x;

    // 按照CPU算法计算梯度
    for (size_t i = threadIdx.x; i < norm_size; i += BLOCK_SIZE) {
        size_t x_idx = x_batch_offset + i * x_strides[ndim - 1];
        size_t grad_y_idx = grad_y_batch_offset + i * grad_y_strides[ndim - 1];
        size_t grad_x_idx = grad_x_batch_offset + i * grad_x_strides[ndim - 1];
        size_t w_idx = i * w_stride;
        
        Tcompute gy = Tcompute(grad_y[grad_y_idx]);
        Tcompute w_val = Tcompute(w[w_idx]);
        Tcompute x_val = Tcompute(x[x_idx]);
        
        // RMSNorm grad_x计算: (w * grad_y) / rms - (x * sum(grad_y * w * x)) / (norm_size * rms³)
        Tcompute gx = (w_val * gy) / rms_val - (x_val * sum_val) / (norm_size_f * rms_cubed);
        grad_x[grad_x_idx] = Tdata(gx);
        
        // grad_w计算: (x * grad_y) / rms
        Tcompute gw = (x_val * gy) / rms_val;
        *(grad_w_ptr + i * batch_size) = Tcompute(gw);
    }
}

/**
 * @brief Kernel to sum up weight gradients across batches
 * 
 * This kernel reduces the weight gradients computed across all batches
 * to produce the final weight gradient tensor.
 * 
 * @tparam BLOCK_SIZE Number of threads per block
 * @tparam Tdata Data type for output gradient tensor
 * @tparam Tcompute Computation type for intermediate calculations
 */
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void sumUpGradWKernel(
    Tdata * grad_w,
    Tcompute * grad_w_cuda,
    size_t batch_size,
    ptrdiff_t grad_w_stride
) {
    size_t norm_index = blockIdx.x;
    
    // Reduce weight gradients across all batches for this feature dimension
    Tcompute sum_grad_w = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tcompute, Tcompute>(
        grad_w_cuda + norm_index * batch_size, batch_size
    );
    
    // Write the final gradient value
    if (threadIdx.x == 0) {
        *(grad_w + norm_index * grad_w_stride) = Tdata(sum_grad_w);
    }
}

} // namespace op::rms_norm_backward::cuda

#endif // __RMS_NORM_BACKWARD_KERNEL_CUH__