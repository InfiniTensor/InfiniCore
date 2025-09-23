#ifndef __LOGSOFTMAX_KERNEL_CUH__
#define __LOGSOFTMAX_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>
#include <type_traits>

template <unsigned int BLOCK_SIZE, typename Tdata_out, typename Tdata_in, typename Tcompute>
__device__ void logSoftmaxKernel(
    Tdata_out *y, const Tdata_in *x,
    size_t batch_size, size_t probs_size, size_t ndim, size_t seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_p,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_p,
    ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
    ptrdiff_t x_stride_0, ptrdiff_t x_stride_1) {

    typedef cub::BlockReduce<Tcompute, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ Tcompute shared_max_val;
    __shared__ Tcompute shared_sum_exp;

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    // Calculate correct memory offsets for 3D tensors
    ptrdiff_t y_offset, x_offset;
    if (ndim == 3) {
        // For 3D tensors, convert linear batch index back to 2D indices
        ptrdiff_t batch_dim_idx = batch_idx / seq_len;
        ptrdiff_t seq_dim_idx = batch_idx % seq_len;
        y_offset = batch_dim_idx * y_stride_0 + seq_dim_idx * y_stride_1;
        x_offset = batch_dim_idx * x_stride_0 + seq_dim_idx * x_stride_1;
    } else {
        // For 2D tensors, use the flattened strides
        y_offset = batch_idx * y_stride_b;
        x_offset = batch_idx * x_stride_b;
    }

    const Tdata_in *x_batch = x + x_offset;
    Tdata_out *y_batch = y + y_offset;

    // Find maximum value for numerical stability
    Tcompute max_val = static_cast<Tcompute>(-INFINITY);
    for (int i = tid; i < probs_size; i += BLOCK_SIZE) {
        if (i < probs_size) { // Add boundary check
            Tcompute val = static_cast<Tcompute>(x_batch[i * x_stride_p]);
            if constexpr (std::is_same_v<Tcompute, float>) {
                max_val = fmaxf(max_val, val);
            } else {
                max_val = fmax(max_val, val);
            }
        }
    }
    max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    if (tid == 0) {
        shared_max_val = max_val;
    }
    __syncthreads();

    // Compute sum of exp(x - max)
    Tcompute sum_exp = static_cast<Tcompute>(0.0);
    for (int i = tid; i < probs_size; i += BLOCK_SIZE) {
        if (i < probs_size) { // Add boundary check
            Tcompute val = static_cast<Tcompute>(x_batch[i * x_stride_p]);
            if constexpr (std::is_same_v<Tcompute, float>) {
                sum_exp += expf(val - shared_max_val);
            } else {
                sum_exp += exp(val - shared_max_val);
            }
        }
    }
    sum_exp = BlockReduce(temp_storage).Sum(sum_exp);
    if (tid == 0) {
        shared_sum_exp = sum_exp;
    }
    __syncthreads();

    // Compute log_softmax = x - max - log(sum_exp)
    Tcompute log_sum_exp;
    if constexpr (std::is_same_v<Tcompute, float>) {
        log_sum_exp = logf(shared_sum_exp);
    } else {
        log_sum_exp = log(shared_sum_exp);
    }
    for (int i = tid; i < probs_size; i += BLOCK_SIZE) {
        if (i < probs_size) { // Add boundary check
            Tcompute val = static_cast<Tcompute>(x_batch[i * x_stride_p]);
            Tcompute result = val - shared_max_val - log_sum_exp;
            y_batch[i * y_stride_p] = static_cast<Tdata_out>(result);
        }
    }
}

template <unsigned int BLOCK_SIZE, typename Tdata_out, typename Tdata_in, typename Tcompute>
__global__ void logSoftmax(
    Tdata_out *y, const Tdata_in *x,
    size_t batch_size, size_t probs_size, size_t ndim, size_t seq_len,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_p,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_p,
    ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
    ptrdiff_t x_stride_0, ptrdiff_t x_stride_1) {
    logSoftmaxKernel<BLOCK_SIZE, Tdata_out, Tdata_in, Tcompute>(y, x, batch_size, probs_size, ndim, seq_len, y_stride_b, y_stride_p, x_stride_b, x_stride_p, y_stride_0, y_stride_1, x_stride_0, x_stride_1);
}

#endif // __LOGSOFTMAX_KERNEL_CUH__
