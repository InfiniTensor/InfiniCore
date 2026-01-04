#ifndef __CROSS_ENTROPY_KERNEL_CUH__
#define __CROSS_ENTROPY_KERNEL_CUH__

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../reduce/cuda/reduce.cuh"

// Tdata: Logits 的数据类型 (half, float...)
// Tidx: Target 的数据类型 (int32_t, int64_t)
// Tcompute: 计算使用的累加类型 (通常 float)
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx, typename Tcompute = float>
__device__ void crossEntropyKernel(
    Tdata *y_,              // Output: Loss [Outer]
    const Tdata *x_,        // Input: Logits [Outer, Vocab]
    const void *target_,    // Input: Labels [Outer]
    size_t outer_size,      // Batch * SeqLen
    size_t vocab_size,      // Vocab Size
    ptrdiff_t x_stride)     // Logits Stride
{
    // 每个 Block 处理一行 (Row)
    size_t row_idx = blockIdx.x;
    if (row_idx >= outer_size) return;

    // 获取当前行的输入输出指针
    const Tdata *x = x_ + row_idx * x_stride;
    const Tidx *target = reinterpret_cast<const Tidx*>(target_);
    
    // 获取当前行的 Label
    Tidx label = target[row_idx];
    
    // ----------------------------------------------------------------
    // 1. [Reduce] Find Max Value (为了数值稳定性)
    // ----------------------------------------------------------------
    // reduce_op::max 只保证 threadIdx.x==0 的返回值正确，因此需要一次显式广播
    Tdata max_val_raw = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(x, vocab_size);
    __shared__ Tcompute max_val_shared;
    if (threadIdx.x == 0) {
        max_val_shared = static_cast<Tcompute>(max_val_raw);
    }
    __syncthreads();
    Tcompute max_val = max_val_shared;

    // ----------------------------------------------------------------
    // 2. [Reduce] Compute Sum of Exp(x - max)
    // ----------------------------------------------------------------
    Tcompute thread_sum = 0.0f;
    for (size_t col = threadIdx.x; col < vocab_size; col += BLOCK_SIZE) {
        Tcompute val = static_cast<Tcompute>(x[col]);
        thread_sum += expf(val - max_val);
    }
    
    // Warp Reduce
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Block Reduce via Shared Memory
    static __shared__ Tcompute shared_sum[32]; // max 1024 threads / 32
    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;
    
    if (lane == 0) {
        shared_sum[warp] = thread_sum;
    }
    __syncthreads();
    
    Tcompute block_sum = 0.0f;
    if (warp == 0) {
        // 将 Warp 结果累加
        if (lane < (BLOCK_SIZE + warpSize - 1) / warpSize) {
             block_sum = shared_sum[lane];
        }
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }
    // 此时 lane 0 (threadIdx.x == 0) 拥有了整个 Block 的 SumExp
    
    // ----------------------------------------------------------------
    // 3. [Scalar] Compute Final Loss
    // ----------------------------------------------------------------
    if (threadIdx.x == 0) {
        Tcompute log_term = logf(block_sum) + max_val;
        
        Tcompute target_logit = 0.0f;
        // 确保 Label 不越界
        if (label >= 0 && static_cast<size_t>(label) < vocab_size) {
            target_logit = static_cast<Tcompute>(x[label]);
        } else {
            // 如果越界（例如 padding），通常设 Loss 为 0
            log_term = 0.0f; 
        }
        
        y_[row_idx] = static_cast<Tdata>(log_term - target_logit);
    }
}

#endif // __CROSS_ENTROPY_KERNEL_CUH__
