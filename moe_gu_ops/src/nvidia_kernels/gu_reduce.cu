#include "nvidia_kernels_moe.h"
#include "cuda_utils.h"

// ========================================================================
// Kernel: Reduce (加权还原) - float4 向量化优化版
// ========================================================================
__global__ void reduce_kernel_opt(
    const float* __restrict__ sorted_output,    // [Total_Tasks, H]
    const int32_t* __restrict__ sorted_row_map, // [Total_Tasks]
    const float* __restrict__ sorted_weights,   // [Total_Tasks]
    float* __restrict__ final_output,           // [Num_Tokens, H]
    int total_tasks, // N * K
    int hidden_dim
) {
    // 策略：每个线程处理 4 个元素 (128 bit)
    // 这样的 grid 维度计算需要除以 4
    int vec_dim = hidden_dim / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. 处理向量化部分 (4的倍数)
    if (tid < total_tasks * vec_dim) {
        int row_idx = tid / vec_dim;         // 第几行
        int vec_idx = tid % vec_dim;         // 第几组 float4
        int col_idx = vec_idx * 4;           // 实际列号

        // 查户口
        int original_token_idx = sorted_row_map[row_idx];
        float weight = sorted_weights[row_idx];

        // 向量化读取 (Load 128-bit)
        // 强转指针为 float4*
        const float4* src_vec_ptr = (const float4*)sorted_output;
        float4 val_vec = src_vec_ptr[tid]; // 直接读 tid 位置的 float4

        // 目标基地址
        float* dst_base = final_output + original_token_idx * hidden_dim + col_idx;

        // 原子累加 (atomicAdd 不支持 float4，必须拆开)
        // 但读取指令减少了，依然有加速
        atomicAdd(dst_base + 0, val_vec.x * weight);
        atomicAdd(dst_base + 1, val_vec.y * weight);
        atomicAdd(dst_base + 2, val_vec.z * weight);
        atomicAdd(dst_base + 3, val_vec.w * weight);
    }

    // 2. 处理尾巴 (Remainder)
    int remainder = hidden_dim % 4;
    if (remainder > 0) {
        int row_idx = tid / vec_dim; // 近似映射
    }
}

// ========================================================================
// Host Launcher
// ========================================================================
void launch_moe_reduce(
    const float* sorted_output,
    const int32_t* sorted_row_map,
    const float* sorted_weights,
    float* final_output,
    int num_tokens,
    int top_k,
    int hidden_dim,
    cudaStream_t stream
) {
    int total_tasks = num_tokens * top_k;
    
    // 优先使用向量化版本
    if (hidden_dim % 4 == 0) {
        int vec_dim = hidden_dim / 4;
        long long total_threads = (long long)total_tasks * vec_dim;
        int block_size = 256;
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        reduce_kernel_opt<<<grid_size, block_size, 0, stream>>>(
            sorted_output, sorted_row_map, sorted_weights, final_output,
            total_tasks, hidden_dim
        );
    } else {
        long long total_elements = (long long)total_tasks * hidden_dim;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // reduce_kernel_scalar<<<...>>>(...); // 你之前的那个函数
        printf("Warning: Hidden dim not divisible by 4, running slow path.\n");
    }
}