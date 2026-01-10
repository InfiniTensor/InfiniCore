#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cstdio>

#define MAX_EXPERTS 256

// 错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "[KERNEL ERROR] %s failed at line %d: %s\n", #call, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// =============================================================
// 1. Count Kernel (纯净版，带越界保护)
// =============================================================
__global__ void count_kernel_sota(
    const int32_t* __restrict__ topk_ids, 
    int32_t* __restrict__ expert_counts,  
    int total_tasks,
    int num_experts
) {
    extern __shared__ int32_t smem_counts[]; 
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;

    // 清空共享内存
    for (int i = tid; i < num_experts; i += blockDim.x) {
        if (i < num_experts) smem_counts[i] = 0;
    }
    __syncthreads();

    // 统计 (带边界检查)
    if (gid < total_tasks) {
        int expert_id = topk_ids[gid];
        // 【关键保护】非法ID直接忽略，防止 crash
        if (expert_id >= 0 && expert_id < num_experts) {
            unsigned int mask = __match_any_sync(__activemask(), expert_id);
            int leader = __ffs(mask) - 1; 
            int lane_id = tid % 32;
            if (lane_id == leader) {
                int agg_count = __popc(mask);
                atomicAdd(&smem_counts[expert_id], agg_count);
            }
        }
    }
    __syncthreads();

    // 写回全局内存
    for (int i = tid; i < num_experts; i += blockDim.x) {
        int count = smem_counts[i];
        if (count > 0) {
            atomicAdd(&expert_counts[i], count);
        }
    }
}

// =============================================================
// 2. Sort Launch (纯净版)
// =============================================================
void launch_moe_sort(
    const int32_t* topk_ids,
    int32_t* expert_counts,   
    int32_t* expert_offsets, 
    int num_tokens,
    int top_k,
    int num_experts,
    cudaStream_t stream
) {
    int total_tasks = num_tokens * top_k;
    int block_size = 256;
    int grid_size = (total_tasks + block_size - 1) / block_size;

    // 1. 清零 Counts (必须覆盖 num_experts + 1)
    CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, (num_experts + 1) * sizeof(int32_t), stream));
    
    // 2. 计算共享内存大小 (这一步绝不能省)
    size_t smem_size = (num_experts + 1) * sizeof(int32_t);
    
    // 3. 启动 Kernel
    count_kernel_sota<<<grid_size, block_size, smem_size, stream>>>(
        topk_ids, expert_counts, total_tasks, num_experts
    );

    // 4. CUB Scan (前缀和)
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    // 查询所需显存
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                  expert_counts, expert_offsets, 
                                  num_experts + 1, stream);
    
    // 分配临时显存 (使用同步 malloc 确保稳定)
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // 执行 Scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, 
                                  expert_counts, expert_offsets, 
                                  num_experts + 1, stream);
                                  
    // 释放
    CUDA_CHECK(cudaFree(d_temp_storage));
}

// =============================================================
// 3. Permute Kernel (纯净版，带越界保护)
// =============================================================
__global__ void permute_kernel(
    const float* __restrict__ input,           
    const int32_t* __restrict__ topk_ids,      
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ running_counters,    
    float* __restrict__ sorted_input,          
    int32_t* __restrict__ sorted_row_map,      
    float* __restrict__ sorted_weights,
    int num_tokens,
    int top_k,
    int hidden_dim,
    int num_experts
) {
    int total_tasks = num_tokens * top_k;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_tasks) return;

    int token_idx = tid / top_k;
    int expert_id = topk_ids[tid];

    // 【关键保护】防止非法ID导致读取 offsets 越界
    if (expert_id < 0 || expert_id >= num_experts) return;

    int base_offset = expert_offsets[expert_id];
    int my_rank = atomicAdd(&running_counters[expert_id], 1);
    int target_row = base_offset + my_rank;

    if (sorted_row_map) sorted_row_map[target_row] = token_idx;
    if (sorted_weights) sorted_weights[target_row] = topk_weights[tid];

    const float* src_ptr = input + token_idx * hidden_dim;
    float* dst_ptr = sorted_input + target_row * hidden_dim;

    // float4 优化拷贝
    const float4* src_vec = (const float4*)src_ptr;
    float4* dst_vec = (float4*)dst_ptr;
    int vec_len = hidden_dim / 4;
    
    for (int i = 0; i < vec_len; ++i) dst_vec[i] = src_vec[i];
    
    // 处理剩余部分
    for (int i = vec_len * 4; i < hidden_dim; ++i) dst_ptr[i] = src_ptr[i];
}

void launch_moe_permute(
    const float* input,
    const int32_t* topk_ids,
    const float* topk_weights,
    const int32_t* expert_offsets,
    float* sorted_input,
    int32_t* sorted_row_map,
    float* sorted_weights,
    int32_t* expert_counts, 
    int num_tokens,
    int top_k,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens * top_k + block_size - 1) / block_size;

    // 复用 expert_counts 作为 running_counters，必须清零
    CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, (num_experts + 1) * sizeof(int32_t), stream));

    permute_kernel<<<grid_size, block_size, 0, stream>>>(
        input, topk_ids, topk_weights, expert_offsets, expert_counts, 
        sorted_input, sorted_row_map, sorted_weights,
        num_tokens, top_k, hidden_dim, num_experts
    );
}

// =============================================================
// 4. Reduce Kernel (纯净版)
// =============================================================
__global__ void reduce_kernel(
    const float* __restrict__ sorted_output,
    const int32_t* __restrict__ sorted_row_map,
    const float* __restrict__ sorted_weights,
    float* __restrict__ final_output,
    int total_tasks,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = total_tasks * hidden_dim;
    
    if (tid >= total_elements) return;

    int row = tid / hidden_dim;     
    int col = tid % hidden_dim;     

    int original_token_idx = sorted_row_map[row];
    float weight = sorted_weights[row];
    float val = sorted_output[tid];

    // 加权求和写回原位置
    float* target_ptr = final_output + original_token_idx * hidden_dim + col;
    atomicAdd(target_ptr, val * weight);
}

void launch_moe_reduce(
    float* sorted_output,
    int32_t* sorted_row_map,
    float* sorted_weights,
    float* final_output,
    int num_tokens,
    int top_k,
    int hidden_dim,
    cudaStream_t stream
) {
    int total_tasks = num_tokens * top_k; 
    int total_elements = total_tasks * hidden_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    reduce_kernel<<<grid_size, block_size, 0, stream>>>(
        sorted_output, sorted_row_map, sorted_weights, final_output,
        total_tasks, hidden_dim
    );
}