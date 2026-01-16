#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>
#include <numeric>

#define MAX_EXPERTS 256

// =================================================================================
// 1. 辅助宏与工具
// =================================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// =================================================================================
// 2. 你的 Kernel 实现 (直接复制粘贴你的代码)
// =================================================================================

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

    for (int i = tid; i < num_experts; i += blockDim.x) {
        smem_counts[i] = 0;
    }
    __syncthreads();

    if (gid < total_tasks) {
        int expert_id = topk_ids[gid];
        // 简单的 Warp 聚合逻辑验证
        unsigned int active_mask = __activemask();
        unsigned int mask = __match_any_sync(active_mask, expert_id);
        int leader = __ffs(mask) - 1; 
        int lane_id = tid % 32;

        if (lane_id == leader) {
            int agg_count = __popc(mask);
            atomicAdd(&smem_counts[expert_id], agg_count);
        }
    }
    
    __syncthreads();

    for (int i = tid; i < num_experts; i += blockDim.x) {
        int count = smem_counts[i];
        if (count > 0) {
            atomicAdd(&expert_counts[i], count);
        }
    }
}

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

    CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream));
    
    count_kernel_sota<<<grid_size, block_size, num_experts * sizeof(int32_t), stream>>>(
        topk_ids, expert_counts, total_tasks, num_experts
    );
    
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, expert_counts, expert_offsets, num_experts + 1, stream);
    CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, expert_counts, expert_offsets, num_experts + 1, stream);
    CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));
}

__global__ void permute_kernel(
    const float* __restrict__ input,           
    const int32_t* __restrict__ topk_ids,      
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ running_counters,    
    float* __restrict__ sorted_input,          
    int32_t* __restrict__ sorted_row_map,      
    int num_tokens,
    int top_k,
    int hidden_dim
) {
    int total_tasks = num_tokens * top_k;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= total_tasks) return;

    int token_idx = tid / top_k;
    int expert_id = topk_ids[tid];

    int base_offset = expert_offsets[expert_id];
    int my_rank = atomicAdd(&running_counters[expert_id], 1);
    int target_row = base_offset + my_rank;

    sorted_row_map[target_row] = token_idx;

    const float* src_ptr = input + token_idx * hidden_dim;
    float* dst_ptr = sorted_input + target_row * hidden_dim;

    int vec_size = hidden_dim / 4;
    int remainder = hidden_dim % 4;
    
    const float4* src_vec = (const float4*)src_ptr;
    float4* dst_vec = (float4*)dst_ptr;

    for (int i = 0; i < vec_size; ++i) {
        dst_vec[i] = src_vec[i];
    }
    for (int i = 0; i < remainder; ++i) {
        int idx = vec_size * 4 + i;
        dst_ptr[idx] = src_ptr[idx];
    }
}

void launch_moe_permute(
    const float* input,
    const int32_t* topk_ids,
    const int32_t* expert_offsets,
    float* sorted_input,
    int32_t* sorted_row_map,
    int32_t* expert_counts, 
    int num_tokens,
    int top_k,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
) {
    int total_tasks = num_tokens * top_k;
    int block_size = 256;
    int grid_size = (total_tasks + block_size - 1) / block_size;

    CUDA_CHECK(cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream));

    permute_kernel<<<grid_size, block_size, 0, stream>>>(
        input, topk_ids, expert_offsets, expert_counts, 
        sorted_input, sorted_row_map, num_tokens, top_k, hidden_dim
    );
}

// =================================================================================
// 3. CPU 基准验证逻辑 (Ground Truth)
// =================================================================================
void verify_results(
    const std::vector<float>& h_input,
    const std::vector<int32_t>& h_topk_ids,
    const std::vector<int32_t>& gpu_offsets,
    const std::vector<float>& gpu_sorted_input,
    const std::vector<int32_t>& gpu_row_map,
    int num_tokens, int top_k, int hidden_dim, int num_experts
) {
    int total_tasks = num_tokens * top_k;

    // 1. CPU Count
    std::vector<int32_t> cpu_counts(num_experts, 0);
    for (int i = 0; i < total_tasks; ++i) {
        cpu_counts[h_topk_ids[i]]++;
    }

    // 2. CPU Offset
    std::vector<int32_t> cpu_offsets(num_experts + 1, 0);
    for (int i = 0; i < num_experts; ++i) {
        cpu_offsets[i + 1] = cpu_offsets[i] + cpu_counts[i];
    }

    // 验证 Offsets
    bool offset_ok = true;
    for (int i = 0; i <= num_experts; ++i) {
        if (cpu_offsets[i] != gpu_offsets[i]) {
            std::cout << "❌ Offset Mismatch at Expert " << i 
                      << ": CPU=" << cpu_offsets[i] << ", GPU=" << gpu_offsets[i] << std::endl;
            offset_ok = false;
        }
    }
    if (offset_ok) std::cout << "✅ Offsets Verification Passed!" << std::endl;

    // 3. CPU Permute
    std::vector<float> cpu_sorted_input(total_tasks * hidden_dim, 0.0f);
    std::vector<int32_t> cpu_row_map(total_tasks, 0);
    std::vector<int32_t> running_counters(num_experts, 0);

    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int task_idx = t * top_k + k;
            int expert_id = h_topk_ids[task_idx];
            
            int base = cpu_offsets[expert_id];
            int rank = running_counters[expert_id]++;
            int target_row = base + rank;

            // 记录 Row Map
            cpu_row_map[target_row] = t;

            // 搬运数据
            for (int h = 0; h < hidden_dim; ++h) {
                cpu_sorted_input[target_row * hidden_dim + h] = h_input[t * hidden_dim + h];
            }
        }
    }

    // 验证 Row Map (注意：多线程下的 row map 顺序对于同一个 Expert 内部可能是不确定的，
    // 但是在这个测试用例中，我们单线程生成数据，GPU 也是顺序 atomic，通常是一致的。
    // 如果不一致，我们要检查是否属于同一个 Expert。
    // 严格来说，只需验证 gpu_sorted_input 中的数据是否等于 input[gpu_row_map[i]] 且 expert id 匹配)
    
    // 我们采用宽松验证：验证 gpu_sorted_input 的值是否正确
    bool data_ok = true;
    for (int i = 0; i < total_tasks * hidden_dim; ++i) {
        float diff = std::abs(gpu_sorted_input[i] - cpu_sorted_input[i]);
        if (diff > 1e-5) {
            std::cout << "❌ Data Mismatch at index " << i 
                      << ": CPU=" << cpu_sorted_input[i] << ", GPU=" << gpu_sorted_input[i] << std::endl;
            data_ok = false;
            if (i > 10) break; // 防止刷屏
        }
    }

    if (data_ok) std::cout << "✅ Sorted Data Verification Passed!" << std::endl;
    else std::cout << "❌ Data Verification Failed." << std::endl;
}


// =================================================================================
// 4. Main
// =================================================================================
int main() {
    // 设置参数
    const int num_tokens = 16;   // 少量 Token 用于调试
    const int hidden_dim = 8;    // 必须是 4 的倍数以便测试 float4
    const int top_k = 2;
    const int num_experts = 4;
    const int total_tasks = num_tokens * top_k;

    std::cout << ">>> Running MoE Sort Test..." << std::endl;
    std::cout << "Tokens: " << num_tokens << ", Hidden: " << hidden_dim 
              << ", TopK: " << top_k << ", Experts: " << num_experts << std::endl;

    // 1. Host 准备数据
    std::vector<float> h_input(num_tokens * hidden_dim);
    std::vector<int32_t> h_topk_ids(total_tasks);

    // 初始化 Input: Token i 的第 j 个数为 i + 0.01*j
    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            h_input[i * hidden_dim + j] = (float)i + 0.01f * j;
        }
    }

    // 初始化 Indices: 简单的循环模式 0, 1, 2, 3...
    for (int i = 0; i < total_tasks; ++i) {
        h_topk_ids[i] = i % num_experts;
    }

    // 2. Device 分配内存
    float *d_input, *d_sorted_input;
    int32_t *d_topk_ids, *d_expert_counts, *d_expert_offsets, *d_sorted_row_map;

    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_ids, h_topk_ids.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_input, total_tasks * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_counts, num_experts * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_offsets, (num_experts + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_row_map, total_tasks * sizeof(int32_t)));

    // 3. 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_topk_ids, h_topk_ids.data(), h_topk_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // 4. 执行 Sort (Count + Scan)
    cudaStream_t stream = 0;
    launch_moe_sort(d_topk_ids, d_expert_counts, d_expert_offsets, num_tokens, top_k, num_experts, stream);
    
    // 5. 执行 Permute
    launch_moe_permute(d_input, d_topk_ids, d_expert_offsets, d_sorted_input, d_sorted_row_map, d_expert_counts, 
                       num_tokens, top_k, hidden_dim, num_experts, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 6. 拷回结果
    std::vector<int32_t> h_offsets_gpu(num_experts + 1);
    std::vector<float> h_sorted_input_gpu(total_tasks * hidden_dim);
    std::vector<int32_t> h_row_map_gpu(total_tasks);

    CUDA_CHECK(cudaMemcpy(h_offsets_gpu.data(), d_expert_offsets, h_offsets_gpu.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sorted_input_gpu.data(), d_sorted_input, h_sorted_input_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_row_map_gpu.data(), d_sorted_row_map, h_row_map_gpu.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // 7. 打印部分结果用于调试
    std::cout << "\n[GPU Offsets]: ";
    for (auto v : h_offsets_gpu) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "[GPU Row Map (First 10)]: ";
    for (int i=0; i<std::min(10, total_tasks); ++i) std::cout << h_row_map_gpu[i] << " ";
    std::cout << std::endl;

    // 8. 验证
    verify_results(h_input, h_topk_ids, h_offsets_gpu, h_sorted_input_gpu, h_row_map_gpu, num_tokens, top_k, hidden_dim, num_experts);

    // 清理
    cudaFree(d_input);
    cudaFree(d_topk_ids);
    cudaFree(d_sorted_input);
    cudaFree(d_expert_counts);
    cudaFree(d_expert_offsets);
    cudaFree(d_sorted_row_map);

    return 0;
}