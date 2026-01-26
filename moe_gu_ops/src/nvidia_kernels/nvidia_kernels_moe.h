#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 算子 1: 排序与偏移计算 (Count + CUB Scan)
void launch_moe_sort(
    const int32_t* topk_ids,
    int32_t* expert_counts,   
    int32_t* expert_offsets, 
    int num_tokens,
    int top_k,
    int num_experts,
    cudaStream_t stream
);

// 算子 2: 数据搬运 (Permutation)
void launch_moe_permute(
    const float* input,
    const int32_t* topk_ids,
    const float* topk_weights, // [Input] 原始权重
    const int32_t* expert_offsets,
    float* sorted_input,
    int32_t* sorted_row_map,
    float* sorted_weights,     // [Output] 排序后的权重
    int32_t* expert_counts, 
    int num_tokens,
    int top_k,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
);

void launch_moe_reduce(
    const float* sorted_output,
    const int32_t* sorted_row_map,
    const float* sorted_weights,
    float* final_output,
    int num_tokens,
    int top_k,
    int hidden_dim,
    cudaStream_t stream
);