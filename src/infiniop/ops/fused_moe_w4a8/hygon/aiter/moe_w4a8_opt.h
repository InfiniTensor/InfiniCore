
// SPDX-License-Identifier: MIT
// Adapted from ROCm/AITER for InfiniCore's Hygon backend.

#ifndef MOE_W4A8_OPT_HIP_H
#define MOE_W4A8_OPT_HIP_H

#include "moe_w4a8_config.h"
#include "moe_w4a8_utils.h"

template <
    typename scalar_t,
    typename Element,
    uint16_t WARP_NUM,
    uint16_t BLOCK_SIZE_M,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    uint16_t WARP_M,
    uint16_t WARP_N,
    uint16_t WARP_K,
    uint16_t GROUP_N,
    uint16_t GROUP_K,
    int STAGES,
    bool mul_topk_weight>
__global__ void __launch_bounds__(512, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP(
    // const Element* __restrict__ input,
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk) {
    const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.x; // pid_n
    const int bidz = blockIdx.y; // pid_k

    // int bidx = blockIdx.x; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // int bidy = blockIdx.y; // pid_n
    // int bidz = blockIdx.z; // pid_k
    // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 4, bidx, bidy);

    if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) {
        return;
    }
    // constexpr int STAGES = 4;
    const uint32_t input_offset = bidz * BLOCK_SIZE_K; // 输入k方向分块的位置
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx];                           // 专家的索引
    const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id; // 这个是对应专家的weight偏移
    // const uint64_t qweight_offset = expert_offset + bidy * size_k * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块
    const uint64_t qweight_offset = expert_offset + bidy * BLOCK_SIZE_N / 32 * 32 * 32;                   // 具体偏移到对应专家的weight的某一个小的分块 BLOCK_SIZE_N 必须为32的倍数
    const uint32_t output_offset = bidy * BLOCK_SIZE_N;                                                   // 计算之后是mxn,这应该是计算输出n方向的位置
    const uint64_t weight_scale_offset = stride_bse /* * stride_bsn */ * expert_id + bidy * BLOCK_SIZE_N; // 具体偏移到对应专家的weight的某一个小的分块

    auto g_input = input;
    auto g_input_scale = input_scale; // 配置全局显存信息
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1) { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16

        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            zero_element_8.data[i] = 0;
        }

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            int topk_ids = (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;
            // const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];

            if (topk_ids < real_topk) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[token_index * size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
    int lane_id = threadIdx.x & 63;                            // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];                             // 声明lds信息
    Element *input_lds = (Element *)&(smem);                      // decode这里没用
    Element *qweight_lds = input_lds;                             // decode这里没用
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds
    // const int32_t * sorted_token_ids_offset = sorted_token_ids;

    // union_vec_opt<Element, 8> A_reg[WARP_M / mfma_m * STAGES][WARP_K/mfma_k];
    // union_vec_opt<Element, 8> B_reg[WARP_N / mfma_n][WARP_K/mfma_k];
    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][STAGES];

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset;     // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
    intx4 C_reg[1][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    float *a_scale_ptr_arr[WARP_M / mfma_m];
    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++) {
        // int sorted_token_offset = sorted_token_ids_offset[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
        const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + row_id, int(sorted_token_lens - 1))];
        // uint32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;

        a_scale_ptr_arr[idx] = input_scale + std::min(sorted_token_ids_element / top_k, size_m - 1) * stride_asm; // 计算M方向的偏移
    }

    float b_scale[(WARP_N / mfma_n) * 4];
    // vec<float,4>::type b_scale[(WARP_N / mfma_n) ];

    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func<64, float>(b_scale_ptr);
#pragma unroll
    for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
#pragma unroll
        for (int it = 0; it < 2; it++) {
            // #pragma unroll
            // for(int it = 0 ;it < 2 ; it++){
            // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
            // inline_buffer_load_dword(b_scale[min_tile_n*4+0],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+0);
            // inline_buffer_load_dword(b_scale[min_tile_n*4+1],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+4);
            // inline_buffer_load_dword(b_scale[min_tile_n*4+2],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+8);
            // inline_buffer_load_dword(b_scale[min_tile_n*4+3],col_id,b_scale_ptr_prepared,min_tile_n * mfma_n+12);
            // b_scale[min_tile_n*4+0] = b_scale_ptr[min_tile_n * mfma_n+0 + col_id];
            // b_scale[min_tile_n*4+1] = b_scale_ptr[min_tile_n * mfma_n+4 + col_id];
            // b_scale[min_tile_n*4+2] = b_scale_ptr[min_tile_n * mfma_n+8 + col_id];
            // b_scale[min_tile_n*4+3] = b_scale_ptr[min_tile_n * mfma_n+12 + col_id];

            b_scale[(min_tile_n * 2 + it) * 4 + 0] = b_scale_ptr[(min_tile_n * 32 + it) + 0 + col_id * 2];
            b_scale[(min_tile_n * 2 + it) * 4 + 1] = b_scale_ptr[(min_tile_n * 32 + it) + 8 + col_id * 2];
            b_scale[(min_tile_n * 2 + it) * 4 + 2] = b_scale_ptr[(min_tile_n * 32 + it) + 16 + col_id * 2];
            b_scale[(min_tile_n * 2 + it) * 4 + 3] = b_scale_ptr[(min_tile_n * 32 + it) + 24 + col_id * 2];

            //  if(threadIdx.x ==0 && blockIdx.x ==0 && blockIdx.y ==0 && blockIdx.z == 0){
            //     printf("b_scale[(min_tile_n * 2 + it) *4+0]: %f", b_scale[(min_tile_n * 2 + it) *4+0]);
            //     printf("b_scale[(min_tile_n * 2 + it) *4+1]: %f", b_scale[(min_tile_n * 2 + it) *4+1]);
            //     printf("b_scale[(min_tile_n * 2 + it) *4+2]: %f", b_scale[(min_tile_n * 2 + it) *4+2]);
            //     printf("b_scale[(min_tile_n * 2 + it) *4+3]: %f", b_scale[(min_tile_n * 2 + it) *4+3]);

            //   }
            // }
        }
    }

    {
        // gemm_nt_first_stage_decode
        //  gemm_nt_two_stage_decode<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>
        //  (g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids_offset, sorted_token_lens, expert_id, bidx);

        gemm_nt_marlin_decode_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, Element>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx);
    }
    // float value_tmp =0;
    // bhalf_t res = 0;
    // float value_ori = 0;
    // save output to lds
    //     float *a_scale = a_scale_ptr_arr[0];

    // if(threadIdx.x ==0 && blockIdx.x ==0 && blockIdx.y ==0 && blockIdx.z == 0){
    //           printf("C_reg : %f a_scale : %f b_scale: %f", C_reg[0][0],a_scale[0],b_scale[0]);
    //         }
    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    {

        // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            for (int i = 0; i < 4; i++) {
                int m_idx = min_tile_m * 16 + i * 4 + col_id;
                int it = m_idx / 4;

                sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
            }
        }
    }

#if 1
    if (warp_k_id == 0 || warp_k_num == 1) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            float *a_scale = a_scale_ptr_arr[min_tile_m];
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
#pragma unroll
                for (int it = 0; it < 2; it++) {
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        // float *b_scale = b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4;
                        // if(threadIdx.x == 0 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)){
                        //     printf("%d ******************************C_reg[0][0 ][0] : %f \n",threadIdx.x, C_reg[0][0 ][0] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][1] : %f \n",threadIdx.x, C_reg[0][0 ][1] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][2] : %f \n",threadIdx.x, C_reg[0][0 ][2] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][3] : %f \n",threadIdx.x, C_reg[0][0 ][3] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);

                        //   }
                        //   if(threadIdx.x == 1 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)){
                        //     printf("%d ******************************C_reg[0][0 ][0] : %f \n",threadIdx.x, C_reg[0][0 ][0] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][1] : %f \n",threadIdx.x, C_reg[0][0 ][1] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][2] : %f \n",threadIdx.x, C_reg[0][0 ][2] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][3] : %f \n",threadIdx.x, C_reg[0][0 ][3] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);

                        //   }
                        //   if(threadIdx.x == 16 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)){
                        //     printf("%d ******************************C_reg[0][0 ][0] : %f \n",threadIdx.x, C_reg[0][0 ][0] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][1] : %f \n",threadIdx.x, C_reg[0][0 ][1] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][2] : %f \n",threadIdx.x, C_reg[0][0 ][2] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][3] : %f \n",threadIdx.x, C_reg[0][0 ][3] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);

                        //   }
                        //   if(threadIdx.x == 32 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)){
                        //     printf("%d ******************************C_reg[0][0 ][0] : %f \n",threadIdx.x, C_reg[0][0 ][0] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][1] : %f \n",threadIdx.x, C_reg[0][0 ][1] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][2] : %f \n",threadIdx.x, C_reg[0][0 ][2] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][3] : %f \n",threadIdx.x, C_reg[0][0 ][3] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);

                        //   }if(threadIdx.x == 48 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)){
                        //     printf("%d ******************************C_reg[0][0 ][0] : %f \n",threadIdx.x, C_reg[0][0 ][0] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][1] : %f \n",threadIdx.x, C_reg[0][0 ][1] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][2] : %f \n",threadIdx.x, C_reg[0][0 ][2] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);
                        //     printf("%d ******************************C_reg[0][0 ][3] : %f \n",threadIdx.x, C_reg[0][0 ][3] * a_scale[0] * b_scale[(min_tile_n * 2 + it )*4+reg_id]);

                        //   }
                        float value = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n * 2 + it][reg_id] * a_scale[0] * b_scale[(min_tile_n * 2 + it) * 4 + reg_id];

                        int index = min_tile_m * mfma_m * BLOCK_SIZE_N + (min_tile_n)*32 + warp_n_id * WARP_N + (lane_id & 15) * BLOCK_SIZE_N + reg_id * 8 + lane_id / 16 * 2 + it + (min_tile_m * mfma_m + (lane_id % 16)) / 2 * 2 /*padding*/;
                        // value_ori = value;
                        output_lds[index] = b32_to_b16<scalar_t>(value);
                        // value_tmp = output_lds[index];
                        // res = f32_to_bf16(value);
                    }
                }
            }
        }
    }

    __syncthreads();
    // float res_float = bf16_to_f32(res);
    // value_tmp = bf16_to_f32(output_lds[0]);
    // if((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&(threadIdx.x == 0) ){
    //   printf("****************************value_ori:%f",value_ori);
    //   printf("****************************value_tmp:%f",value_tmp);
    // //   printf("****************************value_tmp_reg:%f",res_float);

    // }

    // float value_tmp_global =0;
    {
        // 最大化利用dwordx4 通用
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        // using vecname = vec_element_8<scalar_t>;
        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            // const int32_t token_index = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            // int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            // int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
            // int token_index = token_ids * real_topk /* top_k */ + topk_ids;
            if (sorted_token_ids_element < size_m * top_k) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[sorted_token_ids_element * size_n + n_idx * 8]) = *reinterpret_cast<vec_element_8<scalar_t> *>(&output_lds[m_idx * BLOCK_SIZE_N + n_idx * 8 + m_idx / 2 * 2 /*padding*/]);
                // value_tmp_global = bf16_to_f32(g_output[0]) ;
            }
        }
    }
    // __syncthreads();
    // if((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0) &&(threadIdx.x == 0) ){
    //   printf("****************************value_tmp_global:%f",value_tmp_global);
    // }
#else

    if (warp_k_id == 0 || warp_k_num == 1) {

        scalar_t value[WARP_M / mfma_m][4][WARP_N / mfma_n];

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
            float *a_scale = a_scale_ptr_arr[min_tile_m];

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
#pragma unroll
                for (int it = 0; it < 2; it++) {

#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        if (threadIdx.x == 0 && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
                            printf("******************************C_reg[0][0 ][0] : %d ", C_reg[0][0][0]);
                            printf("******************************C_reg[0][0 ][0] : %d ", C_reg[0][0][1]);
                            printf("******************************C_reg[0][0 ][0] : %d ", C_reg[0][0][2]);
                            printf("******************************C_reg[0][0 ][0] : %d ", C_reg[0][0][3]);
                        }
                        float value_tmp = C_reg[0][min_tile_m * WARP_N / mfma_n + min_tile_n * 2 + it][reg_id] * a_scale[0] * b_scale[(min_tile_n * 2 + it) * 4 + reg_id];

                        value[min_tile_m][reg_id][min_tile_n * 2 + it] = b32_to_b16<scalar_t>(value_tmp);
                    }
                }
            }
        }

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
                int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N;
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++) {
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (sorted_token_ids_element_store[it] < size_m * top_k) {

                        *(vec_element_2<scalar_t> *)(&g_output[/* token_index_store[it] * size_n */ sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[min_tile_m][reg_id][min_tile_n * 2]);
                    }
                }
            }
        }
    }

#endif
}

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk) {
    // const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidy = blockIdx.y; // pid_n
    // const int bidz = blockIdx.x; // pid_k

    const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.y; // pid_n
    const int bidz = blockIdx.x; // pid_k

    // block swizzle todo(precision wrong)
    //  constexpr int CU_NUMS = 80;
    //  constexpr int block_num_per_cu = 3;
    //  const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y;
    //  const int block_swizzle_x = bid % CU_NUMS ;
    //  const int block_swizzle_y = bid / CU_NUMS;

    // const int bidy = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) % gridDim.y  ;    // pid_n
    // const int bidx = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) / gridDim.y; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidz = blockIdx.x; // pid_k

    // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // int bidy = blockIdx.y; // pid_n
    // int bidz = blockIdx.x; // pid_k
    // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

    // constexpr int STAGES = 2;

    // if(blockIdx.x == 0 && threadIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0){
    //   printf("**********************************************size_n: %d size_k:%d", size_n,size_k);
    // }

    // uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    // if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

    if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) {
        return;
    }

    const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

    const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id; // 这个是对应专家的weight偏移

    // auto g_input = input;
    auto g_input = input;
    auto g_input_scale = input_scale; // 配置全局显存信息

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
    int lane_id = threadIdx.x & 63;                            // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];                             // 声明lds信息
    Element *input_lds = (Element *)&(smem);                      // decode这里没用
    Element *qweight_lds = input_lds;                             // decode这里没用
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds
    // float* output_lds_float = (float*)&(smem); // 重复使用lds,留给output_lds

    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    float weight_dot_a_scale[WARP_M / mfma_m][4];

    // #pragma unroll
    // for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    //   for(int i = 0 ;i< 4;i++){
    //     int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
    //     int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    //     int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
    //     int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1)  ;
    //     float input_scale_value = *(input_scale + token_index_safe * stride_asm); //计算M方向的偏移
    //     weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;

    //   }
    // }

#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++) {
        for (int i = 0; i < 4; i++) {
            int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
            // int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            // int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
            float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
            weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
        }
    }

    constexpr int n_loop_num = 4;

    const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
    const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1) { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            zero_element_8.data[i] = 0;
        }

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;

            if (topk_ids < real_topk) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    {

        // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            for (int i = 0; i < 4; i++) {
                int m_idx = min_tile_m * 16 + i * 4 + col_id;
                int it = m_idx / 4;
                // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
                // inline_buffer_load_dword(sorted_token_ids_element_store[it],  m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
                sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
                // int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                // tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                // token_index_store[it] = token_ids_store * 8 + tok_ids_store[it];
            }
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
                for (int i = 0; i < 32 / mfma_n; i++) {

                    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

                    // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
                    // inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i ], row_id *  2 ,b_scale_ptr_prepared,min_tile_n * 32 + i );
                    b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
                }
            }
        }
    }

    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    __builtin_amdgcn_sched_barrier(0);

    // scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    if (size_k == 2048) {
        constexpr int SIZE_K = 2048;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 1536) {
        constexpr int SIZE_K = 1536;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 1024) {
        constexpr int SIZE_K = 1024;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 768) {
        constexpr int SIZE_K = 768;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 512) {
        constexpr int SIZE_K = 512;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 384) {
        constexpr int SIZE_K = 384;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 256) {
        constexpr int SIZE_K = 256;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 192) {
        constexpr int SIZE_K = 192;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 128) {
        constexpr int SIZE_K = 128;
        gemm_nt_marlin_decode_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    }

    __builtin_amdgcn_sched_barrier(0);

    scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
#pragma unroll
                for (int i = 0; i < 2; i++) {

                    const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        // float  tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
                        value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t> /* f32_to_bf16  */ (tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
                    }
                }
            }
        }
    }

    //     {

    //   // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){
    //   #pragma unroll
    //   for (int min_tile_m = 0; min_tile_m < WARP_M / 16 ; min_tile_m ++){
    //     #pragma unroll
    //     for (int i =0; i < 4; i++){
    //       int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
    //       // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
    //       int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
    //       tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
    //         token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    //     }
    //   }

    // }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
                int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++) {
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (sorted_token_ids_element_store[it] < size_m * top_k) {

                        *(vec_element_2<scalar_t> *)(&g_output[/* token_index_store[it] * size_n */ sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
                    }
                }
            }
        }

        // __syncthreads();

        // }
    }
}

//////////////////////////////////////////////////////////////////////////////gemm2_marlin////////////////////////////////////////////////////////////////////////////////////////////////
// 不固定block weight重排

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk) {
    // const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidy = blockIdx.y; // pid_n
    // const int bidz = blockIdx.x; // pid_k

    const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.y; // pid_n
    const int bidz = blockIdx.x; // pid_k

    // block swizzle todo(precision wrong)
    //  constexpr int CU_NUMS = 80;
    //  constexpr int block_num_per_cu = 3;
    //  const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y;
    //  const int block_swizzle_x = bid % CU_NUMS ;
    //  const int block_swizzle_y = bid / CU_NUMS;

    // const int bidy = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) % gridDim.y  ;    // pid_n
    // const int bidx = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) / gridDim.y; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidz = blockIdx.x; // pid_k

    // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // int bidy = blockIdx.y; // pid_n
    // int bidz = blockIdx.x; // pid_k
    // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

    // constexpr int STAGES = 2;

    // if(blockIdx.x == 0 && threadIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0){
    //   printf("**********************************************size_n: %d size_k:%d", size_n,size_k);
    // }

    // uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    // if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

    if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) {
        return;
    }

    const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

    const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id; // 这个是对应专家的weight偏移

    // auto g_input = input;
    auto g_input = input;
    auto g_input_scale = input_scale; // 配置全局显存信息

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
    int lane_id = threadIdx.x & 63;                            // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];                             // 声明lds信息
    Element *input_lds = (Element *)&(smem);                      // decode这里没用
    Element *qweight_lds = input_lds;                             // decode这里没用
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds
    // float* output_lds_float = (float*)&(smem); // 重复使用lds,留给output_lds

    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    float weight_dot_a_scale[WARP_M / mfma_m][4];

    // #pragma unroll
    // for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    //   for(int i = 0 ;i< 4;i++){
    //     int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
    //     int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    //     int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
    //     int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1)  ;
    //     float input_scale_value = *(input_scale + token_index_safe * stride_asm); //计算M方向的偏移
    //     weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;

    //   }
    // }

#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++) {
        for (int i = 0; i < 4; i++) {
            int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
            // int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            // int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index_safe = std::min((uint32_t)sorted_token_ids_element, (size_m - 1) * top_k);
            float input_scale_value = *(input_scale + (token_index_safe / top_k) * stride_asm); // 计算M方向的偏移
            weight_dot_a_scale[idx][i] = /* topk_weights[token_index_safe / top_k] * */ input_scale_value;
        }
    }

    constexpr int n_loop_num = 4;

    const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                            // 计算之后是mxn,这应该是计算输出n方向的位置
    // if(output_offset + BLOCK_SIZE_N * n_loop_num >= size_n) return;
    const uint64_t weight_scale_offset = stride_bse /* * stride_bsn */ * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1) { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            zero_element_8.data[i] = 0;
        }

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;

            if (topk_ids < real_topk) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    {

        // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            for (int i = 0; i < 4; i++) {
                int m_idx = min_tile_m * 16 + i * 4 + col_id;
                int it = m_idx / 4;
                // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
                // inline_buffer_load_dword(sorted_token_ids_element_store[it],  m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
                sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
                // int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                // tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                // token_index_store[it] = token_ids_store * 8 + tok_ids_store[it];
            }
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
                for (int i = 0; i < 32 / mfma_n; i++) {

                    // vec<uint,4> b_scale_ptr_prepared =  tcp_cache_swizzle_func_no<128,float>(b_scale_ptr + BLOCK_SIZE_N* n_loop);

                    // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
                    // inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i ], row_id *  2 ,b_scale_ptr_prepared,min_tile_n * 32 + i );
                    b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
                }
            }
        }
    }

    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    __builtin_amdgcn_sched_barrier(0);

    // scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    constexpr int SIZE_K = 0;
    gemm_nt_marlin_prefill_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    __builtin_amdgcn_sched_barrier(0);

    scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
#pragma unroll
                for (int i = 0; i < 2; i++) {

                    const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        // float  tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
                        value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
                    }
                }
            }
        }
    }

    //     {

    //   // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){
    //   #pragma unroll
    //   for (int min_tile_m = 0; min_tile_m < WARP_M / 16 ; min_tile_m ++){
    //     #pragma unroll
    //     for (int i =0; i < 4; i++){
    //       int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
    //       // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
    //       int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
    //       tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
    //         token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    //     }
    //   }

    // }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
                int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++) {
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (sorted_token_ids_element_store[it] < size_m * top_k) {

                        *(vec_element_2<scalar_t> *)(&g_output[/* token_index_store[it] * size_n */ sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
                    }
                }
            }
        }

        // __syncthreads();

        // }
    }
}

//////////////////////////////////////////////////////////////////////////////gemm2_marlin////////////////////////////////////////////////////////////////////////////////////////////////
// 不固定block weight重排

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk) {
    // const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidy = blockIdx.y; // pid_n
    // const int bidz = blockIdx.x; // pid_k

    const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.y; // pid_n
    const int bidz = blockIdx.x; // pid_k

    // block swizzle todo(precision wrong)
    //  constexpr int CU_NUMS = 80;
    //  constexpr int block_num_per_cu = 3;
    //  const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y;
    //  const int block_swizzle_x = bid % CU_NUMS ;
    //  const int block_swizzle_y = bid / CU_NUMS;

    // const int bidy = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) % gridDim.y  ;    // pid_n
    // const int bidx = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) / gridDim.y; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidz = blockIdx.x; // pid_k

    // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // int bidy = blockIdx.y; // pid_n
    // int bidz = blockIdx.x; // pid_k
    // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

    // constexpr int STAGES = 2;

    // if(blockIdx.x == 0 && threadIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0){
    //   printf("**********************************************size_n: %d size_k:%d", size_n,size_k);
    // }

    // uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    // if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

    if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) {
        return;
    }

    const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

    const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id; // 这个是对应专家的weight偏移

    // auto g_input = input;
    auto g_input = input;
    auto g_input_scale = input_scale; // 配置全局显存信息

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
    int lane_id = threadIdx.x & 63;                            // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];                             // 声明lds信息
    Element *input_lds = (Element *)&(smem);                      // decode这里没用
    Element *qweight_lds = input_lds;                             // decode这里没用
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds
    // float* output_lds_float = (float*)&(smem); // 重复使用lds,留给output_lds

    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    float weight_dot_a_scale[WARP_M / mfma_m][4];

    // #pragma unroll
    // for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    //   for(int i = 0 ;i< 4;i++){
    //     int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
    //     int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    //     int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
    //     int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1)  ;
    //     float input_scale_value = *(input_scale + token_index_safe * stride_asm); //计算M方向的偏移
    //     weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;

    //   }
    // }

#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++) {
        for (int i = 0; i < 4; i++) {
            int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
            // int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            // int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index_safe = std::min((uint32_t)sorted_token_ids_element, (size_m - 1) * top_k);
            float input_scale_value = *(input_scale + (token_index_safe / top_k) * stride_asm); // 计算M方向的偏移
            weight_dot_a_scale[idx][i] = /* topk_weights[token_index_safe / top_k] * */ input_scale_value;
        }
    }

    constexpr int n_loop_num = N_LOOP_NUM;

    const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                            // 计算之后是mxn,这应该是计算输出n方向的位置
    // if(output_offset + BLOCK_SIZE_N * n_loop_num >= size_n) return;
    const uint64_t weight_scale_offset = stride_bse /* * stride_bsn */ * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1) { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            zero_element_8.data[i] = 0;
        }

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;

            if (topk_ids < real_topk) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    {

        // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            for (int i = 0; i < 4; i++) {
                int m_idx = min_tile_m * 16 + i * 4 + col_id;
                int it = m_idx / 4;
                // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
                // inline_buffer_load_dword(sorted_token_ids_element_store[it],  m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
                sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
                // int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                // tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                // token_index_store[it] = token_ids_store * 8 + tok_ids_store[it];
            }
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
                for (int i = 0; i < 32 / mfma_n; i++) {

                    // vec<uint,4> b_scale_ptr_prepared =  tcp_cache_swizzle_func_no<128,float>(b_scale_ptr + BLOCK_SIZE_N* n_loop);

                    // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
                    // inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i ], row_id *  2 ,b_scale_ptr_prepared,min_tile_n * 32 + i );
                    b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
                }
            }
        }
    }

    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    __builtin_amdgcn_sched_barrier(0);

    // scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    constexpr int SIZE_K = 0;
    gemm_nt_marlin_prefill_w4a8_gemm1n256<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, n_loop_num, Element, scalar_t>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    __builtin_amdgcn_sched_barrier(0);

    scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
#pragma unroll
                for (int i = 0; i < 2; i++) {

                    const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        // float  tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
                        value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t>(tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
                    }
                }
            }
        }
    }

    //     {

    //   // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){
    //   #pragma unroll
    //   for (int min_tile_m = 0; min_tile_m < WARP_M / 16 ; min_tile_m ++){
    //     #pragma unroll
    //     for (int i =0; i < 4; i++){
    //       int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
    //       // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
    //       int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
    //       tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
    //         token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    //     }
    //   }

    // }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
                int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++) {
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (sorted_token_ids_element_store[it] < size_m * top_k) {

                        *(vec_element_2<scalar_t> *)(&g_output[/* token_index_store[it] * size_n */ sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
                    }
                }
            }
        }

        // __syncthreads();

        // }
    }
}

//////////////////////////////////////////////////////////////////////////////gemm2_marlin////////////////////////////////////////////////////////////////////////////////////////////////
// 不固定block weight重排

template <
    typename scalar_t,
    typename Element,
    int WARP_NUM,
    int BLOCK_SIZE_M,
    int BLOCK_SIZE_N,
    int BLOCK_SIZE_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int GROUP_N,
    int GROUP_K,
    int STAGES,
    int N_LOOP_NUM,
    int FIXED_SIZE_K,
    bool mul_topk_weight> // true
__global__ void __launch_bounds__(512, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN(
    const Element *__restrict__ input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk) {
    // const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidy = blockIdx.y; // pid_n
    // const int bidz = blockIdx.x; // pid_k

    const int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    const int bidy = blockIdx.y; // pid_n
    const int bidz = blockIdx.x; // pid_k

    // block swizzle todo(precision wrong)
    //  constexpr int CU_NUMS = 80;
    //  constexpr int block_num_per_cu = 3;
    //  const int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y;
    //  const int block_swizzle_x = bid % CU_NUMS ;
    //  const int block_swizzle_y = bid / CU_NUMS;

    // const int bidy = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) % gridDim.y  ;    // pid_n
    // const int bidx = ( (block_swizzle_y % block_num_per_cu  )+ block_swizzle_x *  block_num_per_cu + block_swizzle_y / block_num_per_cu * block_num_per_cu * CU_NUMS) / gridDim.y; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // const int bidz = blockIdx.x; // pid_k

    // int bidx = blockIdx.z; // 分别在三个方向上都有block pid_m m方向分块,可以理解为按照专家或者专家对应的token来并行
    // int bidy = blockIdx.y; // pid_n
    // int bidz = blockIdx.x; // pid_k
    // GetBLockIdx(bidx + bidy * gridDim.x, gridDim.x, gridDim.y, 0, 8, bidx, bidy);

    // constexpr int STAGES = 2;

    // if(blockIdx.x == 0 && threadIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0){
    //   printf("**********************************************size_n: %d size_k:%d", size_n,size_k);
    // }

    // uint32_t topk_ids = (sorted_token_ids[bidx * BLOCK_SIZE_M] & 0xFF000000) >> 24;
    // if (topk_ids >= real_topk || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return; // 对于无效的block,直接返回,num_tokens_post_pad[0]=10144

    if (sorted_token_ids[bidx * BLOCK_SIZE_M] >= size_m * top_k || bidx * BLOCK_SIZE_M >= num_tokens_post_pad[0]) {
        return;
    }

    const uint32_t input_offset = bidz * BLOCK_SIZE_K /* bidz * BLOCK_SIZE_K */; // 输入k方向分块的位置
    const int32_t delta_bidx = bidx;
    const int32_t expert_id = expert_ids[delta_bidx]; // 专家的索引

    const uint64_t expert_offset = ((uint64_t)size_n) * size_k / 2 * expert_id; // 这个是对应专家的weight偏移

    // auto g_input = input;
    auto g_input = input;
    auto g_input_scale = input_scale; // 配置全局显存信息

    constexpr int mfma_m = 16;
    constexpr int mfma_n = 16;
    constexpr int mfma_k = 32;

    int warp_id_vec = threadIdx.x / 64;                        // warp id in a block
    int warp_id = __builtin_amdgcn_readfirstlane(warp_id_vec); // 用于对warp id直接进行广播，不同一个block中的每个线程都去计算threadIdx.x / 64
    int lane_id = threadIdx.x & 63;                            // thread_id
    int row_id = lane_id % 16;
    int col_id = lane_id / 16;
    const int warp_n_num = BLOCK_SIZE_N / WARP_N;
    const int warp_k_num = BLOCK_SIZE_K / WARP_K;
    int warp_k_id = warp_id % warp_k_num;
    int warp_n_id = warp_id / warp_k_num;
    extern __shared__ Element smem[];                             // 声明lds信息
    Element *input_lds = (Element *)&(smem);                      // decode这里没用
    Element *qweight_lds = input_lds;                             // decode这里没用
    scalar_t *output_lds = reinterpret_cast<scalar_t *>(&(smem)); // 重复使用lds,留给output_lds
    // float* output_lds_float = (float*)&(smem); // 重复使用lds,留给output_lds

    float *b_scale_lds = (float *)&(smem);

    union_vec_opt<Element, WARP_K / 4> A_reg[WARP_M / mfma_m][STAGES];
    union_vec_opt<Element, WARP_K / 4> B_reg[WARP_N / mfma_n][2][STAGES];

    float weight_dot_a_scale[WARP_M / mfma_m][4];

    // #pragma unroll
    // for(int idx = 0; idx < WARP_M / mfma_m; idx++){
    //   for(int i = 0 ;i< 4;i++){
    //     int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
    //     int token_ids = sorted_token_ids_element & 0x00FFFFFF;
    //     int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
    //     int token_index_safe = std::min(uint32_t(token_ids * real_topk /* top_k */ + topk_ids), size_m - 1)  ;
    //     float input_scale_value = *(input_scale + token_index_safe * stride_asm); //计算M方向的偏移
    //     weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;

    //   }
    // }

#pragma unroll
    for (int idx = 0; idx < WARP_M / mfma_m; idx++) {
        for (int i = 0; i < 4; i++) {
            int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + idx * mfma_m + col_id + i * 4, int(sorted_token_lens - 1))];
            // int token_ids = sorted_token_ids_element & 0x00FFFFFF;
            // int topk_ids =  (sorted_token_ids_element & 0xFF000000) >> 24;
            int token_index_safe = std::min((uint32_t)sorted_token_ids_element, size_m * top_k - 1);
            float input_scale_value = *(input_scale + token_index_safe * stride_asm); // 计算M方向的偏移
            weight_dot_a_scale[idx][i] = topk_weights[token_index_safe] * input_scale_value;
        }
    }

    constexpr int n_loop_num = N_LOOP_NUM;

    const uint64_t qweight_offset = expert_offset + bidy * 32 * BLOCK_SIZE_N * n_loop_num /* + 64 * BLOCK_SIZE_N* n_loop */;                  // 具体偏移到对应专家的weight的某一个小的分块
    const uint32_t output_offset = bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */;                                             // 计算之后是mxn,这应该是计算输出n方向的位置
    const uint64_t weight_scale_offset = stride_bse * stride_bsn * expert_id + bidy * BLOCK_SIZE_N * n_loop_num /* + BLOCK_SIZE_N* n_loop */; // 具体偏移到对应专家的weight的某一个小的分块
    scalar_t *g_output;
    g_output = output + output_offset;

    if (expert_id == -1) { // EP算法处理 epxert_id为-1 写回0
        const int tid = threadIdx.x;
        constexpr int N_thread = BLOCK_SIZE_N / 8; // N方向需要的线程数 使用dwordx4即8个bf16
        vec_element_8<scalar_t> zero_element_8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            zero_element_8.data[i] = 0;
        }

        int m_idx = threadIdx.x / N_thread;
        int n_idx = threadIdx.x % N_thread;
        for (; m_idx < BLOCK_SIZE_M; m_idx += (WARP_NUM * 64) / N_thread) {
            const int32_t sorted_token_ids_element = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M + m_idx, int(sorted_token_lens - 1))];
            const int32_t token_ids = sorted_token_ids_element & 0x00FFFFFF;
            const int32_t topk_ids = sorted_token_ids_element & 0xFF000000;
            int token_index = token_ids * real_topk /* top_k */ + topk_ids;

            if (topk_ids < real_topk) {
                *reinterpret_cast<vec_element_8<scalar_t> *>(&g_output[(token_index)*size_n + n_idx * 8]) = zero_element_8;
            }
        }
        return;
    }

    constexpr int store_size = WARP_M / 4;
    int token_index_store[store_size];
    int tok_ids_store[store_size];
    int32_t sorted_token_ids_element_store[store_size];
    auto g_sorted_token_ids_offset = tcp_cache_swizzle_func_no<64, int32_t>(sorted_token_ids);

    {

        // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){ // 会导致循环无法展开 最终导致core dump
        for (int min_tile_m = 0; min_tile_m < WARP_M / mfma_m; min_tile_m++) {
            for (int i = 0; i < 4; i++) {
                int m_idx = min_tile_m * 16 + i * 4 + col_id;
                int it = m_idx / 4;
                // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
                // inline_buffer_load_dword(sorted_token_ids_element_store[it],  m_idx, g_sorted_token_ids_offset, bidx * BLOCK_SIZE_M);
                sorted_token_ids_element_store[it] = sorted_token_ids[bidx * BLOCK_SIZE_M + m_idx];
                // int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
                // tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
                // token_index_store[it] = token_ids_store * 8 + tok_ids_store[it];
            }
        }
    }

    auto g_qweight = qweight + qweight_offset;
    auto g_weight_scale = weight_scale + weight_scale_offset; // 配置weight的scale信息 todo 这里n_loop=0没有计算偏移
    float *b_scale_ptr = weight_scale + weight_scale_offset + warp_n_id * WARP_N;

    float b_scale[n_loop_num][(WARP_N / mfma_n)];

    {
#pragma unroll
        for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < WARP_N / 32; min_tile_n++) {
                for (int i = 0; i < 32 / mfma_n; i++) {

                    vec<uint, 4> b_scale_ptr_prepared = tcp_cache_swizzle_func_no<128, float>(b_scale_ptr + BLOCK_SIZE_N * n_loop);

                    // b_scale[min_tile_n*4 + reg_id] = (b_scale_ptr + min_tile_n * mfma_n + col_id + reg_id * 4)[0];
                    // inline_buffer_load_dword(b_scale[n_loop][min_tile_n * 2 + i ], row_id *  2 ,b_scale_ptr_prepared,min_tile_n * 32 + i );
                    b_scale[n_loop][min_tile_n * 2 + i] = b_scale_ptr[BLOCK_SIZE_N * n_loop + min_tile_n * 32 + i + row_id * 2];
                }
            }
        }
    }

    intx4 C_reg[n_loop_num][(WARP_M / 16) * (WARP_N / 16)] = {0, 0, 0, 0}; // [4][2]  每个warp在n方向重复两次 tileN = 16*2
    __builtin_amdgcn_sched_barrier(0);

    // scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    float tmp[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
    if constexpr (FIXED_SIZE_K == 192) {
        constexpr int SIZE_K = 192;
        gemm_nt_marlin_prefill_2_w4a8_k192<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    } else if constexpr (FIXED_SIZE_K == 384) {
        constexpr int SIZE_K = 384;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    } else if constexpr (FIXED_SIZE_K == 768) {
        constexpr int SIZE_K = 768;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    } else if constexpr (FIXED_SIZE_K == 1536) {
        constexpr int SIZE_K = 1536;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    } else if (size_k == 3072) {
        constexpr int SIZE_K = 3072;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 2048) {
        constexpr int SIZE_K = 2048;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 1536) {
        constexpr int SIZE_K = 1536;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 1024) {
        constexpr int SIZE_K = 1024;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 768) {
        constexpr int SIZE_K = 768;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 512) {
        constexpr int SIZE_K = 512;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 384) {
        constexpr int SIZE_K = 384;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 256) {
        constexpr int SIZE_K = 256;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 192) {
        constexpr int SIZE_K = 192;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);

    } else if (size_k == 128) {
        constexpr int SIZE_K = 128;
        gemm_nt_marlin_prefill_2_w4a8<false, 0, 0, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES, GROUP_N, GROUP_K, SIZE_K, Element, scalar_t, n_loop_num>(g_input, g_qweight, input_lds, qweight_lds, g_input_scale, g_weight_scale, size_m, A_reg, B_reg, C_reg, warp_id, size_k, size_k, stride_asm, stride_ask, stride_bse, stride_bsn, stride_bsk, top_k, sorted_token_ids, sorted_token_lens, expert_id, bidx, weight_dot_a_scale, b_scale, tmp, real_topk);
    }

    __builtin_amdgcn_sched_barrier(0);

    scalar_t value[n_loop_num][WARP_M / mfma_m][4][WARP_N / mfma_n];
#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {

#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {
#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
#pragma unroll
                for (int i = 0; i < 2; i++) {

                    const int tile_idx = min_tile_m * (WARP_N / mfma_n) + min_tile_n * 2 + i;
#pragma unroll
                    for (int reg_id = 0; reg_id < 4; reg_id++) {
                        // float  tmp = C_reg[n_loop][tile_idx][reg_id] * weight_dot_a_scale[min_tile_m][reg_id] * b_scale[n_loop][min_tile_n * 2 + i];
                        value[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i] = b32_to_b16<scalar_t> /* f32_to_bf16  */ (tmp[n_loop][min_tile_m][reg_id][min_tile_n * 2 + i]);
                    }
                }
            }
        }
    }

    //     {

    //   // for (int col_id_tmp = col_id; col_id_tmp < BLOCK_SIZE_M; col_id_tmp += 4  ){
    //   #pragma unroll
    //   for (int min_tile_m = 0; min_tile_m < WARP_M / 16 ; min_tile_m ++){
    //     #pragma unroll
    //     for (int i =0; i < 4; i++){
    //       int it = (min_tile_m * 16 + i * 4 + col_id) / 4;
    //       // sorted_token_ids_element_store[it] = sorted_token_ids[std::min(bidx * BLOCK_SIZE_M+ col_id_tmp, int(sorted_token_lens - 1))];
    //       int token_ids_store = sorted_token_ids_element_store[it] & 0x00FFFFFF;
    //       tok_ids_store[it] =  (sorted_token_ids_element_store[it] & 0xFF000000) >> 24;
    //         token_index_store[it] = token_ids_store * real_topk + tok_ids_store[it];
    //     }
    //   }

    // }

#pragma unroll
    for (int n_loop = 0; n_loop < n_loop_num; n_loop++) {
#pragma unroll
        for (int min_tile_m = 0; min_tile_m < (WARP_M / mfma_m); min_tile_m++) {

#pragma unroll
            for (int min_tile_n = 0; min_tile_n < (WARP_N / 32); min_tile_n++) {
                int index = row_id * 2 + min_tile_n * 32 + warp_id * WARP_N + BLOCK_SIZE_N * n_loop;
#pragma unroll
                for (int reg_id = 0; reg_id < 4; reg_id++) {
                    int it = (min_tile_m * 16 + reg_id * 4 + col_id) / 4;
                    if (sorted_token_ids_element_store[it] < size_m * top_k) {

                        *(vec_element_2<scalar_t> *)(&g_output[/* token_index_store[it] * size_n */ sorted_token_ids_element_store[it] * size_n + index]) = *(vec_element_2<scalar_t> *)(&value[n_loop][min_tile_m][reg_id][min_tile_n * 2]);
                    }
                }
            }
        }

        // __syncthreads();

        // }
    }
}

template <typename Element>
__device__ __forceinline__ int32_t w4a8_load_kblocked_int4(
    const Element *__restrict__ qweight,
    uint64_t expert_offset,
    uint32_t n,
    uint32_t k,
    uint32_t size_k) {
    const uint32_t byte_k = (k / 8) * 4 + (k % 4);
    const uint8_t packed = static_cast<uint8_t>(qweight[expert_offset + static_cast<uint64_t>(n) * (size_k / 2) + byte_k]);
    const uint8_t raw = (k % 8) < 4 ? ((packed >> 4) & 0xF) : (packed & 0xF);
    return raw >= 8 ? static_cast<int32_t>(raw) - 16 : static_cast<int32_t>(raw);
}

template <
    typename scalar_t,
    typename Element,
    uint16_t BLOCK_SIZE_N,
    uint16_t BLOCK_SIZE_K,
    bool MUL_TOPK_WEIGHT>
__global__ void __launch_bounds__(256, 1) MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_GENERAL(
    const Element *input,
    const Element *__restrict__ qweight,
    scalar_t *__restrict__ output,
    float *__restrict__ input_scale,
    float *__restrict__ weight_scale,
    const float *__restrict__ topk_weights,
    const int32_t *sorted_token_ids,
    const int32_t *__restrict__ expert_ids,
    const int32_t *__restrict__ num_tokens_post_pad,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t stride_asm,
    uint32_t stride_ask,
    uint32_t stride_bse,
    uint32_t stride_bsn,
    uint32_t stride_bsk,
    uint32_t sorted_token_lens,
    uint32_t top_k,
    uint32_t real_topk,
    uint32_t block_size_m,
    uint32_t n_loop) {
    const uint32_t tile_n = BLOCK_SIZE_N * n_loop;
    const uint32_t bidx = blockIdx.z;
    const uint32_t n_base = blockIdx.y * tile_n;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_n = tid % 32;
    const uint32_t lane_m = tid / 32;
    const int8_t *input_i8 = reinterpret_cast<const int8_t *>(input);

    if (bidx * block_size_m >= static_cast<uint32_t>(num_tokens_post_pad[0])) {
        return;
    }
    if (sorted_token_ids[bidx * block_size_m] >= static_cast<int32_t>(size_m * top_k)) {
        return;
    }

    const int32_t expert_id = expert_ids[bidx];
    if (expert_id < 0) {
        for (uint32_t n_inner = lane_n; n_inner < tile_n; n_inner += 32) {
            const uint32_t n = n_base + n_inner;
            if (n >= size_n) {
                continue;
            }
            for (uint32_t m_idx = lane_m; m_idx < block_size_m; m_idx += 8) {
                const uint32_t sorted_idx = bidx * block_size_m + m_idx;
                if (sorted_idx >= sorted_token_lens) {
                    continue;
                }
                const int32_t sorted_token_id = sorted_token_ids[sorted_idx];
                if (sorted_token_id >= static_cast<int32_t>(size_m * top_k)) {
                    continue;
                }
                output[static_cast<uint64_t>(sorted_token_id) * size_n + n] = b32_to_b16<scalar_t>(0.0f);
            }
        }
        return;
    }

    const uint64_t expert_offset = static_cast<uint64_t>(expert_id) * size_n * size_k / 2;

    for (uint32_t n_inner = lane_n; n_inner < tile_n; n_inner += 32) {
        const uint32_t n = n_base + n_inner;
        if (n >= size_n) {
            continue;
        }

        const float b_scale = 16.0f * weight_scale[static_cast<uint64_t>(expert_id) * stride_bse + static_cast<uint64_t>(n) * stride_bsn];

        for (uint32_t m_idx = lane_m; m_idx < block_size_m; m_idx += 8) {
            const uint32_t sorted_idx = bidx * block_size_m + m_idx;
            if (sorted_idx >= sorted_token_lens) {
                continue;
            }

            const int32_t sorted_token_id = sorted_token_ids[sorted_idx];
            if (sorted_token_id >= static_cast<int32_t>(size_m * top_k)) {
                continue;
            }

            const uint32_t input_row = MUL_TOPK_WEIGHT
                                         ? static_cast<uint32_t>(sorted_token_id)
                                         : static_cast<uint32_t>(sorted_token_id) / real_topk;

            float acc = 0.0f;
            for (uint32_t k_base = 0; k_base < size_k; k_base += BLOCK_SIZE_K) {
#pragma unroll
                for (uint32_t kk = 0; kk < BLOCK_SIZE_K; ++kk) {
                    const uint32_t k = k_base + kk;
                    if (k >= size_k) {
                        break;
                    }
                    const int32_t a = static_cast<int32_t>(input_i8[static_cast<uint64_t>(input_row) * size_k + k]);
                    const int32_t b = w4a8_load_kblocked_int4(qweight, expert_offset, n, k, size_k);
                    acc += static_cast<float>(a * b);
                }
            }

            const float a_scale = input_scale[static_cast<uint64_t>(input_row) * stride_asm];
            float value = acc * a_scale * b_scale;
            if constexpr (MUL_TOPK_WEIGHT) {
                value *= topk_weights[sorted_token_id];
            }
            output[static_cast<uint64_t>(sorted_token_id) * size_n + n] = b32_to_b16<scalar_t>(value);
        }
    }
}

template <typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_general(
    const GemmParams_w4a8<T, T_hidden> &params,
    uint32_t block_size_m,
    uint32_t block_size_n,
    uint32_t block_size_k,
    uint32_t n_loop) {
    dim3 blockDim, gridDim;
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, block_size_m));
    gridDim.y = DIVIDE(params.size_n, block_size_n * n_loop);
    gridDim.x = 1;

    const hipStream_t stream = params.stream;
    MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_GENERAL<T_hidden, T, 128, 64, false>
        <<<gridDim, blockDim, 0, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk,
            block_size_m,
            n_loop);
}

template <typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_general(
    const GemmParams_w4a8<T, T_hidden> &params,
    uint32_t block_size_m,
    uint32_t block_size_n,
    uint32_t block_size_k,
    uint32_t n_loop) {
    dim3 blockDim, gridDim;
    blockDim.x = 256;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, block_size_m));
    gridDim.y = DIVIDE(params.size_n, block_size_n * n_loop);
    gridDim.x = 1;

    const hipStream_t stream = params.stream;
    MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_GENERAL<T_hidden, T, 128, 64, true>
        <<<gridDim, blockDim, 0, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk,
            block_size_m,
            n_loop);
}

// launch_gemm1_decode
// warp在n和k方向排列
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill(const GemmParams_w4a8<T, T_hidden> &params) {
    // constexpr int STAGES = 2;
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;
    // std::cout<<"second: "<<"BLOCK_SIZE_M"<<BLOCK_SIZE_M<<"BLOCK_SIZE_N"<<BLOCK_SIZE_N<<"BLOCK_SIZE_K"<<BLOCK_SIZE_K<<"WARP_M"<<WARP_M<<"WARP_N"<<WARP_N<<"WARP_K"<<WARP_K<<std::endl;

    // gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.x = 1; // k方向
    // unsigned int lens = params.num_tokens_post_pad_ptr[0];
    constexpr int n_loop_num = 4;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    // if(params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0) return;
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
    // printf("**********************************************size_n: %d BLOCK_SIZE_N: %d",params.size_n,BLOCK_SIZE_N);
    gridDim.x = 1; // k方向

    // block swizzle 效果提升不是很明显 需要tuning验证
    // gridDim.x = min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    // //gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.z = 1; // k方向

    const int lds_size = BLOCK_SIZE_M * WARP_K * 2 /* + 64 * BLOCK_SIZE_M / 16 */;
    // const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * 2; // 假设GEMM2 K方向没有wave 不需要在共享内存做累加

    // printf("****************************************** BLOCK_SIZE_M %d BLOCK_SIZE_N %d BLOCK_SIZE_K %d WARP_K %d lds_size   %d",BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,WARP_K,lds_size);
    // const int lds_size =16*1024;
    const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = params.stream;

    if (params.is_marlin == false) {
        // MOE_W8A8_I8_PERCHANNEL_HIP_NT_DECODE_DOWN<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
        //   GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        //   params.ptr_A,
        //   params.ptr_B0,
        //   params.ptr_C,
        //   params.ptr_A_scale,
        //   params.ptr_B_scale,
        //   params.topk_weights,
        //   params.sorted_token_ids,
        //   params.expert_ids,
        //   params.num_tokens_post_pad_ptr,
        //   params.size_m,
        //   params.size_n,
        //   params.size_k,
        //   params.stride_asm,
        //   params.stride_ask,
        //   params.stride_bse,
        //   params.stride_bsn,
        //   params.stride_bsk,
        //   params.sorted_token_lens,
        //   params.top_k,
        //   params.delta);
    } else {
        // printf("*****************************************MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN");
        // hipGetLastError();
        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                        GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);

        // hipDeviceSynchronize();
    }
    // hipDeviceSynchronize();

    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    // }
}

// launch_gemm1_decode
// warp在n和k方向排列
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1N256(const GemmParams_w4a8<T, T_hidden> &params) {
    // constexpr int STAGES = 2;
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;
    // std::cout<<"second: "<<"BLOCK_SIZE_M"<<BLOCK_SIZE_M<<"BLOCK_SIZE_N"<<BLOCK_SIZE_N<<"BLOCK_SIZE_K"<<BLOCK_SIZE_K<<"WARP_M"<<WARP_M<<"WARP_N"<<WARP_N<<"WARP_K"<<WARP_K<<std::endl;

    // gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.x = 1; // k方向
    // unsigned int lens = params.num_tokens_post_pad_ptr[0];
    constexpr int n_loop_num = 2;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    // if(params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0) return;
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
    // printf("**********************************************size_n: %d BLOCK_SIZE_N: %d",params.size_n,BLOCK_SIZE_N);
    gridDim.x = 1; // k方向

    // block swizzle 效果提升不是很明显 需要tuning验证
    // gridDim.x = min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    // //gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.z = 1; // k方向

    const int lds_size = BLOCK_SIZE_M * WARP_K * 2 /* + 64 * BLOCK_SIZE_M / 16 */;
    // const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * 2; // 假设GEMM2 K方向没有wave 不需要在共享内存做累加

    // printf("****************************************** BLOCK_SIZE_M %d BLOCK_SIZE_N %d BLOCK_SIZE_K %d WARP_K %d lds_size   %d",BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,WARP_K,lds_size);
    // const int lds_size =16*1024;
    const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = params.stream;

    if (params.is_marlin == false) {
        // MOE_W8A8_I8_PERCHANNEL_HIP_NT_DECODE_DOWN<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
        //   GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        //   params.ptr_A,
        //   params.ptr_B0,
        //   params.ptr_C,
        //   params.ptr_A_scale,
        //   params.ptr_B_scale,
        //   params.topk_weights,
        //   params.sorted_token_ids,
        //   params.expert_ids,
        //   params.num_tokens_post_pad_ptr,
        //   params.size_m,
        //   params.size_n,
        //   params.size_k,
        //   params.stride_asm,
        //   params.stride_ask,
        //   params.stride_bse,
        //   params.stride_bsn,
        //   params.stride_bsk,
        //   params.sorted_token_lens,
        //   params.top_k,
        //   params.delta);
    } else {
        // printf("*****************************************MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN");
        // hipGetLastError();
        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                                  GROUP_N, GROUP_K, STAGES, 2, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);

        // hipDeviceSynchronize();
    }
    // hipDeviceSynchronize();

    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    // }
}

// HY3 special GEMM1 path. N_LOOP_NUM lets the same special kernel cover more
// N tilings without falling back to the general kernel.
template <int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1Nloop(const GemmParams_w4a8<T, T_hidden> &params) {
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
    gridDim.x = 1;
    const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2;
    const hipStream_t stream = params.stream;

    if (params.is_marlin == true) {
        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_UP_GEMM1N256<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                                  GROUP_N, GROUP_K, STAGES, n_loop_num, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);
    }
}

// n=192 intermediate uses GEMM1 output N=384 = BLOCK_N(128) * N_LOOP(3).
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_prefill_GEMM1N384(const GemmParams_w4a8<T, T_hidden> &params) {
    launch_moe_w4a8_first_stage_prefill_GEMM1Nloop<3, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES>(params);
}

// launch_gemm2_decode
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill(const GemmParams_w4a8<T, T_hidden> &params) {
    // const int STAGES = 2;
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;
    // std::cout<<"second: "<<"BLOCK_SIZE_M"<<BLOCK_SIZE_M<<"BLOCK_SIZE_N"<<BLOCK_SIZE_N<<"BLOCK_SIZE_K"<<BLOCK_SIZE_K<<"WARP_M"<<WARP_M<<"WARP_N"<<WARP_N<<"WARP_K"<<WARP_K<<std::endl;

    // gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.x = 1; // k方向
    // unsigned int lens = params.num_tokens_post_pad_ptr[0];
    constexpr int n_loop_num = 4;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0) {
        return;
    }
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
    // printf("**********************************************size_n: %d BLOCK_SIZE_N: %d",params.size_n,BLOCK_SIZE_N);
    gridDim.x = 1; // k方向

    // block swizzle 效果提升不是很明显 需要tuning验证
    // gridDim.x = min(DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M), DIVIDE(params.num_tokens_post_pad, BLOCK_SIZE_M)); // m方向
    // //gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
    // gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
    // gridDim.z = 1; // k方向

    const int lds_size = BLOCK_SIZE_M * WARP_K * 2 /* + 64 * BLOCK_SIZE_M / 16 */;
    // const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * 2; // 假设GEMM2 K方向没有wave 不需要在共享内存做累加

    // printf("****************************************** BLOCK_SIZE_M %d BLOCK_SIZE_N %d BLOCK_SIZE_K %d WARP_K %d lds_size   %d",BLOCK_SIZE_M,BLOCK_SIZE_N,BLOCK_SIZE_K,WARP_K,lds_size);
    // const int lds_size = 8*1024;
    const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = params.stream;

    if (params.is_marlin == false) {
        // MOE_W8A8_I8_PERCHANNEL_HIP_NT_DECODE_DOWN<char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
        //   GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        //   params.ptr_A,
        //   params.ptr_B0,
        //   params.ptr_C,
        //   params.ptr_A_scale,
        //   params.ptr_B_scale,
        //   params.topk_weights,
        //   params.sorted_token_ids,
        //   params.expert_ids,
        //   params.num_tokens_post_pad_ptr,
        //   params.size_m,
        //   params.size_n,
        //   params.size_k,
        //   params.stride_asm,
        //   params.stride_ask,
        //   params.stride_bse,
        //   params.stride_bsn,
        //   params.stride_bsk,
        //   params.sorted_token_lens,
        //   params.top_k,
        //   params.delta);
    } else {
        // printf("*****************************************MOE_W8A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN");
        // hipGetLastError();
        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                          GROUP_N, GROUP_K, STAGES, n_loop_num, 0, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);

        // hipDeviceSynchronize();
    }

    // cudaDeviceSynchronize();

    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error in gemm2: %s\n", cudaGetErrorString(err));
    // }
}

template <int FIXED_SIZE_K, int N_LOOP_NUM, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_fixed_k(const GemmParams_w4a8<T, T_hidden> &params) {
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    constexpr int n_loop_num = N_LOOP_NUM;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M));
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0) {
        return;
    }
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num);
    gridDim.x = 1;
    const int shared_mem_size = BLOCK_SIZE_M * WARP_K * 2;
    const hipStream_t stream = params.stream;

    if (params.is_marlin == true) {
        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_PREFILL_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                          GROUP_N, GROUP_K, STAGES, n_loop_num, FIXED_SIZE_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);
    }
}

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_prefill_K192(const GemmParams_w4a8<T, T_hidden> &params) {
    launch_moe_w4a8_second_stage_prefill_fixed_k<192, 4, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K, STAGES>(params);
}

// launch_gemm1_decode
// warp在n和k方向排列
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_first_stage_decode(const GemmParams_w4a8<T, T_hidden> &params) {
    // constexpr int STAGES = 2;
    constexpr int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = false;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;

    // std::cout<<"first:"<<"BLOCK_SIZE_M"<<BLOCK_SIZE_M<<"BLOCK_SIZE_N"<<BLOCK_SIZE_N<<"BLOCK_SIZE_K"<<BLOCK_SIZE_K<<"WARP_M"<<WARP_M<<"WARP_N"<<WARP_N<<"WARP_K"<<WARP_K<<std::endl;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;

    // const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 2 + BLOCK_SIZE_M / 2 *16;
    const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * (BLOCK_SIZE_K / WARP_K) * 4; // max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    // const int lds_size = BLOCK_SIZE_M * BLOCK_SIZE_N * 2;
    // const int lds_size =10*1024;//max(BLOCK_SIZE_M * BLOCK_SIZE_N * 2, BLOCK_SIZE_M * BLOCK_SIZE_N * 2);
    const int shared_mem_size = lds_size;

    // std::cout<<"shared_mem_size: "<<shared_mem_size<<std::endl<<"blockDim.x: "<<blockDim.x<<std::endl;
    const hipStream_t stream = params.stream;

    if (params.is_marlin == false) {
        // gridDim.z = DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M); // m方向
        // gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
        // gridDim.y = 1; // k方向

        // MOE_W8A8_I8_PERCHANNEL_HIP_NT_DECODE_UP<T_hidden,char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
        //   GROUP_N, GROUP_K, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
        //   params.ptr_A,
        //   params.ptr_B0,
        //   params.ptr_C,
        //   params.ptr_A_scale,
        //   params.ptr_B_scale,
        //   params.topk_weights,
        //   params.sorted_token_ids,
        //   params.expert_ids,
        //   params.num_tokens_post_pad_ptr,
        //   params.size_m,
        //   params.size_n,
        //   params.size_k,
        //   params.stride_asm,
        //   params.stride_ask,
        //   params.stride_bse,
        //   params.stride_bsn,
        //   params.stride_bsk,
        //   params.sorted_token_lens,
        //   params.top_k,
        //   params.delta);
    } else { // marlin版本

        gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
        // gridDim.z = params.size_m*params.top_k; // m方向
        gridDim.x = DIVIDE(params.size_n, BLOCK_SIZE_N); // n方向
        gridDim.y = 1;                                   // k方向
                                                         // std::cout<<"gridDim.z: "<<gridDim.z<<std::endl<<"gridDim.y: "<<gridDim.y<<std::endl;

        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_UP<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                       GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);
    }
    // hipDeviceSynchronize();

    // auto err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error in gemm1: %s\n", cudaGetErrorString(err));
    // }
}

// launch_gemm2_decode
template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int WARP_M, int WARP_N, int WARP_K, int STAGES, typename T, typename T_hidden>
void launch_moe_w4a8_second_stage_decode(const GemmParams_w4a8<T, T_hidden> &params) {
    // const int STAGES = 2;
    const int WARP_NUM = (BLOCK_SIZE_N / WARP_N) * (BLOCK_SIZE_K / WARP_K);
    const bool mul_topk_weight = true;
    constexpr int GROUP_N = 1;
    constexpr int GROUP_K = 1;
    dim3 blockDim, gridDim;
    blockDim.x = WARP_NUM * 64;
    blockDim.y = 1;
    blockDim.z = 1;

    constexpr int n_loop_num = 4;
    gridDim.z = std::min(params.size_m * params.top_k, DIVIDE(params.sorted_token_lens, BLOCK_SIZE_M)); // m方向
    if (params.size_n % (BLOCK_SIZE_N * n_loop_num) != 0) {
        return;
    }
    gridDim.y = DIVIDE(params.size_n, BLOCK_SIZE_N * n_loop_num); // n方向
    gridDim.x = 1;                                                // k方向

    const int lds_size = BLOCK_SIZE_M * WARP_K * 2 /* + 64 * BLOCK_SIZE_M / 16 */;

    const int shared_mem_size = lds_size; // + BLOCK_SIZE_M * 4 * 2 + 32; // 额外分配sort_token_ids的空间
    const hipStream_t stream = params.stream;

    if (params.is_marlin == false) {
        ;
    } else {

        MOE_W4A8_I8_PERCHANNEL_MARLIN_HIP_NT_DECODE_DOWN<T_hidden, char, WARP_NUM, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_M, WARP_N, WARP_K,
                                                         GROUP_N, GROUP_K, STAGES, mul_topk_weight><<<gridDim, blockDim, shared_mem_size, stream>>>(
            params.ptr_A,
            params.ptr_B0,
            params.ptr_C,
            params.ptr_A_scale,
            params.ptr_B_scale,
            params.topk_weights,
            params.sorted_token_ids,
            params.expert_ids,
            params.num_tokens_post_pad_ptr,
            params.size_m,
            params.size_n,
            params.size_k,
            params.stride_asm,
            params.stride_ask,
            params.stride_bse,
            params.stride_bsn,
            params.stride_bsk,
            params.sorted_token_lens,
            params.top_k,
            params.real_topk);
    }
}

#endif
