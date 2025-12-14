#ifndef __ADDBMM_MOORE_KERNEL_H__
#define __ADDBMM_MOORE_KERNEL_H__

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h> 

#include <type_traits> // 用于 std::is_same_v

namespace op::addbmm::moore {


typedef struct AddbmmOp {
public:
    template <typename T>
    __device__ __forceinline__ void operator()(
        // 当前线程处理的输出坐标 (n, p)
        const int n, 
        const int p,
        
        // 维度信息
        const int B, // Batch size
        const int M, // 中间维度

        // 标量系数
        const float alpha,
        const float beta,

        // 数据指针 (Base Pointers)
        const T* input,
        const T* batch1,
        const T* batch2,
        T* output,

        // Strides (解包传递)
        // input/output: (n, p)
        const int64_t in_s0, const int64_t in_s1,
        const int64_t out_s0, const int64_t out_s1,
        // batch1: (b, n, m)
        const int64_t b1_s0, const int64_t b1_s1, const int64_t b1_s2,
        // batch2: (b, m, p)
        const int64_t b2_s0, const int64_t b2_s1, const int64_t b2_s2
    ) const {
        
       
        float matmul_sum = 0.0f;

        
        int64_t b1_n_offset = n * b1_s1;
        // Batch2 的 p 维度偏移
        int64_t b2_p_offset = p * b2_s2;

        // 遍历 Batch 维度
        for (int b = 0; b < B; ++b) {
            
            // 预计算当前 Batch 的偏移
            int64_t b1_b_offset = b * b1_s0;
            int64_t b2_b_offset = b * b2_s0;

            // 遍历中间维度 M (矩阵乘法)
            for (int m = 0; m < M; ++m) {
                // 计算实际内存偏移
                // Batch1[b, n, m] -> ptr + b*s0 + n*s1 + m*s2
                int64_t offset1 = b1_b_offset + b1_n_offset + m * b1_s2;
                // Batch2[b, m, p] -> ptr + b*s0 + m*s1 + p*s2
                int64_t offset2 = b2_b_offset + m * b2_s1 + b2_p_offset;

                T val1_t = batch1[offset1];
                T val2_t = batch2[offset2];

                float val1_f, val2_f;

                // 类型转换：T -> float
                if constexpr (std::is_same_v<T, half>) {
                    val1_f = __half2float(val1_t);
                    val2_f = __half2float(val2_t);
                } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                    val1_f = __bfloat162float(val1_t);
                    val2_f = __bfloat162float(val2_t);
                } else {
                    val1_f = static_cast<float>(val1_t);
                    val2_f = static_cast<float>(val2_t);
                }

                matmul_sum += val1_f * val2_f;
            }
        }

        
        int64_t in_offset = n * in_s0 + p * in_s1;
        T in_val_t = input[in_offset];
        float in_val_f;

        if constexpr (std::is_same_v<T, half>) {
            in_val_f = __half2float(in_val_t);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            in_val_f = __bfloat162float(in_val_t);
        } else {
            in_val_f = static_cast<float>(in_val_t);
        }

       
        float result_f = beta * in_val_f + alpha * matmul_sum;

        // 4. 写回 Output[n, p]
        int64_t out_offset = n * out_s0 + p * out_s1;

        if constexpr (std::is_same_v<T, half>) {
            output[out_offset] = __float2half(result_f);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[out_offset] = __float2bfloat16(result_f);
        } else {
            output[out_offset] = static_cast<T>(result_f);
        }
    }

} AddbmmOp;

} // namespace op::addbmm::moore

#endif // __ADDBMM_MOORE_KERNEL_H__