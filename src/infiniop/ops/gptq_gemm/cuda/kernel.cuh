/*
Adapted from https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa
*/

#include <cstdint>
#include <cstdio>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "compat.cuh"
#include "matrix_view.cuh"
#include "qdq_2.cuh"
#include "qdq_3.cuh"
#include "qdq_4.cuh"
#include "qdq_8.cuh"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
#define MAX_Q_GEMM_ROWS_8BIT 24
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM)
#include <hipblas/hipblas.h>
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transA,
    hipblasOperation_t transB, int m, int n, int k, const half *alpha,
    const half *AP, int lda, const half *BP, int ldb, const half *beta,
    half *CP, int ldc) {
    return hipblasHgemm(handle, transA, transB, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(AP), lda,
                        reinterpret_cast<const hipblasHalf *>(BP), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(CP), ldc);
}
#define hipblasHgemm __compat_hipblasHgemm

// Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
#define rocblas_operation_none HIPBLAS_OP_N
#define rocblas_hgemm __compat_hipblasHgemm
#endif

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half *a_ptr,
                                         const half2 g_result) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half *a_ptr) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half *a_ptr,
                                         const half2 g_result,
                                         const half qs_h) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2 (&dq)[8], const half *a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2 (&dq)[16], const half *a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 16; i += 1) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half *a_ptr,
                                           const float g_result,
                                           const float qs_f) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_16_f(half2 (&dq)[8], const half *a_ptr,
                                            const float g_result,
                                            const float qs_f) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_32_f(half2 (&dq)[16], const half *a_ptr,
                                            const float g_result,
                                            const float qs_f) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 16; i += 1) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    float result_f = __half2float(__low2half(result)) + __half2float(__high2half(result));
    return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ half dot22_8_h(half2 (&dq)[4], const half *a_ptr,
                                          const half g_result,
                                          const half qs_h) {
    // Use FP32 accumulator to avoid potential overflow since unscaled weights are
    // in the range -128..127

    float result = {};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        half2 w01 = dq[i];
        float w0 = __low2float(w01);
        float w1 = __high2float(w01);
        float x0 = __half2float(*a_ptr++);
        float x1 = __half2float(*a_ptr++);
        result = fma(w0, x0, result);
        result = fma(w1, x1, result);
    }
    float qs = __half2float(qs_h);
    result *= qs;
    half result_h = __float2half_rn(result);
    return __hadd(result_h, g_result);
}

__forceinline__ __device__ half dot22_16_h(half2 (&dq)[8], const half *a_ptr,
                                           const half g_result,
                                           const half qs_h) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    half result_h = __hadd(__low2half(result), __high2half(result));
    return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_h(half2 (&dq)[16], const half *a_ptr,
                                           const half g_result,
                                           const half qs_h) {
    half2 result = {};
    const half2 *a2_ptr = (const half2 *)a_ptr;
#pragma unroll
    for (int i = 0; i < 16; i += 1) {
        result = __hfma2(dq[i], *a2_ptr++, result);
    }
    half result_h = __hadd(__low2half(result), __high2half(result));
    return __hfma(result_h, qs_h, g_result);
}

} // namespace gptq
} // namespace vllm
