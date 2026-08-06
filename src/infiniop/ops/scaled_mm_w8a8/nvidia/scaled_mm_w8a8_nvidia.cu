#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "scaled_mm_w8a8_nvidia.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(ENABLE_HYGON_API)
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <tuple>
#endif

namespace {
constexpr size_t THREADS = 256;

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
INFINIOP_CUDA_KERNEL scaledMmW8A8Kernel(
    T *__restrict__ out,
    const int8_t *__restrict__ a,
    const int8_t *__restrict__ b,
    const float *__restrict__ a_scales,
    const float *__restrict__ b_scales,
    const T *__restrict__ bias,
    size_t m,
    size_t n,
    size_t k,
    bool trans_weight) {
    const size_t row = blockIdx.y;
    const size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    int32_t acc = 0;
    for (size_t inner = 0; inner < k; ++inner) {
        const int8_t weight = trans_weight
                                ? b[column * k + inner]
                                : b[inner * n + column];
        acc += static_cast<int32_t>(a[row * k + inner])
             * static_cast<int32_t>(weight);
    }
    float value = static_cast<float>(acc) * a_scales[row] * b_scales[column];
    if (bias != nullptr) {
        value += static_cast<float>(bias[column]);
    }
    out[row * n + column] = from_float<T>(value);
}

#if defined(ENABLE_HYGON_API)
struct HygonLtPlan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t input_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
};

using HygonLtPlanKey = std::tuple<int, size_t, size_t, size_t>;

HygonLtPlan *createHygonLtPlan(size_t m, size_t n, size_t k) {
    auto *plan = new HygonLtPlan();
    if (cublasLtCreate(&plan->handle) != CUBLAS_STATUS_SUCCESS
        || cublasLtMatmulDescCreate(
               &plan->operation, CUBLAS_COMPUTE_32I, CUDA_R_32I)
               != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(
               &plan->weight_layout, CUDA_R_8I, n, k, n)
               != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(
               &plan->input_layout, CUDA_R_8I, k, m, k)
               != CUBLAS_STATUS_SUCCESS
        || cublasLtMatrixLayoutCreate(
               &plan->output_layout, CUDA_R_32I, n, m, n)
               != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }

    cublasLtMatmulPreference_t preference = nullptr;
    if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }
    size_t max_workspace_size = 0;
    if (cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &max_workspace_size,
            sizeof(max_workspace_size))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(preference);
        return nullptr;
    }

    constexpr int MAX_ALGORITHMS = 32;
    cublasLtMatmulHeuristicResult_t results[MAX_ALGORITHMS];
    int returned_results = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        plan->handle,
        plan->operation,
        plan->weight_layout,
        plan->input_layout,
        plan->output_layout,
        plan->output_layout,
        preference,
        MAX_ALGORITHMS,
        results,
        &returned_results);
    cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS || returned_results == 0) {
        return nullptr;
    }
    plan->algorithm = results[0].algo;
    return plan;
}

HygonLtPlan *getHygonLtPlan(size_t m, size_t n, size_t k) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return nullptr;
    }
    static auto *plans = new std::map<HygonLtPlanKey, HygonLtPlan *>();
    static auto *mutex = new std::mutex();
    const HygonLtPlanKey key{device, m, n, k};
    std::lock_guard<std::mutex> guard(*mutex);
    const auto it = plans->find(key);
    if (it != plans->end()) {
        return it->second;
    }
    auto *plan = createHygonLtPlan(m, n, k);
    plans->emplace(key, plan);
    return plan;
}

template <typename T>
INFINIOP_CUDA_KERNEL scaledMmW8A8PostprocessKernel(
    T *__restrict__ out,
    const int32_t *__restrict__ acc,
    const float *__restrict__ a_scales,
    const float *__restrict__ b_scales,
    const T *__restrict__ bias,
    size_t elements,
    size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const size_t row = index / n;
    const size_t column = index - row * n;
    float value = static_cast<float>(acc[index])
                * a_scales[row] * b_scales[column];
    if (bias != nullptr) {
        value += static_cast<float>(bias[column]);
    }
    out[index] = from_float<T>(value);
}

bool runHygonLt(
    void *out,
    const void *a,
    const void *b,
    const void *a_scales,
    const void *b_scales,
    const void *bias,
    size_t m,
    size_t n,
    size_t k,
    infiniDtype_t out_dtype,
    cudaStream_t stream) {
    auto *plan = getHygonLtPlan(m, n, k);
    if (plan == nullptr) {
        return false;
    }

    int32_t *acc = nullptr;
    const size_t acc_size = m * n * sizeof(int32_t);
    if (cudaMallocAsync(reinterpret_cast<void **>(&acc), acc_size, stream)
        != cudaSuccess) {
        return false;
    }
    const int32_t alpha = 1;
    const int32_t beta = 0;
    const auto status = cublasLtMatmul(
        plan->handle,
        plan->operation,
        &alpha,
        b,
        plan->weight_layout,
        a,
        plan->input_layout,
        &beta,
        acc,
        plan->output_layout,
        acc,
        plan->output_layout,
        &plan->algorithm,
        nullptr,
        0,
        stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(acc, stream);
        return false;
    }

    const size_t elements = m * n;
    const dim3 grid(static_cast<unsigned int>((elements + THREADS - 1) / THREADS));
    if (out_dtype == INFINI_DTYPE_F16) {
        scaledMmW8A8PostprocessKernel<half><<<grid, THREADS, 0, stream>>>(
            static_cast<half *>(out), acc,
            static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales),
            static_cast<const half *>(bias), elements, n);
    } else {
        scaledMmW8A8PostprocessKernel<__nv_bfloat16><<<grid, THREADS, 0, stream>>>(
            static_cast<__nv_bfloat16 *>(out), acc,
            static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales),
            static_cast<const __nv_bfloat16 *>(bias), elements, n);
    }
    const bool ok = cudaGetLastError() == cudaSuccess;
    cudaFreeAsync(acc, stream);
    return ok;
}
#endif
} // namespace

namespace op::scaled_mm_w8a8::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t a_scales_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight) {
    const auto out_shape = out_desc->shape();
    const auto a_shape = a_desc->shape();
    const auto b_shape = b_desc->shape();
    const auto as_shape = a_scales_desc->shape();
    const auto bs_shape = b_scales_desc->shape();
    CHECK_OR_RETURN(out_shape.size() == 2 && a_shape.size() == 2 && b_shape.size() == 2
                        && as_shape.size() == 2 && bs_shape.size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t m = a_shape[0];
    const size_t k = a_shape[1];
    const size_t n = trans_weight ? b_shape[0] : b_shape[1];
    CHECK_OR_RETURN(out_shape[0] == m && out_shape[1] == n
                        && ((!trans_weight && b_shape[0] == k)
                            || (trans_weight && b_shape[1] == k))
                        && as_shape[0] == m && as_shape[1] == 1
                        && bs_shape[0] == n && bs_shape[1] == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(out_desc->isContiguous() && a_desc->isContiguous()
                        && b_desc->isContiguous() && a_scales_desc->isContiguous()
                        && b_scales_desc->isContiguous()
                        && (bias_desc == nullptr || bias_desc->isContiguous()),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(a_desc->dtype() == INFINI_DTYPE_I8 && b_desc->dtype() == INFINI_DTYPE_I8
                        && a_scales_desc->dtype() == INFINI_DTYPE_F32
                        && b_scales_desc->dtype() == INFINI_DTYPE_F32
                        && (out_desc->dtype() == INFINI_DTYPE_F16 || out_desc->dtype() == INFINI_DTYPE_BF16)
                        && (bias_desc == nullptr || (bias_desc->dtype() == out_desc->dtype() && bias_desc->shape().size() == 1 && bias_desc->shape()[0] == n)),
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    *desc_ptr = new Descriptor(m, n, k, out_desc->dtype(), trans_weight, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *out, const void *a, const void *b, const void *a_scales,
    const void *b_scales, const void *bias, void *stream) const {
    const dim3 grid(static_cast<unsigned int>((_n + THREADS - 1) / THREADS),
                    static_cast<unsigned int>(_m));
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#if defined(ENABLE_HYGON_API)
    if (!_trans_weight
        && runHygonLt(
            out, a, b, a_scales, b_scales, bias,
            _m, _n, _k, _out_dtype, cuda_stream)) {
        return INFINI_STATUS_SUCCESS;
    }
#endif
    if (_out_dtype == INFINI_DTYPE_F16) {
        scaledMmW8A8Kernel<half><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<half *>(out), static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b), static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales), static_cast<const half *>(bias),
            _m, _n, _k, _trans_weight);
    } else {
        scaledMmW8A8Kernel<__nv_bfloat16><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16 *>(out), static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b), static_cast<const float *>(a_scales),
            static_cast<const float *>(b_scales), static_cast<const __nv_bfloat16 *>(bias),
            _m, _n, _k, _trans_weight);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::scaled_mm_w8a8::nvidia
