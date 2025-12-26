#include "../../../devices/nvidia/nvidia_common.cuh"
#include "linear_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>
#include <cublasLt.h>

#include "../cuda/kernel.cuh"

#if defined ENABLE_NVIDIA_API
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
void int8Gemm(
    const int8_t *x_packed, const int8_t *w_packed, int32_t *y_packed,
    int M, int N, int K, cudaStream_t stream) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = int32_t;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // Use SIMT opclass to avoid tensor-op interleaved layout requirements
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementC, // accumulator type
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm75>;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args{
        problem_size,
        {x_packed, K},
        {w_packed, N},
        {y_packed, N},
        {y_packed, N},
        {1, 0}};

    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        printf("[CUTLASS SIMT] initialize failed: %d\n", int(status));
        return;
    }
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        printf("[CUTLASS SIMT] run failed: %d\n", int(status));
        return;
    }
}
#endif

template <typename Tdata>
INFINIOP_CUDA_KERNEL post(
    Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta) {
    postKernel<Tdata>(y, y_packed, c, bias, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
}
template <typename Tdata>
INFINIOP_CUDA_KERNEL post(
    Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, int M, int K, int N, float alpha, float beta) {
    postKernel<Tdata>(y, y_packed, c, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
}

template <typename Tdata>
INFINIOP_CUDA_KERNEL postSym(
    Tdata *y, int32_t *y_packed, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta) {
    postSymKernel<Tdata>(y, y_packed, c, bias, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
}
template <typename Tdata>
INFINIOP_CUDA_KERNEL postSym(
    Tdata *y, int32_t *y_packed, const Tdata *c, const int8_t *x_packed, const Tdata *x_scale, const int8_t *w_packed, const Tdata *w_scale, int M, int K, int N, float alpha, float beta) {
    postSymKernel<Tdata>(y, y_packed, c, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
}

namespace op::linear::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t d_desc,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t x_scale_desc,
    infiniopTensorDescriptor_t x_zero_desc,
    infiniopTensorDescriptor_t weights_desc,
    infiniopTensorDescriptor_t weights_scale_desc,
    infiniopTensorDescriptor_t weights_zero_desc,
    float alpha,
    float beta) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto info = LinearInfo::createLinearInfo(d_desc, c_desc, bias_desc, x_desc, x_scale_desc, x_zero_desc, weights_desc, weights_scale_desc, weights_zero_desc, alpha, beta);
    CHECK_RESULT(info);
    size_t workspace_size = c_desc->dim(0) * c_desc->dim(1) * sizeof(int32_t);
    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t Descriptor::launchKernel(const LinearInfo &info, Tdata *y, const Tdata *c, const Tdata *bias, const int8_t *x_packed, const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed, const Tdata *w_scale, const Tdata *w_zero, void *stream_, void *workspace) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    int M = (int)info.M;
    int K = (int)info.K;
    int N = (int)info.N;
    float alpha = info.alpha;
    float beta = info.beta;
    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    int32_t *y_packed = reinterpret_cast<int32_t *>(workspace_ptr);
#if defined ENABLE_NVIDIA_API
    int8Gemm(x_packed, w_packed, y_packed, M, N, K, stream);
#elif defined ENABLE_QY_API
    const int32_t alpha_I = 1;
    const int32_t beta_I = 0;
    CHECK_STATUS(this->_opaque->internal->useCublas(
        stream,
        [&](cublasHandle_t handle) {
            CHECK_CUBLAS(cublasGemmEx(
                handle,
                CUBLAS_OP_N, // A = w_packed, column-major view
                CUBLAS_OP_N, // B = x_packed, column-major view
                N,           // m = N
                M,           // n = M
                K,           // k = K
                &alpha_I,
                w_packed, CUDA_R_8I, N, // lda = m = N
                x_packed, CUDA_R_8I, K, // ldb = k = K
                &beta_I,
                y_packed, CUDA_R_32I, N, // ldc = m = N
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT));
            return INFINI_STATUS_SUCCESS;
        }));
#endif
    constexpr unsigned int BLOCK_SIZE_x = 32;
    constexpr unsigned int BLOCK_SIZE_y = 32;

    int num_block_x = (N + BLOCK_SIZE_x - 1) / BLOCK_SIZE_x;
    int num_block_y = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
    dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    if (bias == nullptr) {
        if (x_zero == nullptr && w_zero == nullptr) {
            postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, c, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
        } else {
            post<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, c, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
        }
    } else {
        if (x_zero == nullptr && w_zero == nullptr) {
            postSym<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, c, bias, x_packed, x_scale, w_packed, w_scale, M, K, N, alpha, beta);
        } else {
            post<Tdata><<<grid_dim, block_dim, 0, stream>>>(y, y_packed, c, bias, x_packed, x_scale, x_zero, w_packed, w_scale, w_zero, M, K, N, alpha, beta);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *d,
                                     const void *c,
                                     const void *bias,
                                     const void *x,
                                     const void *x_scale,
                                     const void *x_zero,
                                     const void *weights,
                                     const void *weights_scale,
                                     const void *weights_zero,
                                     void *stream) const {
#define CALCULATE_LINEAR(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)d, (const TDATA *)c, (const TDATA *)bias, (const int8_t *)x, (const TDATA *)x_scale, (const TDATA *)x_zero, (const int8_t *)weights, (const TDATA *)weights_scale, (const TDATA *)weights_zero, stream, workspace)
#define CALCULATE_LINEAR_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                           \
        if (_info.dtype == INFINI_DTYPE_F16)                    \
            return CALCULATE_LINEAR(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)               \
            return CALCULATE_LINEAR(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)              \
            return CALCULATE_LINEAR(BLOCK_SIZE, __nv_bfloat16); \
        else                                                    \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;              \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::linear::nvidia
