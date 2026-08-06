#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/concat_mla_q.h"

#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

template <typename T>
INFINIOP_CUDA_KERNEL concatMlaQKernel(
    const T *ql_nope,
    const T *q_pe,
    T *q_out,
    size_t rows,
    size_t nope_dim,
    size_t pe_dim) {
    const size_t out_dim = nope_dim + pe_dim;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * out_dim) {
        return;
    }
    const size_t row = index / out_dim;
    const size_t column = index % out_dim;
    q_out[index] = column < nope_dim
                     ? ql_nope[row * nope_dim + column]
                     : q_pe[row * pe_dim + column - nope_dim];
}
} // namespace

__INFINI_C infiniStatus_t infiniopConcatMlaQ(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t ql_nope_desc,
    infiniopTensorDescriptor_t q_pe_desc,
    infiniopTensorDescriptor_t q_out_desc,
    const void *ql_nope,
    const void *q_pe,
    void *q_out,
    void *stream) {
    (void)handle;
    const auto ql_shape = ql_nope_desc->shape();
    const auto pe_shape = q_pe_desc->shape();
    const auto out_shape = q_out_desc->shape();
    CHECK_OR_RETURN(ql_shape.size() == 3 && pe_shape.size() == 3
                        && out_shape.size() == 3,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(ql_shape[0] == pe_shape[0] && ql_shape[0] == out_shape[0]
                        && ql_shape[1] == pe_shape[1] && ql_shape[1] == out_shape[1]
                        && ql_shape[2] + pe_shape[2] == out_shape[2],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto dtype = ql_nope_desc->dtype();
    CHECK_OR_RETURN((dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16
                     || dtype == INFINI_DTYPE_F32)
                        && q_pe_desc->dtype() == dtype
                        && q_out_desc->dtype() == dtype,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(ql_nope_desc->isContiguous() && q_pe_desc->isContiguous()
                        && q_out_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    const size_t rows = ql_shape[0] * ql_shape[1];
    const size_t nope_dim = ql_shape[2];
    const size_t pe_dim = pe_shape[2];
    const size_t total = rows * (nope_dim + pe_dim);
    const size_t blocks = (total + THREADS - 1) / THREADS;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
#define LAUNCH(T)                                                      \
    concatMlaQKernel<T><<<blocks, THREADS, 0, cuda_stream>>>(          \
        static_cast<const T *>(ql_nope), static_cast<const T *>(q_pe), \
        static_cast<T *>(q_out), rows, nope_dim, pe_dim)
    if (dtype == INFINI_DTYPE_F16) {
        LAUNCH(uint16_t);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCH(uint16_t);
    } else {
        LAUNCH(float);
    }
#undef LAUNCH
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
