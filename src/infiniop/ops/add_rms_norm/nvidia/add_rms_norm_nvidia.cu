#include "../../../devices/nvidia/nvidia_common.cuh"
#include "add_rms_norm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
INFINIOP_CUDA_KERNEL add_rmsnormKernel(
    Tdata *__restrict__ y,
    Tdata *__restrict__ residual_out,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_nhead,
    ptrdiff_t stride_residual_out_batch,
    ptrdiff_t stride_residual_out_nhead,
    const Tdata *__restrict__ a,
    ptrdiff_t stride_a_batch,
    ptrdiff_t stride_a_nhead,
    const Tdata *__restrict__ b,
    ptrdiff_t stride_b_batch,
    ptrdiff_t stride_b_nhead,
    const Tweight *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    add_rmsnormBlock<BLOCK_SIZE, Tcompute>(
        y, residual_out,
        stride_y_batch, stride_y_nhead,
        stride_residual_out_batch, stride_residual_out_nhead,
        a, stride_a_batch, stride_a_nhead,
        b, stride_b_batch, stride_b_nhead,
        w, nhead, dim, epsilon);
}

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Ty, typename Ta, typename Tb, typename Tweight>
INFINIOP_CUDA_KERNEL add_rmsnormMixedKernel(
    Ty *__restrict__ y,
    Ty *__restrict__ residual_out,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_nhead,
    ptrdiff_t stride_residual_out_batch,
    ptrdiff_t stride_residual_out_nhead,
    const Ta *__restrict__ a,
    ptrdiff_t stride_a_batch,
    ptrdiff_t stride_a_nhead,
    const Tb *__restrict__ b,
    ptrdiff_t stride_b_batch,
    ptrdiff_t stride_b_nhead,
    const Tweight *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    const size_t batch_idx = blockIdx.x / nhead;
    const size_t head_idx = blockIdx.x % nhead;
    auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
    auto residual_ptr = residual_out
        + batch_idx * stride_residual_out_batch + head_idx * stride_residual_out_nhead;
    auto a_ptr = a + batch_idx * stride_a_batch + head_idx * stride_a_nhead;
    auto b_ptr = b + batch_idx * stride_b_batch + head_idx * stride_b_nhead;

    Tcompute sum_squared = 0;
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        const Tcompute sum_val = Tcompute(a_ptr[i]) + Tcompute(b_ptr[i]);
        residual_ptr[i] = Ty(sum_val);
        sum_squared += sum_val * sum_val;
    }

    using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum_squared = BlockReduce(temp_storage).Sum(sum_squared);
    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tcompute(rsqrtf(sum_squared / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    // Recompute the F32 sum instead of reading the BF16 residual_out.  This
    // keeps the GGUF linear accumulator precision through normalization while
    // restoring the model's ordinary BF16 boundary for subsequent layers.
    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        const Tcompute sum_val = Tcompute(a_ptr[i]) + Tcompute(b_ptr[i]);
        y_ptr[i] = Ty(sum_val * Tcompute(w[i]) * rms);
    }
}

namespace op::add_rms_norm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t residual_out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon) {
    auto result = AddRMSNormInfo::create(y_desc, residual_out_desc, a_desc, b_desc, weight_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// launch kernel with different data types
template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t batch_size, size_t nhead, size_t dim,
    void *y, infiniDtype_t ytype, ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    void *residual_out, ptrdiff_t stride_residual_out_batch, ptrdiff_t stride_residual_out_nhead,
    const void *a, infiniDtype_t atype, ptrdiff_t stride_a_batch, ptrdiff_t stride_a_nhead,
    const void *b, infiniDtype_t btype, ptrdiff_t stride_b_batch, ptrdiff_t stride_b_nhead,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute)                                                                  \
    add_rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight><<<batch_size * nhead, BLOCK_SIZE, 0, cuda_stream>>>( \
        reinterpret_cast<Tdata *>(y),                                                                            \
        reinterpret_cast<Tdata *>(residual_out),                                                                 \
        stride_y_batch,                                                                                          \
        stride_y_nhead,                                                                                          \
        stride_residual_out_batch,                                                                               \
        stride_residual_out_nhead,                                                                               \
        reinterpret_cast<const Tdata *>(a),                                                                      \
        stride_a_batch,                                                                                          \
        stride_a_nhead,                                                                                          \
        reinterpret_cast<const Tdata *>(b),                                                                      \
        stride_b_batch,                                                                                          \
        stride_b_nhead,                                                                                          \
        reinterpret_cast<const Tweight *>(w),                                                                    \
        nhead,                                                                                                   \
        dim,                                                                                                     \
        epsilon)

    if (ytype == INFINI_DTYPE_BF16 && atype == INFINI_DTYPE_F32
        && btype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_BF16) {
        add_rmsnormMixedKernel<BLOCK_SIZE, float, __nv_bfloat16, float, __nv_bfloat16, __nv_bfloat16>
            <<<batch_size * nhead, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(y),
                reinterpret_cast<__nv_bfloat16 *>(residual_out),
                stride_y_batch, stride_y_nhead,
                stride_residual_out_batch, stride_residual_out_nhead,
                reinterpret_cast<const float *>(a), stride_a_batch, stride_a_nhead,
                reinterpret_cast<const __nv_bfloat16 *>(b), stride_b_batch, stride_b_nhead,
                reinterpret_cast<const __nv_bfloat16 *>(w), nhead, dim, epsilon);
    } else if (ytype == INFINI_DTYPE_F32 && atype == INFINI_DTYPE_BF16
        && btype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_BF16) {
        add_rmsnormMixedKernel<BLOCK_SIZE, float, float, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>
            <<<batch_size * nhead, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<float *>(residual_out),
                stride_y_batch, stride_y_nhead,
                stride_residual_out_batch, stride_residual_out_nhead,
                reinterpret_cast<const __nv_bfloat16 *>(a), stride_a_batch, stride_a_nhead,
                reinterpret_cast<const __nv_bfloat16 *>(b), stride_b_batch, stride_b_nhead,
                reinterpret_cast<const __nv_bfloat16 *>(w), nhead, dim, epsilon);
    } else if (ytype == INFINI_DTYPE_F16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float);
    } else if (ytype == INFINI_DTYPE_F16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(half, __nv_bfloat16, float);
    } else if (ytype == INFINI_DTYPE_F16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float);
    } else if (ytype == INFINI_DTYPE_BF16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, __nv_bfloat16, float);
    } else if (ytype == INFINI_DTYPE_BF16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(__nv_bfloat16, half, float);
    } else if (ytype == INFINI_DTYPE_BF16 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(__nv_bfloat16, float, float);
    } else if (ytype == INFINI_DTYPE_F32 && atype == ytype && btype == ytype && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float, float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, void *residual_out, const void *a, const void *b, const void *weight,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_a_batch = _info.a_strides[0];
    auto stride_a_nhead = _info.a_strides[1];
    auto stride_b_batch = _info.b_strides[0];
    auto stride_b_nhead = _info.b_strides[1];
    auto stride_y_batch = _info.y_strides[0];
    auto stride_y_nhead = _info.y_strides[1];
    auto stride_residual_out_batch = _info.residual_out_strides[0];
    auto stride_residual_out_nhead = _info.residual_out_strides[1];
    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    size_t nhead = _info.shape.size() > 2 ? _info.shape[1] : 1;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // launch kernel with different block sizes
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            batch_size, nhead, dim,
            y, _info.ytype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, _info.atype, stride_a_batch, stride_a_nhead,
            b, _info.btype, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            batch_size, nhead, dim,
            y, _info.ytype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, _info.atype, stride_a_batch, stride_a_nhead,
            b, _info.btype, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(
            batch_size, nhead, dim,
            y, _info.ytype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, _info.atype, stride_a_batch, stride_a_nhead,
            b, _info.btype, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            batch_size, nhead, dim,
            y, _info.ytype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, _info.atype, stride_a_batch, stride_a_nhead,
            b, _info.btype, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add_rms_norm::nvidia
