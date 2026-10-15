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

template <typename T>
struct alignas(16) Vector8 {
    T values[8];
};

template <unsigned int BLOCK_SIZE, typename Tdata>
INFINIOP_CUDA_KERNEL addRmsnormContiguousVectorKernel(
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
    const Tdata *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    const size_t batch_idx = blockIdx.x / nhead;
    const size_t head_idx = blockIdx.x % nhead;
    auto *y_row = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
    auto *residual_row = residual_out
                       + batch_idx * stride_residual_out_batch
                       + head_idx * stride_residual_out_nhead;
    const auto *a_row = a + batch_idx * stride_a_batch + head_idx * stride_a_nhead;
    const auto *b_row = b + batch_idx * stride_b_batch + head_idx * stride_b_nhead;
    const size_t vector_count = dim / 8;

    float sum_squared = 0.0f;
    for (size_t vector_idx = threadIdx.x; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        const auto a_value = reinterpret_cast<const Vector8<Tdata> *>(a_row)[vector_idx];
        const auto b_value = reinterpret_cast<const Vector8<Tdata> *>(b_row)[vector_idx];
        Vector8<Tdata> residual_value;
#pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            const float sum = static_cast<float>(a_value.values[i])
                            + static_cast<float>(b_value.values[i]);
            residual_value.values[i] = static_cast<Tdata>(sum);
            sum_squared += sum * sum;
        }
        reinterpret_cast<Vector8<Tdata> *>(residual_row)[vector_idx] = residual_value;
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    sum_squared = BlockReduce(reduce_storage).Sum(sum_squared);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum_squared / static_cast<float>(dim) + epsilon);
    }
    __syncthreads();

    for (size_t vector_idx = threadIdx.x; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        const auto a_value = reinterpret_cast<const Vector8<Tdata> *>(a_row)[vector_idx];
        const auto b_value = reinterpret_cast<const Vector8<Tdata> *>(b_row)[vector_idx];
        const auto weight = reinterpret_cast<const Vector8<Tdata> *>(w)[vector_idx];
        Vector8<Tdata> output;
#pragma unroll
        for (size_t i = 0; i < 8; ++i) {
            const float sum = static_cast<float>(a_value.values[i])
                            + static_cast<float>(b_value.values[i]);
            output.values[i] = static_cast<Tdata>(
                sum * static_cast<float>(weight.values[i]) * inverse_rms);
        }
        reinterpret_cast<Vector8<Tdata> *>(y_row)[vector_idx] = output;
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
    void *y, infiniDtype_t atype, ptrdiff_t stride_y_batch, ptrdiff_t stride_y_nhead,
    void *residual_out, ptrdiff_t stride_residual_out_batch, ptrdiff_t stride_residual_out_nhead,
    const void *a, ptrdiff_t stride_a_batch, ptrdiff_t stride_a_nhead,
    const void *b, ptrdiff_t stride_b_batch, ptrdiff_t stride_b_nhead,
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

    if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, half, float);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(half, __nv_bfloat16, float);
    } else if (atype == INFINI_DTYPE_F16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(half, float, float);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__nv_bfloat16, __nv_bfloat16, float);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(__nv_bfloat16, half, float);
    } else if (atype == INFINI_DTYPE_BF16 && wtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(__nv_bfloat16, float, float);
    } else if (atype == INFINI_DTYPE_F32 && wtype == INFINI_DTYPE_F32) {
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

#ifdef ENABLE_HYGON_API
    const bool vectorized = device_type == INFINI_DEVICE_HYGON
                         && _info.atype == _info.wtype
                         && dim % 8 == 0
                         && stride_y_batch % 8 == 0
                         && stride_residual_out_batch % 8 == 0
                         && stride_a_batch % 8 == 0
                         && stride_b_batch % 8 == 0
                         && (nhead == 1 || (stride_y_nhead % 8 == 0 && stride_residual_out_nhead % 8 == 0 && stride_a_nhead % 8 == 0 && stride_b_nhead % 8 == 0));
    if (vectorized) {
        constexpr unsigned int BLOCK_SIZE = 1024;
        const auto grid_size = batch_size * nhead;
#define LAUNCH_VECTORIZED(Tdata)                                                                    \
    addRmsnormContiguousVectorKernel<BLOCK_SIZE, Tdata><<<grid_size, BLOCK_SIZE, 0, cuda_stream>>>( \
        static_cast<Tdata *>(y), static_cast<Tdata *>(residual_out),                                \
        stride_y_batch, stride_y_nhead,                                                             \
        stride_residual_out_batch, stride_residual_out_nhead,                                       \
        static_cast<const Tdata *>(a), stride_a_batch, stride_a_nhead,                              \
        static_cast<const Tdata *>(b), stride_b_batch, stride_b_nhead,                              \
        static_cast<const Tdata *>(weight), nhead, dim, _info.epsilon)
        if (_info.atype == INFINI_DTYPE_F16) {
            LAUNCH_VECTORIZED(half);
            return INFINI_STATUS_SUCCESS;
        }
        if (_info.atype == INFINI_DTYPE_BF16) {
            LAUNCH_VECTORIZED(__nv_bfloat16);
            return INFINI_STATUS_SUCCESS;
        }
#undef LAUNCH_VECTORIZED
    }
#endif

    // launch kernel with different block sizes
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            batch_size, nhead, dim,
            y, _info.atype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, stride_a_batch, stride_a_nhead,
            b, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            batch_size, nhead, dim,
            y, _info.atype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, stride_a_batch, stride_a_nhead,
            b, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(
            batch_size, nhead, dim,
            y, _info.atype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, stride_a_batch, stride_a_nhead,
            b, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            batch_size, nhead, dim,
            y, _info.atype, stride_y_batch, stride_y_nhead,
            residual_out, stride_residual_out_batch, stride_residual_out_nhead,
            a, stride_a_batch, stride_a_nhead,
            b, stride_b_batch, stride_b_nhead,
            weight, _info.wtype, _info.epsilon, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::add_rms_norm::nvidia
