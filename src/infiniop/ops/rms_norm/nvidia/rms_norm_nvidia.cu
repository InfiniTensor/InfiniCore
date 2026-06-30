#include "../../../devices/nvidia/nvidia_common.cuh"
#include "rms_norm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

#ifdef ENABLE_NINETOOTHED
#include "../../../../../build/ninetoothed/rms_norm.h"
#endif

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
INFINIOP_CUDA_KERNEL rmsnormKernel(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_nhead,
    const Tdata *__restrict__ x,
    ptrdiff_t stride_x_batch,
    ptrdiff_t stride_x_nhead,
    const Tweight *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    rmsnormBlock<BLOCK_SIZE, Tcompute>(y, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, nhead, dim, epsilon);
}

namespace op::rms_norm::nvidia {

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
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
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
    const void *x, ptrdiff_t stride_x_batch, ptrdiff_t stride_x_nhead,
    const void *w, infiniDtype_t wtype,
    float epsilon,
    cudaStream_t cuda_stream) {

#define LAUNCH_KERNEL(Tdata, Tweight, Tcompute)                                                              \
    rmsnormKernel<BLOCK_SIZE, Tcompute, Tdata, Tweight><<<batch_size * nhead, BLOCK_SIZE, 0, cuda_stream>>>( \
        reinterpret_cast<Tdata *>(y),                                                                        \
        stride_y_batch,                                                                                      \
        stride_y_nhead,                                                                                      \
        reinterpret_cast<const Tdata *>(x),                                                                  \
        stride_x_batch,                                                                                      \
        stride_x_nhead,                                                                                      \
        reinterpret_cast<const Tweight *>(w),                                                                \
        nhead,                                                                                               \
        dim,                                                                                                 \
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
    void *y, const void *x, const void *w,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto stride_x_batch = _info.x_strides[0];
    auto stride_x_nhead = _info.x_strides[1];
    auto stride_y_batch = _info.y_strides[0];
    auto stride_y_nhead = _info.y_strides[1];
    auto dim = _info.dim();
    uint32_t batch_size = static_cast<uint32_t>(_info.shape[0]);
    size_t nhead = _info.shape.size() > 2 ? _info.shape[1] : 1;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#ifndef ENABLE_NINETOOTHED
    // launch kernel with different block sizes
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, cuda_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(batch_size, nhead, dim, y, _info.atype, stride_y_batch, stride_y_nhead, x, stride_x_batch, stride_x_nhead, w, _info.wtype, _info.epsilon, cuda_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
#else
    const auto &ndim{_info.ndim()};
    uint64_t dim_ = dim;

    std::vector<uint64_t> empty_shape_vec(ndim);
    std::vector<int64_t> empty_strides_vec(ndim);
    const auto &empty_shape{empty_shape_vec.data()};
    const auto &empty_strides{empty_strides_vec.data()};

    auto &x_shape_vec{_info.shape};
    auto &x_strides_vec{_info.x_strides};
    auto x_data{x};
    auto x_shape{x_shape_vec.data()};
    auto x_strides{x_strides_vec.data()};
    const NineToothedTensor input{const_cast<void *>(x_data), const_cast<uint64_t *>(x_shape), const_cast<int64_t *>(x_strides)};
    auto &w_shape_vec{_info.shape};
    std::vector<int64_t> w_strides_vec(ndim);
    w_strides_vec[ndim - 1] = 1;
    auto w_data{w};
    auto w_shape{w_shape_vec.data()};
    auto w_strides{w_strides_vec.data()};
    const NineToothedTensor weight{const_cast<void *>(w_data), const_cast<uint64_t *>(w_shape), const_cast<int64_t *>(w_strides)};
    const NineToothedTensor eps{const_cast<float *>(&_info.epsilon), empty_shape, empty_strides};
    auto &y_shape_vec{_info.shape};
    auto &y_strides_vec{_info.y_strides};
    auto y_data{y};
    auto y_shape{y_shape_vec.data()};
    auto y_strides{y_strides_vec.data()};
    const NineToothedTensor output{y_data, const_cast<uint64_t *>(y_shape), const_cast<int64_t *>(y_strides)};
    const NineToothedTensor num_normalized_elements{const_cast<uint64_t *>(&dim_), empty_shape, empty_strides};
    constexpr auto num_normalized_dims{1};
    constexpr auto block_size{1024};

    if (launch_rms_norm(stream, input, weight, eps, output, num_normalized_elements, ndim, num_normalized_dims, _info.atype, _info.wtype, _info.atype, block_size)) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
#endif
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm::nvidia
