#include "prelu_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::prelu::nvidia {

Descriptor::~Descriptor() = default;

static bool build_meta(
    op::prelu::cuda::TensorMeta &meta,
    size_t ndim,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides) {

    if (ndim > static_cast<size_t>(op::prelu::cuda::kPreluMaxDims)) {
        return false;
    }
    if (shape.size() != ndim || strides.size() != ndim) {
        return false;
    }

    meta.ndim = static_cast<int>(ndim);
    for (size_t i = 0; i < static_cast<size_t>(op::prelu::cuda::kPreluMaxDims); ++i) {
        meta.shape[i] = (i < ndim) ? shape[i] : 1;
        meta.strides[i] = (i < ndim) ? strides[i] : 0;
    }
    return true;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    const auto &x_desc = input_desc_vec.at(0);
    const auto &w_desc = input_desc_vec.at(1);

    if (x_desc->dtype() != dtype || w_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    const auto &x_shape = x_desc->shape();
    const auto &y_shape = y_desc->shape();
    if (x_shape != y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t x_ndim = x_desc->ndim();
    const int channel_axis = (x_ndim >= 2) ? 1 : 0;
    const size_t channels = (x_ndim >= 2) ? x_desc->dim(1) : 1;

    WeightMode w_mode = WeightMode::SCALAR;
    ptrdiff_t w_stride0 = 0;

    const size_t w_numel = w_desc->numel();
    if (w_numel == 1) {
        w_mode = WeightMode::SCALAR;
    } else if (w_desc->ndim() == 1 && channels > 0 && w_desc->dim(0) == channels) {
        w_mode = WeightMode::PER_CHANNEL;
        w_stride0 = w_desc->stride(0);
    } else if (w_desc->ndim() == x_ndim && w_desc->shape() == x_shape) {
        w_mode = WeightMode::ELEMENTWISE;
    } else {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(
        dtype,
        x_desc->numel(),
        x_ndim,
        x_shape,
        y_desc->strides(),
        x_desc->strides(),
        w_desc->shape(),
        w_desc->strides(),
        w_mode,
        w_stride0,
        channel_axis,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    std::vector<const void *> inputs,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((numel + kBlock - 1) / kBlock);

    const void *x = inputs.at(0);
    const void *w = inputs.at(1);

    op::prelu::cuda::TensorMeta out_meta{};
    op::prelu::cuda::TensorMeta in_meta{};
    op::prelu::cuda::TensorMeta w_meta{};

    if (!build_meta(out_meta, ndim, shape, y_strides) || !build_meta(in_meta, ndim, shape, x_strides)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (weight_mode == WeightMode::ELEMENTWISE) {
        if (!build_meta(w_meta, ndim, weight_shape, weight_strides)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    } else {
        w_meta.ndim = 0;
    }

    const int weight_mode_i = static_cast<int>(weight_mode);

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        op::prelu::cuda::prelu_kernel<half, float><<<blocks, kBlock, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x),
            reinterpret_cast<const half *>(w),
            numel,
            out_meta,
            in_meta,
            weight_mode_i,
            w_meta,
            weight_stride0,
            channel_axis);
        break;
    case INFINI_DTYPE_BF16:
        op::prelu::cuda::prelu_kernel<cuda_bfloat16, float><<<blocks, kBlock, 0, cuda_stream>>>(
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x),
            reinterpret_cast<const cuda_bfloat16 *>(w),
            numel,
            out_meta,
            in_meta,
            weight_mode_i,
            w_meta,
            weight_stride0,
            channel_axis);
        break;
    case INFINI_DTYPE_F32:
        op::prelu::cuda::prelu_kernel<float, float><<<blocks, kBlock, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            reinterpret_cast<const float *>(w),
            numel,
            out_meta,
            in_meta,
            weight_mode_i,
            w_meta,
            weight_stride0,
            channel_axis);
        break;
    case INFINI_DTYPE_F64:
        op::prelu::cuda::prelu_kernel<double, double><<<blocks, kBlock, 0, cuda_stream>>>(
            reinterpret_cast<double *>(y),
            reinterpret_cast<const double *>(x),
            reinterpret_cast<const double *>(w),
            numel,
            out_meta,
            in_meta,
            weight_mode_i,
            w_meta,
            weight_stride0,
            channel_axis);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::prelu::nvidia

