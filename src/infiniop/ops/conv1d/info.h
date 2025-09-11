#ifndef __CONV1D_INFO_H__
#define __CONV1D_INFO_H__

#include "../../../utils.h"
#include "../../../utils/result.hpp"
#include "../../operator.h"
#include "../../tensor.h"

#ifdef ENABLE_CUDA_API
#include "../../devices/nvidia/nvidia_handle.cuh"
#endif

namespace op::conv1d {
class Conv1dInfo;
} // namespace op::conv1d

namespace op::conv1d {

class Conv1dInfo {
private:
    std::vector<size_t> _meta;
    size_t _ndim;
    size_t _batch;
    size_t _in_channels;
    size_t _out_channels;
    size_t _spatial_sizes;
    size_t _bias_dims_size;
    size_t _padded_shape_size;
    bool _gated;  // Whether this is a gated conv1d (output channels = 2 * actual output)
    size_t _groups;  // Number of groups for grouped/depthwise convolution

    Conv1dInfo(std::vector<size_t> meta,
             size_t ndim,
             size_t batch,
             size_t in_channels,
             size_t out_channels,
             size_t spatial_sizes,
             size_t bias_dims_size,
             size_t padded_shape_size,
             bool gated,
             size_t groups)
        : _meta(std::move(meta)),
          _ndim(ndim),
          _batch(batch),
          _in_channels(in_channels),
          _out_channels(out_channels),
          _spatial_sizes(spatial_sizes),
          _bias_dims_size(bias_dims_size),
          _padded_shape_size(padded_shape_size),
          _gated(gated),
          _groups(groups) {}

public:
    inline size_t ndim() const { return _ndim; }
    inline size_t batch() const { return _batch; }
    inline size_t in_channels() const { return _in_channels; }
    inline size_t out_channels() const { return _out_channels; }
    inline size_t spatial_sizes() const { return _spatial_sizes; }
    inline size_t bias_dims_size() const { return _bias_dims_size; }
    inline size_t padded_shape_size() const { return _padded_shape_size; }
    inline bool gated() const { return _gated; }
    inline size_t groups() const { return _groups; }

    inline size_t getMetaMemSize() const {
        return _meta.size() * sizeof(size_t);
    }
    inline const int8_t *getMetaStart() const {
        return reinterpret_cast<const int8_t *>(_meta.data());
    }

    inline const size_t *getInputDims() const { return _meta.data(); }
    inline const size_t *getKernelDims() const { return getInputDims() + _ndim; }
    inline const size_t *getOutputDims() const { return getKernelDims() + _ndim; }
    inline const size_t *getBiasDims() const { return getOutputDims() + _ndim; }
    inline const size_t *getPadsInfo() const { return getBiasDims() + _bias_dims_size; }
    inline const ptrdiff_t *getStridesInfo() const { return reinterpret_cast<const ptrdiff_t *>(getPadsInfo()) + _ndim; }
    inline const size_t *getDilationsInfo() const { return reinterpret_cast<const size_t *>(getStridesInfo()) + _ndim; }
    inline const size_t *getPaddedShape() const { return getDilationsInfo() + _ndim; }
    inline const size_t *getGroupsInfo() const { return getPaddedShape() + _padded_shape_size; }

    inline size_t input_dim(size_t i) const {
        return i < _ndim ? getInputDims()[i] : 0;
    }
    inline size_t kernel_dim(size_t i) const {
        return i < _ndim ? getKernelDims()[i] : 0;
    }
    inline size_t output_dim(size_t i) const {
        return i < _ndim ? getOutputDims()[i] : 0;
    }
    inline size_t bias_dim(size_t i) const {
        return i < _bias_dims_size ? getBiasDims()[i] : 0;
    }
    inline size_t pad_info(size_t i) const {
        return i < _ndim ? getPadsInfo()[i] : 0;
    }
    inline ptrdiff_t stride_info(size_t i) const {
        return i < _ndim ? getStridesInfo()[i] : 0;
    }
    inline size_t dilation_info(size_t i) const {
        return i < _ndim ? getDilationsInfo()[i] : 0;
    }
    inline size_t padded_shape_dim(size_t i) const {
        return i < _padded_shape_size ? getPaddedShape()[i] : 0;
    }

    using ResultType = utils::Result<Conv1dInfo>;

    static ResultType create(
        infiniopHandle_t handle_,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n,
        size_t groups = 1);
};

inline utils::Result<size_t> calculateConv1dOutputSize(
    size_t input_size,
    size_t kernel_size,
    size_t padding,
    size_t stride,
    size_t dilation) {
    if (stride == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (dilation == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (kernel_size == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    size_t effective_kernel = dilation * (kernel_size - 1) + 1;

    size_t padded_input = input_size + 2 * padding;

    if (padded_input < effective_kernel) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t output_size = (padded_input - effective_kernel) / stride + 1;

    return utils::Result<size_t>(output_size);
}

inline utils::Result<Conv1dInfo> Conv1dInfo::create(
    infiniopHandle_t /*handle_*/,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n,
    size_t groups) {

    auto dtype = y_desc->dtype();
    if (dtype != x_desc->dtype() || dtype != w_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    const size_t ndim = n;
    const size_t new_dims = n + 2; // For conv1d test: x/y use [B, L, C], w [OC_raw, Cin_per_group, K]

    if (x_desc->ndim() < new_dims || y_desc->ndim() < new_dims || w_desc->ndim() < new_dims) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Test harness expects x:[B, L, Cin], y:[B, Lout, Cout]
    const size_t batch = x_desc->shape()[0];
    const size_t in_channels = x_desc->shape()[2];
    const size_t out_channels_raw = w_desc->shape()[0];

    // Gated detection: weight OC = 2 * y.C
    bool gated = false;
    size_t out_channels = out_channels_raw;
    if (out_channels_raw % 2 == 0 && y_desc->shape()[2] == out_channels_raw / 2) {
        gated = true;
        out_channels = out_channels_raw / 2;
    }

    if (y_desc->shape()[0] != batch) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (y_desc->shape()[2] != out_channels) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Infer groups if not explicitly compatible with 1
    // For grouped conv: w.Cin_per_group = in_channels / groups
    // Infer groups = in_channels / w.shape[1] when divisible and OC divisible by groups
    size_t inferred_groups = groups;
    if (inferred_groups == 0) inferred_groups = 1;
    const size_t w_cin = w_desc->shape()[1];
    if (inferred_groups == 1) {
        if (w_cin > 0 && in_channels % w_cin == 0) {
            size_t g = in_channels / w_cin;
            if (g > 0 && out_channels_raw % g == 0) {
                inferred_groups = g;
            }
        }
    }
    // Validate groups
    if (inferred_groups == 0 || in_channels % inferred_groups != 0 || out_channels_raw % inferred_groups != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Pointers to arrays
    const size_t *pads_ptr = reinterpret_cast<const size_t *>(pads);
    const ptrdiff_t *strides_ptr = reinterpret_cast<const ptrdiff_t *>(strides);
    const size_t *dilations_ptr = reinterpret_cast<const size_t *>(dilations);

    bool has_padding = false;
    if (pads_ptr != nullptr) {
        for (size_t i = 0; i < ndim; ++i) {
            if (pads_ptr[i] > 0) { has_padding = true; break; }
        }
    }
    const size_t padded_shape_size = has_padding ? (ndim + 2) : 0;
    const size_t bias_dims_size = (b_desc != nullptr) ? x_desc->ndim() : 0;

    // meta layout (match conv/info.h style for spatial dims):
    // [input_dims(ndim), kernel_dims(ndim), output_dims(ndim), bias_dims(bias_dims_size), pads(ndim), strides(ndim), dilations(ndim), padded_shape(ndim+2)?, groups(1)]
    const size_t meta_size = ndim * 6 + bias_dims_size + padded_shape_size + 1;
    std::vector<size_t> meta(meta_size);

    size_t *input_dims = meta.data();
    size_t *kernel_dims = input_dims + ndim;
    size_t *output_dims = kernel_dims + ndim;
    size_t *bias_dims = output_dims + ndim;
    size_t *pads_info = bias_dims + bias_dims_size;
    ptrdiff_t *strides_info = reinterpret_cast<ptrdiff_t *>(pads_info) + ndim;
    size_t *dilations_info = reinterpret_cast<size_t *>(strides_info) + ndim;
    size_t *padded_shape = dilations_info + ndim;

    size_t spatial_sizes = 1;
    // 1D: spatial dim index is 0; map x:[B,L,C], y:[B,Lout,Cout], w:[OC_raw,Cin_per_group,K]
    for (size_t i = 0; i < ndim; ++i) {
        input_dims[i] = x_desc->shape()[1];
        kernel_dims[i] = w_desc->shape()[2];
        output_dims[i] = y_desc->shape()[1];
        pads_info[i] = pads_ptr ? pads_ptr[i] : 0;
        strides_info[i] = strides_ptr ? strides_ptr[i] : 1;
        dilations_info[i] = dilations_ptr ? dilations_ptr[i] : 1;
        spatial_sizes *= output_dims[i];

        auto output_result = calculateConv1dOutputSize(
            input_dims[i], kernel_dims[i], pads_info[i], static_cast<size_t>(strides_info[i]), dilations_info[i]);
        CHECK_RESULT(output_result);
        size_t expected = output_result.take();
        if (output_dims[i] != expected) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    if (bias_dims_size > 0) {
        std::fill(bias_dims, bias_dims + bias_dims_size, 1);
        // Bias is [Cout] typically
        bias_dims[1] = gated ? out_channels * 2 : out_channels;
    }

    if (padded_shape_size > 0) {
        // Store a [B,C,L_padded]-like tuple for convenience
        padded_shape[0] = batch;
        padded_shape[1] = in_channels;
        for (size_t i = 0; i < ndim; ++i) {
            padded_shape[i + 2] = input_dims[i] + 2 * pads_info[i];
        }
    }

    // Store groups at the end
    meta[meta_size - 1] = inferred_groups;

    Conv1dInfo info(
        std::move(meta), ndim, batch, in_channels, out_channels,
        spatial_sizes, bias_dims_size, padded_shape_size, gated, inferred_groups);

    return utils::Result<Conv1dInfo>(std::move(info));
}

} // namespace op::conv1d

#endif // __CONV1D_INFO_H__
