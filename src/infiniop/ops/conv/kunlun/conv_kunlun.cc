#include "conv_kunlun.h"
#include "../../../../utils.h"
#include "../../../devices/kunlun/kunlun_common.h"
#include "../../../devices/kunlun/kunlun_handle.h"

namespace op::conv::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
                                   pads, strides, dilations, n);

    CHECK_RESULT(result);
    auto conv_info = result.take();
    size_t min_workspace_size = conv_info.bias_dims_size() * sizeof(float);
    *desc_ptr = new Descriptor(
        dtype,
        conv_info,
        min_workspace_size,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t conv_kernel(
    std::shared_ptr<device::kunlun::Handle::Internal> internal,
    const ConvInfo &info,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) {
    char *workspace_value = reinterpret_cast<char *>(workspace);
    int64_t bias_ndims = info.bias_dims_size();
    int64_t bias_size = 1;
    if (bias_ndims > 0) {
        for (int64_t i = 0; i < bias_ndims; i++) {
            bias_size *= info.bias_dim(i);
        }
    } else {
        bias_size = 0;
    }
    float *bias_F32 = (float *)workspace_value;
    CHECK_STATUS(internal->useXdnn(
        (kunlunStream_t)stream,
        [&](xdnnHandle_t handle) {
            if (bias_size > 0) {
                if constexpr (std::is_same<Tdata, float16>::value) {
                    CHECK_KUNLUN((xdnn::cast<Tdata, float>(handle, (Tdata *)bias, bias_F32, bias_size)));
                } else if constexpr (std::is_same<Tdata, float>::value) {
                    bias_F32 = (float *)bias;
                }
            } else {
                bias_F32 = nullptr;
            }
            return INFINI_STATUS_SUCCESS;
        }));
    switch (info.ndim()) {
    case 1: {
        int64_t ksize = (int64_t)info.kernel_dim(0);
        int64_t stride = (int64_t)info.stride_info(0);
        std::initializer_list<int64_t> pad = {(int64_t)info.pad_info(0)};
        int64_t dilation = (int64_t)info.dilation_info(0);

        CHECK_STATUS(internal->useXdnn(
            (kunlunStream_t)stream,
            [&](xdnnHandle_t handle) {
                CHECK_KUNLUN((xdnn::conv1d_fusion<Tdata, Tdata, Tdata, int16_t>(handle, (Tdata *)x, (Tdata *)w, (Tdata *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                (int64_t)info.out_channels(), ksize,
                                                                                stride, pad,
                                                                                dilation, 1, nullptr,
                                                                                nullptr, nullptr, true, bias_F32,
                                                                                nullptr, xdnn::Activation_t::LINEAR,
                                                                                nullptr)));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    }
    case 2: {
        std::vector<int64_t> ksize = {(int64_t)info.kernel_dim(0), (int64_t)info.kernel_dim(1)};
        std::vector<int64_t> stride = {(int64_t)info.stride_info(0), (int64_t)info.stride_info(1)};
        std::vector<int64_t> pad = {
            (int64_t)info.pad_info(0),
            (int64_t)info.pad_info(1)};
        std::vector<int64_t> dilation = {(int64_t)info.dilation_info(0), (int64_t)info.dilation_info(1)};
        CHECK_STATUS(internal->useXdnn(
            (kunlunStream_t)stream,
            [&](xdnnHandle_t handle) {
                CHECK_KUNLUN((xdnn::conv2d_fusion<Tdata, Tdata, Tdata, int16_t>(handle, (Tdata *)x, (Tdata *)w, (Tdata *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                (int64_t)info.input_dim(1), (int64_t)info.out_channels(), ksize,
                                                                                stride, pad,
                                                                                dilation, 1, nullptr,
                                                                                nullptr, nullptr, true, bias_F32,
                                                                                nullptr, xdnn::Activation_t::LINEAR, nullptr,
                                                                                nullptr, -1)));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    }
    case 3: {
        std::vector<int64_t> ksize = {(int64_t)info.kernel_dim(0), (int64_t)info.kernel_dim(1), (int64_t)info.kernel_dim(2)};
        std::vector<int64_t> stride = {(int64_t)info.stride_info(0), (int64_t)info.stride_info(1), (int64_t)info.stride_info(2)};
        std::vector<int64_t> pad = {(int64_t)info.pad_info(0), (int64_t)info.pad_info(1), (int64_t)info.pad_info(2)};
        std::vector<int64_t> dilation = {(int64_t)info.dilation_info(0), (int64_t)info.dilation_info(1), (int64_t)info.dilation_info(2)};

        CHECK_STATUS(internal->useXdnn(
            (kunlunStream_t)stream,
            [&](xdnnHandle_t handle) {
                CHECK_KUNLUN((xdnn::conv3d_fusion<Tdata, Tdata, Tdata, int16_t>(handle, (Tdata *)x, (Tdata *)w, (Tdata *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                (int64_t)info.input_dim(1), (int64_t)info.input_dim(2), (int64_t)info.out_channels(), ksize,
                                                                                stride, pad,
                                                                                dilation, 1, nullptr,
                                                                                nullptr, nullptr, true, bias_F32,
                                                                                nullptr, xdnn::Activation_t::LINEAR,
                                                                                nullptr)));
                return INFINI_STATUS_SUCCESS;
            }));
        return INFINI_STATUS_SUCCESS;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *bias,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(conv_kernel<float16>(
            _opaque->internal,
            _info,
            workspace,
            workspace_size,
            y,
            x,
            w,
            bias,
            stream));
    } else if (_dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(conv_kernel<float>(
            _opaque->internal,
            _info,
            workspace,
            workspace_size,
            y,
            x,
            w,
            bias,
            stream));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::conv::kunlun
