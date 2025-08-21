#include "conv_kunlun.h"
#include "../../../../utils.h"
#include "../../../devices/kunlun/kunlun_common.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include <xpu/refactor/context/xpu_act_type.h>
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

infiniStatus_t conv_kernel(
    std::shared_ptr<device::kunlun::Handle::Internal> internal,
    const ConvInfo &info,
    infiniDtype_t dtype,
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
    switch (info.ndim()) {
    case 1: {
        int64_t ksize = (int64_t)info.kernel_dim(0);
        int64_t stride = (int64_t)info.stride_info(0);
        std::initializer_list<int64_t> pad = {(int64_t)info.pad_info(0)};
        int64_t dilation = (int64_t)info.dilation_info(0);

        if (dtype == INFINI_DTYPE_F16) {

            if (bias_size > 0) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::cast<float16, float>(handle, (float16 *)bias, bias_F32, bias_size)));
                        CHECK_KUNLUN((xdnn::conv1d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, bias_F32,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                              nullptr)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            } else {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::conv1d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, nullptr,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                              nullptr)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            }
            return INFINI_STATUS_SUCCESS;

        } else if (dtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(internal->useXdnn(
                (kunlunStream_t)stream,
                [&](xdnnHandle_t handle) {
                    CHECK_KUNLUN((xdnn::conv1d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                    (int64_t)info.out_channels(), ksize,
                                                                                    stride, pad,
                                                                                    dilation, 1, nullptr,
                                                                                    nullptr, nullptr, true, (float *)bias,
                                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                    nullptr)));
                    return INFINI_STATUS_SUCCESS;
                }));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;
    }
    case 2: {
        std::vector<int64_t> ksize = {(int64_t)info.kernel_dim(0), (int64_t)info.kernel_dim(1)};
        std::vector<int64_t> stride = {(int64_t)info.stride_info(0), (int64_t)info.stride_info(1)};
        std::vector<int64_t> pad = {
            (int64_t)info.pad_info(0),
            (int64_t)info.pad_info(0),
            (int64_t)info.pad_info(1),
            (int64_t)info.pad_info(1)};
        std::vector<int64_t> dilation = {(int64_t)info.dilation_info(0), (int64_t)info.dilation_info(1)};

        if (dtype == INFINI_DTYPE_F16) {
            if (bias_size > 0) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::cast<float16, float>(handle, (float16 *)bias, bias_F32, bias_size)));
                        CHECK_KUNLUN((xdnn::conv2d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.input_dim(1), (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, bias_F32,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR, nullptr,
                                                                                              nullptr, -1)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            } else {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::conv2d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.input_dim(1), (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, nullptr,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR, nullptr,
                                                                                              nullptr, -1)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            }
            return INFINI_STATUS_SUCCESS;

        } else if (dtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(internal->useXdnn(
                (kunlunStream_t)stream,
                [&](xdnnHandle_t handle) {
                    CHECK_KUNLUN((xdnn::conv2d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                    (int64_t)info.input_dim(1), (int64_t)info.out_channels(), ksize,
                                                                                    stride, pad,
                                                                                    dilation, 1, nullptr,
                                                                                    nullptr, nullptr, true, (float *)bias,
                                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR, nullptr,
                                                                                    nullptr, -1)));
                    return INFINI_STATUS_SUCCESS;
                }));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;
    }
    case 3: {
        std::vector<int64_t> ksize = {(int64_t)info.kernel_dim(0), (int64_t)info.kernel_dim(1), (int64_t)info.kernel_dim(2)};
        std::vector<int64_t> stride = {(int64_t)info.stride_info(0), (int64_t)info.stride_info(1), (int64_t)info.stride_info(2)};
        std::vector<int64_t> pad = {(int64_t)info.pad_info(0), (int64_t)info.pad_info(1), (int64_t)info.pad_info(2)};
        std::vector<int64_t> dilation = {(int64_t)info.dilation_info(0), (int64_t)info.dilation_info(1), (int64_t)info.dilation_info(2)};

        if (dtype == INFINI_DTYPE_F16) {
            if (bias_size > 0) {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::cast<float16, float>(handle, (float16 *)bias, bias_F32, bias_size)));
                        CHECK_KUNLUN((xdnn::conv3d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.input_dim(1), (int64_t)info.input_dim(2), (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, bias_F32,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                              nullptr)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            } else {
                CHECK_STATUS(internal->useXdnn(
                    (kunlunStream_t)stream,
                    [&](xdnnHandle_t handle) {
                        CHECK_KUNLUN((xdnn::conv3d_fusion<float16, float16, float16, int16_t>(handle, (float16 *)x, (float16 *)w, (float16 *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                              (int64_t)info.input_dim(1), (int64_t)info.input_dim(2), (int64_t)info.out_channels(), ksize,
                                                                                              stride, pad,
                                                                                              dilation, 1, nullptr,
                                                                                              nullptr, nullptr, true, nullptr,
                                                                                              nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                              nullptr)));
                        return INFINI_STATUS_SUCCESS;
                    }));
            }
            return INFINI_STATUS_SUCCESS;
        } else if (dtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(internal->useXdnn(
                (kunlunStream_t)stream,
                [&](xdnnHandle_t handle) {
                    CHECK_KUNLUN((xdnn::conv3d_fusion<float, float, float, int16_t>(handle, (float *)x, (float *)w, (float *)y, (int64_t)info.batch(), (int64_t)info.in_channels(), (int64_t)info.input_dim(0),
                                                                                    (int64_t)info.input_dim(1), (int64_t)info.input_dim(2), (int64_t)info.out_channels(), ksize,
                                                                                    stride, pad,
                                                                                    dilation, 1, nullptr,
                                                                                    nullptr, nullptr, true, (float *)bias,
                                                                                    nullptr, baidu::xpu::api::Activation_t::LINEAR,
                                                                                    nullptr)));
                    return INFINI_STATUS_SUCCESS;
                }));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;
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
    CHECK_STATUS(conv_kernel(
        _opaque->internal,
        _info,
        _dtype,
        workspace,
        workspace_size,
        y,
        x,
        w,
        bias,
        stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::conv::kunlun
