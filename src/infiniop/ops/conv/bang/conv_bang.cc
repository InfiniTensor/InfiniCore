#include "conv_bang.h"
#include "../../../devices/bang/common_bang.h"

#include <algorithm>
#include <array>
#include <vector>

namespace op::conv::bang {

static size_t alignSize(size_t value) {
    return (value + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
}

static size_t tensorElements(const std::vector<int> &dims) {
    size_t n = 1;
    for (int d : dims) {
        n *= static_cast<size_t>(d);
    }
    return n;
}

static infiniStatus_t setTensor(cnnlTensorDescriptor_t desc,
                                cnnlTensorLayout_t layout,
                                infiniDtype_t dtype,
                                const std::vector<int> &dims) {
    CHECK_BANG(cnnlSetTensorDescriptor(
        desc,
        layout,
        device::bang::getCnnlDtype(dtype),
        static_cast<int>(dims.size()),
        const_cast<int *>(dims.data())));
    return INFINI_STATUS_SUCCESS;
}

struct ConvShapes {
    std::vector<int> x_orig;
    std::vector<int> x_cnnl;
    std::vector<int> w_orig;
    std::vector<int> w_cnnl;
    std::vector<int> y_orig;
    std::vector<int> y_cnnl;
    std::vector<int> pad;
    std::vector<int> stride;
    std::vector<int> dilation;
    std::vector<int> perm_to_cnnl;
    std::vector<int> perm_to_orig;
    cnnlTensorLayout_t orig_layout;
    cnnlTensorLayout_t cnnl_layout;
    int conv_dim;
};

static ConvShapes makeShapes(const ConvInfo &info) {
    ConvShapes s;
    if (info.ndim() == 1) {
        s.x_orig = {static_cast<int>(info.batch()), static_cast<int>(info.in_channels()), 1, static_cast<int>(info.input_dim(0))};
        s.x_cnnl = {static_cast<int>(info.batch()), 1, static_cast<int>(info.input_dim(0)), static_cast<int>(info.in_channels())};
        s.w_orig = {static_cast<int>(info.out_channels()), static_cast<int>(info.in_channels()), 1, static_cast<int>(info.kernel_dim(0))};
        s.w_cnnl = {static_cast<int>(info.out_channels()), 1, static_cast<int>(info.kernel_dim(0)), static_cast<int>(info.in_channels())};
        s.y_orig = {static_cast<int>(info.batch()), static_cast<int>(info.out_channels()), 1, static_cast<int>(info.output_dim(0))};
        s.y_cnnl = {static_cast<int>(info.batch()), 1, static_cast<int>(info.output_dim(0)), static_cast<int>(info.out_channels())};
        s.pad = {0, 0, static_cast<int>(info.pad_info(0)), static_cast<int>(info.pad_info(0))};
        s.stride = {1, static_cast<int>(info.stride_info(0))};
        s.dilation = {1, static_cast<int>(info.dilation_info(0))};
        s.perm_to_cnnl = {0, 2, 3, 1};
        s.perm_to_orig = {0, 3, 1, 2};
        s.orig_layout = CNNL_LAYOUT_NCHW;
        s.cnnl_layout = CNNL_LAYOUT_NHWC;
        s.conv_dim = 4;
    } else if (info.ndim() == 2) {
        s.x_orig = {static_cast<int>(info.batch()), static_cast<int>(info.in_channels()), static_cast<int>(info.input_dim(0)), static_cast<int>(info.input_dim(1))};
        s.x_cnnl = {static_cast<int>(info.batch()), static_cast<int>(info.input_dim(0)), static_cast<int>(info.input_dim(1)), static_cast<int>(info.in_channels())};
        s.w_orig = {static_cast<int>(info.out_channels()), static_cast<int>(info.in_channels()), static_cast<int>(info.kernel_dim(0)), static_cast<int>(info.kernel_dim(1))};
        s.w_cnnl = {static_cast<int>(info.out_channels()), static_cast<int>(info.kernel_dim(0)), static_cast<int>(info.kernel_dim(1)), static_cast<int>(info.in_channels())};
        s.y_orig = {static_cast<int>(info.batch()), static_cast<int>(info.out_channels()), static_cast<int>(info.output_dim(0)), static_cast<int>(info.output_dim(1))};
        s.y_cnnl = {static_cast<int>(info.batch()), static_cast<int>(info.output_dim(0)), static_cast<int>(info.output_dim(1)), static_cast<int>(info.out_channels())};
        s.pad = {static_cast<int>(info.pad_info(0)), static_cast<int>(info.pad_info(0)), static_cast<int>(info.pad_info(1)), static_cast<int>(info.pad_info(1))};
        s.stride = {static_cast<int>(info.stride_info(0)), static_cast<int>(info.stride_info(1))};
        s.dilation = {static_cast<int>(info.dilation_info(0)), static_cast<int>(info.dilation_info(1))};
        s.perm_to_cnnl = {0, 2, 3, 1};
        s.perm_to_orig = {0, 3, 1, 2};
        s.orig_layout = CNNL_LAYOUT_NCHW;
        s.cnnl_layout = CNNL_LAYOUT_NHWC;
        s.conv_dim = 4;
    } else {
        s.x_orig = {static_cast<int>(info.batch()), static_cast<int>(info.in_channels()), static_cast<int>(info.input_dim(0)), static_cast<int>(info.input_dim(1)), static_cast<int>(info.input_dim(2))};
        s.x_cnnl = {static_cast<int>(info.batch()), static_cast<int>(info.input_dim(0)), static_cast<int>(info.input_dim(1)), static_cast<int>(info.input_dim(2)), static_cast<int>(info.in_channels())};
        s.w_orig = {static_cast<int>(info.out_channels()), static_cast<int>(info.in_channels()), static_cast<int>(info.kernel_dim(0)), static_cast<int>(info.kernel_dim(1)), static_cast<int>(info.kernel_dim(2))};
        s.w_cnnl = {static_cast<int>(info.out_channels()), static_cast<int>(info.kernel_dim(0)), static_cast<int>(info.kernel_dim(1)), static_cast<int>(info.kernel_dim(2)), static_cast<int>(info.in_channels())};
        s.y_orig = {static_cast<int>(info.batch()), static_cast<int>(info.out_channels()), static_cast<int>(info.output_dim(0)), static_cast<int>(info.output_dim(1)), static_cast<int>(info.output_dim(2))};
        s.y_cnnl = {static_cast<int>(info.batch()), static_cast<int>(info.output_dim(0)), static_cast<int>(info.output_dim(1)), static_cast<int>(info.output_dim(2)), static_cast<int>(info.out_channels())};
        s.pad = {static_cast<int>(info.pad_info(0)), static_cast<int>(info.pad_info(0)), static_cast<int>(info.pad_info(1)), static_cast<int>(info.pad_info(1)), static_cast<int>(info.pad_info(2)), static_cast<int>(info.pad_info(2))};
        s.stride = {static_cast<int>(info.stride_info(0)), static_cast<int>(info.stride_info(1)), static_cast<int>(info.stride_info(2))};
        s.dilation = {static_cast<int>(info.dilation_info(0)), static_cast<int>(info.dilation_info(1)), static_cast<int>(info.dilation_info(2))};
        s.perm_to_cnnl = {0, 2, 3, 4, 1};
        s.perm_to_orig = {0, 4, 1, 2, 3};
        s.orig_layout = CNNL_LAYOUT_NCDHW;
        s.cnnl_layout = CNNL_LAYOUT_NDHWC;
        s.conv_dim = 5;
    }
    return s;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::bang::Handle::Internal> internal;
    cnnlTensorDescriptor_t x_orig_desc = nullptr;
    cnnlTensorDescriptor_t x_cnnl_desc = nullptr;
    cnnlTensorDescriptor_t w_orig_desc = nullptr;
    cnnlTensorDescriptor_t w_cnnl_desc = nullptr;
    cnnlTensorDescriptor_t y_orig_desc = nullptr;
    cnnlTensorDescriptor_t y_cnnl_desc = nullptr;
    cnnlTensorDescriptor_t bias_desc = nullptr;
    cnnlConvolutionDescriptor_t conv_desc = nullptr;
    cnnlQuantizeExDescriptor_t y_quant_desc = nullptr;
    cnnlTransposeDescriptor_t x_trans_desc = nullptr;
    cnnlTransposeDescriptor_t w_trans_desc = nullptr;
    cnnlTransposeDescriptor_t y_trans_desc = nullptr;
    cnnlConvolutionForwardAlgo_t algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;
    size_t x_cnnl_bytes = 0;
    size_t w_cnnl_bytes = 0;
    size_t y_cnnl_bytes = 0;
    size_t transpose_workspace_size = 0;
    size_t conv_workspace_size = 0;

    ~Opaque() {
        if (x_orig_desc) {
            cnnlDestroyTensorDescriptor(x_orig_desc);
        }
        if (x_cnnl_desc) {
            cnnlDestroyTensorDescriptor(x_cnnl_desc);
        }
        if (w_orig_desc) {
            cnnlDestroyTensorDescriptor(w_orig_desc);
        }
        if (w_cnnl_desc) {
            cnnlDestroyTensorDescriptor(w_cnnl_desc);
        }
        if (y_orig_desc) {
            cnnlDestroyTensorDescriptor(y_orig_desc);
        }
        if (y_cnnl_desc) {
            cnnlDestroyTensorDescriptor(y_cnnl_desc);
        }
        if (bias_desc) {
            cnnlDestroyTensorDescriptor(bias_desc);
        }
        if (conv_desc) {
            cnnlDestroyConvolutionDescriptor(conv_desc);
        }
        if (y_quant_desc) {
            cnnlDestroyQuantizeExDescriptor(y_quant_desc);
        }
        if (x_trans_desc) {
            cnnlDestroyTransposeDescriptor(x_trans_desc);
        }
        if (w_trans_desc) {
            cnnlDestroyTransposeDescriptor(w_trans_desc);
        }
        if (y_trans_desc) {
            cnnlDestroyTransposeDescriptor(y_trans_desc);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

static infiniStatus_t createTransposeDesc(cnnlTransposeDescriptor_t *desc, const std::vector<int> &perm) {
    CHECK_BANG(cnnlCreateTransposeDescriptor(desc));
    CHECK_BANG(cnnlSetTransposeDescriptor(*desc, static_cast<int>(perm.size()), const_cast<int *>(perm.data())));
    return INFINI_STATUS_SUCCESS;
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
    auto handle = reinterpret_cast<device::bang::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n);
    CHECK_RESULT(result);
    auto info = result.take();
    if (info.ndim() < 1 || info.ndim() > 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto shapes = makeShapes(info);
    auto opaque = new Opaque();
    opaque->internal = handle->internal();
    opaque->x_cnnl_bytes = alignSize(tensorElements(shapes.x_cnnl) * infiniSizeOf(dtype));
    opaque->w_cnnl_bytes = alignSize(tensorElements(shapes.w_cnnl) * infiniSizeOf(dtype));
    opaque->y_cnnl_bytes = alignSize(tensorElements(shapes.y_cnnl) * infiniSizeOf(dtype));

    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_orig_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->x_cnnl_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->w_orig_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->w_cnnl_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_orig_desc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->y_cnnl_desc));
    CHECK_STATUS(setTensor(opaque->x_orig_desc, shapes.orig_layout, dtype, shapes.x_orig));
    CHECK_STATUS(setTensor(opaque->x_cnnl_desc, shapes.cnnl_layout, dtype, shapes.x_cnnl));
    CHECK_STATUS(setTensor(opaque->w_orig_desc, shapes.orig_layout, dtype, shapes.w_orig));
    CHECK_STATUS(setTensor(opaque->w_cnnl_desc, shapes.cnnl_layout, dtype, shapes.w_cnnl));
    CHECK_STATUS(setTensor(opaque->y_orig_desc, shapes.orig_layout, dtype, shapes.y_orig));
    CHECK_STATUS(setTensor(opaque->y_cnnl_desc, shapes.cnnl_layout, dtype, shapes.y_cnnl));

    if (b_desc != nullptr) {
        int bias_dims[1] = {static_cast<int>(info.out_channels())};
        CHECK_BANG(cnnlCreateTensorDescriptor(&opaque->bias_desc));
        CHECK_BANG(cnnlSetTensorDescriptor(opaque->bias_desc, CNNL_LAYOUT_ARRAY, device::bang::getCnnlDtype(dtype), 1, bias_dims));
    }

    CHECK_BANG(cnnlCreateConvolutionDescriptor(&opaque->conv_desc));
    cnnlDataType_t compute_dtype = dtype == INFINI_DTYPE_BF16
                                     ? CNNL_DTYPE_FLOAT
                                     : device::bang::getCnnlDtype(dtype);
    CHECK_BANG(cnnlSetConvolutionDescriptor(
        opaque->conv_desc,
        shapes.conv_dim,
        shapes.pad.data(),
        shapes.stride.data(),
        shapes.dilation.data(),
        1,
        compute_dtype));
    if (dtype == INFINI_DTYPE_BF16) {
        CHECK_BANG(cnnlCreateQuantizeExDescriptor(&opaque->y_quant_desc));
        CHECK_BANG(cnnlSetQuantizeExDescriptorQuantSchemeAndDtype(
            opaque->y_quant_desc, CNNL_QUANTIZE_NONE, CNNL_DTYPE_FLOAT));
        CHECK_BANG(cnnlSetConvolutionDescriptorQuant(
            opaque->conv_desc, nullptr, nullptr, opaque->y_quant_desc));
    }
    opaque->algo = CNNL_CONVOLUTION_FWD_ALGO_DIRECT;

    CHECK_STATUS(createTransposeDesc(&opaque->x_trans_desc, shapes.perm_to_cnnl));
    CHECK_STATUS(createTransposeDesc(&opaque->w_trans_desc, shapes.perm_to_cnnl));
    CHECK_STATUS(createTransposeDesc(&opaque->y_trans_desc, shapes.perm_to_orig));

    size_t tx_ws = 0, tw_ws = 0, ty_ws = 0, conv_ws = 0;
    CHECK_STATUS(opaque->internal->useCnnl(
        nullptr,
        [&](cnnlHandle_t cnnl_handle) {
            CHECK_BANG(cnnlGetTransposeWorkspaceSize(cnnl_handle, opaque->x_orig_desc, opaque->x_trans_desc, &tx_ws));
            CHECK_BANG(cnnlGetTransposeWorkspaceSize(cnnl_handle, opaque->w_orig_desc, opaque->w_trans_desc, &tw_ws));
            CHECK_BANG(cnnlGetTransposeWorkspaceSize(cnnl_handle, opaque->y_cnnl_desc, opaque->y_trans_desc, &ty_ws));
            CHECK_BANG(cnnlGetConvolutionForwardWorkspaceSize(
                cnnl_handle,
                opaque->x_cnnl_desc,
                opaque->w_cnnl_desc,
                opaque->y_cnnl_desc,
                opaque->bias_desc,
                opaque->conv_desc,
                opaque->algo,
                &conv_ws));
            return INFINI_STATUS_SUCCESS;
        }));
    opaque->transpose_workspace_size = alignSize(std::max({tx_ws, tw_ws, ty_ws}));
    opaque->conv_workspace_size = alignSize(conv_ws);
    size_t workspace_size = opaque->x_cnnl_bytes + opaque->w_cnnl_bytes + opaque->y_cnnl_bytes + opaque->transpose_workspace_size + opaque->conv_workspace_size;

    *desc_ptr = new Descriptor(dtype, std::move(info), workspace_size, opaque, handle_->device, handle_->device_id);
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

    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    char *buffer = reinterpret_cast<char *>(workspace);
    void *x_cnnl = buffer;
    void *w_cnnl = buffer + _opaque->x_cnnl_bytes;
    void *y_cnnl = buffer + _opaque->x_cnnl_bytes + _opaque->w_cnnl_bytes;
    void *transpose_workspace = buffer + _opaque->x_cnnl_bytes + _opaque->w_cnnl_bytes + _opaque->y_cnnl_bytes;
    void *conv_workspace = buffer + _opaque->x_cnnl_bytes + _opaque->w_cnnl_bytes + _opaque->y_cnnl_bytes + _opaque->transpose_workspace_size;

    CHECK_STATUS(_opaque->internal->useCnnl(
        queue,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlTranspose_v2(
                handle,
                _opaque->x_trans_desc,
                _opaque->x_orig_desc,
                x,
                _opaque->x_cnnl_desc,
                x_cnnl,
                transpose_workspace,
                _opaque->transpose_workspace_size));
            CHECK_BANG(cnnlTranspose_v2(
                handle,
                _opaque->w_trans_desc,
                _opaque->w_orig_desc,
                w,
                _opaque->w_cnnl_desc,
                w_cnnl,
                transpose_workspace,
                _opaque->transpose_workspace_size));
            CHECK_BANG(cnnlConvolutionForward(
                handle,
                _opaque->conv_desc,
                _opaque->algo,
                nullptr,
                _opaque->x_cnnl_desc,
                x_cnnl,
                _opaque->w_cnnl_desc,
                w_cnnl,
                _opaque->bias_desc,
                bias,
                conv_workspace,
                _opaque->conv_workspace_size,
                nullptr,
                _opaque->y_cnnl_desc,
                y_cnnl));
            CHECK_BANG(cnnlTranspose_v2(
                handle,
                _opaque->y_trans_desc,
                _opaque->y_cnnl_desc,
                y_cnnl,
                _opaque->y_orig_desc,
                y,
                transpose_workspace,
                _opaque->transpose_workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(queue);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::conv::bang
