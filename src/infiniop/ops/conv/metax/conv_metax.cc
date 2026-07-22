#include "conv_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "mcdnn/mcdnn.h" // Metax DNN Library Header
#include <stdexcept>
#include <vector>

namespace op::conv::metax {

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/**
 * @brief Maps InfiniCore data types to mcDNN data types.
 * @param dtype The InfiniCore data type.
 * @return The corresponding mcDNN data type.
 */
inline mcdnnDataType_t toMcDNNDataType(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F32:
        return MCDNN_DATA_FLOAT;
    case INFINI_DTYPE_F16:
        return MCDNN_DATA_HALF;
    case INFINI_DTYPE_BF16:
        return MCDNN_DATA_BFLOAT16;
    default:
        return MCDNN_DATA_FLOAT;
    }
}

/**
 * @brief Checks mcDNN API return status and converts to InfiniCore status.
 */
inline infiniStatus_t checkMcDNNStatus(mcdnnStatus_t status) {
    if (status == MCDNN_STATUS_SUCCESS) {
        return INFINI_STATUS_SUCCESS;
    }
    // Log the error if a logging mechanism is available
    return INFINI_STATUS_INTERNAL_ERROR;
}

struct Descriptor::Opaque {
    mcdnnHandle_t mcdnn_handle = nullptr;
    mcdnnConvolutionDescriptor_t conv_desc = nullptr;
    mcdnnFilterDescriptor_t filter_desc = nullptr;
    mcdnnTensorDescriptor_t input_desc = nullptr;
    mcdnnTensorDescriptor_t output_desc = nullptr;
    mcdnnTensorDescriptor_t bias_desc = nullptr;
    mcdnnActivationDescriptor_t activation_desc = nullptr;
    mcdnnConvolutionFwdAlgo_t algo = MCDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    bool direct_conv2d_kernel = false;
    bool patch_embed_kernel = false;

    ~Opaque() {
        if (mcdnn_handle) {
            mcdnnDestroy(mcdnn_handle);
        }
        if (conv_desc) {
            mcdnnDestroyConvolutionDescriptor(conv_desc);
        }
        if (filter_desc) {
            mcdnnDestroyFilterDescriptor(filter_desc);
        }
        if (input_desc) {
            mcdnnDestroyTensorDescriptor(input_desc);
        }
        if (output_desc) {
            mcdnnDestroyTensorDescriptor(output_desc);
        }
        if (bias_desc) {
            mcdnnDestroyTensorDescriptor(bias_desc);
        }
        if (activation_desc) {
            mcdnnDestroyActivationDescriptor(activation_desc);
        }
    }
};

Descriptor::~Descriptor() = default;

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

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
                                   pads, strides, dilations, n);
    CHECK_RESULT(result);
    ConvInfo info = result.take();

    size_t workspace_size = 0;
    auto opaque = new Opaque();
    mcdnnStatus_t status;

    const bool can_use_direct_conv2d_kernel = info.ndim() == 2;

    if (can_use_direct_conv2d_kernel) {
        opaque->direct_conv2d_kernel = true;
        *desc_ptr = new Descriptor(
            dtype, std::move(info), workspace_size,
            opaque, handle->device, handle->device_id);
        return INFINI_STATUS_SUCCESS;
    }

    const bool can_use_patch_embed_kernel = info.ndim() == 3 && info.output_dim(0) == 1 && info.output_dim(1) == 1 && info.output_dim(2) == 1 && info.input_dim(0) == info.kernel_dim(0) && info.input_dim(1) == info.kernel_dim(1) && info.input_dim(2) == info.kernel_dim(2) && info.pad_info(0) == 0 && info.pad_info(1) == 0 && info.pad_info(2) == 0 && info.stride_info(0) == static_cast<ptrdiff_t>(info.kernel_dim(0)) && info.stride_info(1) == static_cast<ptrdiff_t>(info.kernel_dim(1)) && info.stride_info(2) == static_cast<ptrdiff_t>(info.kernel_dim(2)) && info.dilation_info(0) == 1 && info.dilation_info(1) == 1 && info.dilation_info(2) == 1;

    if (can_use_patch_embed_kernel) {
        opaque->patch_embed_kernel = true;
        *desc_ptr = new Descriptor(
            dtype, std::move(info), workspace_size,
            opaque, handle->device, handle->device_id);
        return INFINI_STATUS_SUCCESS;
    }

    do {
        // 1. Create mcDNN Handle
        status = mcdnnCreate(&opaque->mcdnn_handle);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        // 2. Create Convolution Descriptor
        std::vector<int> pad_vec(info.ndim());
        std::vector<int> stride_vec(info.ndim());
        std::vector<int> dilation_vec(info.ndim());
        for (size_t i = 0; i < info.ndim(); ++i) {
            pad_vec[i] = static_cast<int>(info.pad_info(i));
            stride_vec[i] = static_cast<int>(info.stride_info(i));
            dilation_vec[i] = static_cast<int>(info.dilation_info(i));
        }

        status = mcdnnCreateConvolutionDescriptor(&opaque->conv_desc);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        status = mcdnnSetConvolutionNdDescriptor(
            opaque->conv_desc, info.ndim(), pad_vec.data(), stride_vec.data(),
            dilation_vec.data(), MCDNN_CONVOLUTION, toMcDNNDataType(dtype));
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        // 3. Create Filter Descriptor
        std::vector<int> filter_dim(info.ndim() + 2);
        filter_dim[0] = info.out_channels();
        filter_dim[1] = info.in_channels(); // groups = 1
        for (size_t i = 0; i < info.ndim(); ++i) {
            filter_dim[i + 2] = info.kernel_dim(i);
        }

        status = mcdnnCreateFilterDescriptor(&opaque->filter_desc);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        status = mcdnnSetFilterNdDescriptor(
            opaque->filter_desc, toMcDNNDataType(dtype), MCDNN_TENSOR_NCHW,
            info.ndim() + 2, filter_dim.data());
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        // 4. Create Tensor Descriptors
        std::vector<int> input_dim(info.ndim() + 2);
        input_dim[0] = info.batch();
        input_dim[1] = info.in_channels();
        for (size_t i = 0; i < info.ndim(); ++i) {
            input_dim[i + 2] = info.input_dim(i);
        }

        status = mcdnnCreateTensorDescriptor(&opaque->input_desc);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        status = mcdnnSetTensorNdDescriptor(opaque->input_desc, toMcDNNDataType(dtype),
                                            info.ndim() + 2, input_dim.data(), nullptr);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        std::vector<int> output_dim(info.ndim() + 2);
        output_dim[0] = info.batch();
        output_dim[1] = info.out_channels();
        for (size_t i = 0; i < info.ndim(); ++i) {
            output_dim[i + 2] = info.output_dim(i);
        }

        status = mcdnnCreateTensorDescriptor(&opaque->output_desc);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        status = mcdnnSetTensorNdDescriptor(opaque->output_desc, toMcDNNDataType(dtype),
                                            info.ndim() + 2, output_dim.data(), nullptr);
        if (status != MCDNN_STATUS_SUCCESS) {
            break;
        }

        if (b_desc != nullptr) {
            std::vector<int> bias_dim(info.ndim() + 2, 1);
            bias_dim[1] = info.out_channels(); // 1xCx1x1

            status = mcdnnCreateTensorDescriptor(&opaque->bias_desc);
            if (status != MCDNN_STATUS_SUCCESS) {
                break;
            }

            status = mcdnnSetTensorNdDescriptor(
                opaque->bias_desc,
                toMcDNNDataType(dtype),
                info.ndim() + 2,
                bias_dim.data(),
                nullptr);
            if (status != MCDNN_STATUS_SUCCESS) {
                break;
            }
        }

        // 5. Get Workspace Size
        status = mcdnnGetConvolutionForwardWorkspaceSize(
            opaque->mcdnn_handle,
            opaque->input_desc, opaque->filter_desc, opaque->conv_desc,
            opaque->output_desc, opaque->algo, &workspace_size);

        if (status != MCDNN_STATUS_SUCCESS) {
            workspace_size = 0;
        }

        // Success: Create Descriptor
        *desc_ptr = new Descriptor(
            dtype, std::move(info), workspace_size,
            opaque, handle->device, handle->device_id);

        return INFINI_STATUS_SUCCESS;

    } while (0);

    // Error Handling: Cleanup
    if (opaque) {
        // auto clean mcdnn_handle and descriptors
        delete opaque;
    }
    return INFINI_STATUS_INTERNAL_ERROR;
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

    if (_opaque->direct_conv2d_kernel) {
        return launchDirectConv2d(_dtype, _info, y, x, w, bias, stream);
    }

    if (_opaque->patch_embed_kernel) {
        return launchPatchEmbedConv3d(_dtype, _info, y, x, w, bias, stream);
    }

    mcdnnStatus_t status;
    float alpha = 1.0f;
    float beta = 0.0f;

    // ---------------------------------------------------------------
    // Step 1:  y = conv(x, w)
    // ---------------------------------------------------------------
    status = mcdnnConvolutionForward(
        _opaque->mcdnn_handle,
        &alpha,
        _opaque->input_desc, x,
        _opaque->filter_desc, w,
        _opaque->conv_desc,
        _opaque->algo,
        workspace, workspace_size,
        &beta,
        _opaque->output_desc, y);

    CHECK_STATUS(checkMcDNNStatus(status));

    // ---------------------------------------------------------------
    // Step 2: y = y + bias
    // ---------------------------------------------------------------
    if (bias != nullptr) {
        // mcdnnAddTensor: y = alpha * bias + beta * y
        // we need do: y = 1.0 * bias + 1.0 * y_old
        float alpha_add = 1.0f;
        float beta_add = 1.0f;

        status = mcdnnAddTensor(
            _opaque->mcdnn_handle,
            &alpha_add,
            _opaque->bias_desc, bias,
            &beta_add,
            _opaque->output_desc, y);

        CHECK_STATUS(checkMcDNNStatus(status));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::conv::metax
