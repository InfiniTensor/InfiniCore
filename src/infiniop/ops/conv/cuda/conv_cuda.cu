#include "../../../devices/cuda/cuda_handle.cuh"
#include "conv_cuda.cuh"

namespace op::conv::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnActivationDescriptor_t act_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;

    Opaque(std::shared_ptr<device::cuda::Handle::Internal> internal_ptr,
           ConvInfo &info,
           infiniDtype_t data_type)
        : internal(internal_ptr) {

        auto status = GenHandler(info, data_type, CUDNN_DATA_FLOAT);
        if (status != INFINI_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to generate CUDNN descriptors");
        }
    }

    ~Opaque() {
        if (x_desc) {
            cudnnDestroyTensorDescriptor(x_desc);
            x_desc = nullptr;
        }
        if (y_desc) {
            cudnnDestroyTensorDescriptor(y_desc);
            y_desc = nullptr;
        }
        if (w_desc) {
            cudnnDestroyFilterDescriptor(w_desc);
            w_desc = nullptr;
        }
        if (b_desc) {
            cudnnDestroyTensorDescriptor(b_desc);
            b_desc = nullptr;
        }
        if (act_desc) {
            cudnnDestroyActivationDescriptor(act_desc);
            act_desc = nullptr;
        }
        if (conv_desc) {
            cudnnDestroyConvolutionDescriptor(conv_desc);
            conv_desc = nullptr;
        }
    }

    infiniStatus_t GenHandler(
        ConvInfo &info,
        infiniDtype_t data_type,
        cudnnDataType_t compute_type) {

        bool is_1d_conv = (info.ndim() == 1);
        int actual_tensor_ndim = is_1d_conv ? 4 : static_cast<int>(info.ndim() + 2);
        int spatial_ndim_for_conv_desc = static_cast<int>(info.ndim());

        if (is_1d_conv) {
            spatial_ndim_for_conv_desc = 2;
        }

        std::vector<int> input_dims_arr(actual_tensor_ndim);
        std::vector<int> output_dims_arr(actual_tensor_ndim);
        std::vector<int> filter_dims_arr(actual_tensor_ndim);
        std::vector<int> input_strides_arr(actual_tensor_ndim);
        std::vector<int> output_strides_arr(actual_tensor_ndim);

        std::vector<int> pads_arr(spatial_ndim_for_conv_desc);
        std::vector<int> strides_arr(spatial_ndim_for_conv_desc);
        std::vector<int> dilations_arr(spatial_ndim_for_conv_desc);

        input_dims_arr[0] = static_cast<int>(info.batch());
        input_dims_arr[1] = static_cast<int>(info.in_channels());
        output_dims_arr[0] = static_cast<int>(info.batch());
        output_dims_arr[1] = static_cast<int>(info.out_channels());
        filter_dims_arr[0] = static_cast<int>(info.out_channels());
        filter_dims_arr[1] = static_cast<int>(info.in_channels());

        if (is_1d_conv) {
            input_dims_arr[2] = 1;
            input_dims_arr[3] = static_cast<int>(info.input_dim(0));
            output_dims_arr[2] = 1;
            output_dims_arr[3] = static_cast<int>(info.output_dim(0));
            filter_dims_arr[2] = 1;
            filter_dims_arr[3] = static_cast<int>(info.kernel_dim(0));

            pads_arr[0] = 0;
            pads_arr[1] = static_cast<int>(info.pad_info(0));
            strides_arr[0] = 1;
            strides_arr[1] = static_cast<int>(info.stride_info(0));
            dilations_arr[0] = 1;
            dilations_arr[1] = static_cast<int>(info.dilation_info(0));
        } else {
            for (size_t i = 0; i < info.ndim(); ++i) {
                input_dims_arr[i + 2] = static_cast<int>(info.input_dim(i));
                output_dims_arr[i + 2] = static_cast<int>(info.output_dim(i));
                filter_dims_arr[i + 2] = static_cast<int>(info.kernel_dim(i));

                pads_arr[i] = static_cast<int>(info.pad_info(i));
                strides_arr[i] = static_cast<int>(info.stride_info(i));
                dilations_arr[i] = static_cast<int>(info.dilation_info(i));
            }
        }

        input_strides_arr[actual_tensor_ndim - 1] = 1;
        output_strides_arr[actual_tensor_ndim - 1] = 1;
        for (int d = actual_tensor_ndim - 2; d >= 0; --d) {
            input_strides_arr[d] = input_strides_arr[d + 1] * input_dims_arr[d + 1];
            output_strides_arr[d] = output_strides_arr[d + 1] * output_dims_arr[d + 1];
        }

        cudnnDataType_t cudnn_data_type;
        if (data_type == INFINI_DTYPE_F16) {
            cudnn_data_type = CUDNN_DATA_HALF;
        } else if (data_type == INFINI_DTYPE_F32) {
            cudnn_data_type = CUDNN_DATA_FLOAT;
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            x_desc, CUDNN_TENSOR_NCHW, cudnn_data_type, actual_tensor_ndim, input_dims_arr.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            y_desc, CUDNN_TENSOR_NCHW, cudnn_data_type, actual_tensor_ndim, output_dims_arr.data()));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            w_desc, cudnn_data_type, CUDNN_TENSOR_NCHW, actual_tensor_ndim, filter_dims_arr.data()));

        if (info.bias_dims_size() == 0) {
            b_desc = nullptr;
            act_desc = nullptr;
        } else {
            std::vector<int> bias_dims_arr(actual_tensor_ndim);
            bias_dims_arr[0] = 1;
            bias_dims_arr[1] = static_cast<int>(info.out_channels());
            for (int i = 2; i < actual_tensor_ndim; ++i) {
                bias_dims_arr[i] = 1;
            }
            std::vector<int> bias_strides_arr(actual_tensor_ndim);
            if (actual_tensor_ndim == 4) {
                bias_strides_arr[0] = static_cast<int>(info.out_channels());
                bias_strides_arr[1] = 1;
                bias_strides_arr[2] = 1;
                bias_strides_arr[3] = 1;
            } else {
                bias_strides_arr[actual_tensor_ndim - 1] = 1;
                for (int d = actual_tensor_ndim - 2; d >= 0; --d) {
                    bias_strides_arr[d] = bias_strides_arr[d + 1] * bias_dims_arr[d + 1];
                }
            }
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&b_desc));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(
                b_desc, cudnn_data_type, bias_dims_arr.size(), bias_dims_arr.data(), bias_strides_arr.data()));
            CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc));
            CHECK_CUDNN(cudnnSetActivationDescriptor(
                act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        }

        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            conv_desc,
            spatial_ndim_for_conv_desc,
            pads_arr.data(),
            strides_arr.data(),
            dilations_arr.data(),
            CUDNN_CROSS_CORRELATION,
            compute_type));

        if (info.bias_dims_size() == 0) {
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            CHECK_STATUS(internal->useCudnn(
                nullptr,
                [&](cudnnHandle_t handle) {
                    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                        handle,
                        x_desc,
                        w_desc,
                        conv_desc,
                        y_desc,
                        algo,
                        &workspace_size));
                    return INFINI_STATUS_SUCCESS;
                }));
        } else {
            int maxAlgoCount = 0;
            CHECK_STATUS(internal->useCudnn(
                nullptr,
                [&](cudnnHandle_t handle) {
                    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgoCount));
                    return INFINI_STATUS_SUCCESS;
                }));
            if (maxAlgoCount <= 0) {
                maxAlgoCount = 8;
            }
            std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(maxAlgoCount);
            int algoCounts = 0;
            CHECK_STATUS(internal->useCudnn(
                nullptr, [&](cudnnHandle_t handle) {
                    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
                        handle,
                        x_desc,
                        w_desc,
                        conv_desc,
                        y_desc,
                        maxAlgoCount,
                        &algoCounts,
                        perf_results.data()));
                    return INFINI_STATUS_SUCCESS;
                }));
            if (algoCounts < 1) {
                return INFINI_STATUS_BAD_PARAM;
            }
            for (int i = 0; i < algoCounts; ++i) {
                CHECK_STATUS(internal->useCudnn(
                    nullptr,
                    [&](cudnnHandle_t handle) {
                        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                            handle,
                            x_desc,
                            w_desc,
                            conv_desc,
                            y_desc,
                            perf_results[i].algo,
                            &workspace_size));
                        return INFINI_STATUS_SUCCESS;
                    }));
                algo = perf_results[i].algo;
                break;
            }
        }
        return INFINI_STATUS_SUCCESS;
    }
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
    auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc, b_desc,
                                   pads, strides, dilations, n);

    CHECK_RESULT(result);
    auto conv_info = result.take();
    auto opaque = new Opaque(handle->internal(), conv_info, dtype);

    *desc_ptr = new Descriptor(
        dtype,
        std::move(conv_info),
        opaque->workspace_size,
        opaque,
        handle->device,
        handle->device_id);
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
    const float alpha = 1.0f, beta = 0.0f;
    if (bias != nullptr) {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
                    handle,
                    &alpha,
                    _opaque->x_desc,
                    x,
                    _opaque->w_desc,
                    w,
                    _opaque->conv_desc,
                    _opaque->algo,
                    workspace, workspace_size,
                    &beta,
                    _opaque->y_desc,
                    y,
                    _opaque->b_desc,
                    bias,
                    _opaque->act_desc,
                    _opaque->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    } else {
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionForward(
                    handle,
                    &alpha,
                    _opaque->x_desc,
                    x,
                    _opaque->w_desc,
                    w,
                    _opaque->conv_desc,
                    _opaque->algo,
                    workspace, workspace_size,
                    &beta,
                    _opaque->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }));
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::conv::cuda
