#ifndef __CONV_INFO_H__
#define __CONV_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

#ifdef ENABLE_CUDA_API
#include "../../devices/cuda/cuda_handle.cuh"
#endif

namespace op::conv {
class ConvInfo;
#ifdef ENABLE_CUDA_API
class CudnnConvHandler;
#endif
} // namespace op::conv

namespace op::conv {

class ConvInfo {
    ConvInfo() = default;

public:
    size_t ndim;
    size_t batch;
    size_t in_channels;
    size_t out_channels;
    size_t spatial_sizes;
    std::vector<size_t> input_dims;
    std::vector<size_t> kernel_dims;
    std::vector<size_t> output_dims;
    std::vector<size_t> bias_dims;
    std::vector<size_t> pads_info;
    std::vector<size_t> strides_info;
    std::vector<size_t> dilations_info;
#ifdef ENABLE_CUDA_API
    std::shared_ptr<CudnnConvHandler> handler = nullptr;
#endif
    static utils::Result<ConvInfo> create(
        infiniopHandle_t handle_,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n);
};

#ifdef ENABLE_CUDA_API
class CudnnConvHandler {
public:
    CudnnConvHandler() = default;
    ~CudnnConvHandler() {
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
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t b_desc;
    cudnnActivationDescriptor_t act_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    std::shared_ptr<device::cuda::Handle::Internal> internal;
    size_t workspace_size = 0;

    infiniStatus_t GenHandler(
        ConvInfo &info,
        infiniDtype_t data_type,
        cudnnDataType_t compute_type);
};
#endif

inline utils::Result<ConvInfo> ConvInfo::create(
    infiniopHandle_t handle_,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
    auto dtype = y_desc->dtype();
    if (dtype != x_desc->dtype() || dtype != w_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    ConvInfo info;

    info.ndim = n;
    size_t new_dims = n + 2;

    if (x_desc->ndim() < new_dims || y_desc->ndim() < new_dims || w_desc->ndim() < new_dims) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    info.batch = x_desc->shape()[0];
    info.in_channels = x_desc->shape()[1];
    info.out_channels = w_desc->shape()[0];

    if (y_desc->shape()[0] != info.batch || y_desc->shape()[1] != info.out_channels || w_desc->shape()[1] != info.in_channels) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    info.input_dims.resize(info.ndim);
    info.kernel_dims.resize(info.ndim);
    info.output_dims.resize(info.ndim);
    info.pads_info.resize(info.ndim);
    info.strides_info.resize(info.ndim);
    info.dilations_info.resize(info.ndim);

    const size_t *pads_ptr = reinterpret_cast<const size_t *>(pads);           // Renamed to avoid conflict
    const size_t *strides_ptr = reinterpret_cast<const size_t *>(strides);     // Renamed
    const size_t *dilations_ptr = reinterpret_cast<const size_t *>(dilations); // Renamed

    info.spatial_sizes = 1;
    for (size_t i = 0; i < info.ndim; i++) {
        info.input_dims[i] = x_desc->shape()[i + 2];
        info.kernel_dims[i] = w_desc->shape()[i + 2];
        info.output_dims[i] = y_desc->shape()[i + 2];
        info.pads_info[i] = pads_ptr == nullptr ? 0 : pads_ptr[i];
        info.strides_info[i] = strides_ptr == nullptr ? 1 : strides_ptr[i];
        info.dilations_info[i] = dilations_ptr == nullptr ? 1 : dilations_ptr[i];
        info.spatial_sizes = info.spatial_sizes * info.output_dims[i];
        size_t expected_output = (info.input_dims[i] + info.pads_info[i] * 2 - info.dilations_info[i] * (info.kernel_dims[i] - 1) - 1) / info.strides_info[i] + 1;
        if (info.output_dims[i] != expected_output) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (b_desc != nullptr) {
        info.bias_dims.resize(x_desc->ndim());
        std::fill(info.bias_dims.begin(), info.bias_dims.end(), 1);
        info.bias_dims[1] = b_desc->shape()[0];
    }

#ifdef ENABLE_CUDA_API
    if (handle_->device == INFINI_DEVICE_NVIDIA) {
        info.handler = std::make_shared<CudnnConvHandler>();
        auto handle = reinterpret_cast<device::cuda::Handle *>(handle_);
        info.handler->internal = handle->internal();
        CHECK_STATUS(info.handler->GenHandler(info, dtype, CUDNN_DATA_FLOAT));
    }
#endif
    return utils::Result<ConvInfo>(info);
}

#ifdef ENABLE_CUDA_API
inline infiniStatus_t CudnnConvHandler::GenHandler(
    ConvInfo &info,
    infiniDtype_t data_type,
    cudnnDataType_t compute_type) {

    bool is_1d_conv = (info.ndim == 1);

    int actual_tensor_ndim = is_1d_conv ? 4 : static_cast<int>(info.ndim + 2);
    int spatial_ndim_for_conv_desc = static_cast<int>(info.ndim);

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

    input_dims_arr[0] = static_cast<int>(info.batch);
    input_dims_arr[1] = static_cast<int>(info.in_channels);
    output_dims_arr[0] = static_cast<int>(info.batch);
    output_dims_arr[1] = static_cast<int>(info.out_channels);
    filter_dims_arr[0] = static_cast<int>(info.out_channels);
    filter_dims_arr[1] = static_cast<int>(info.in_channels);

    if (is_1d_conv) {
        input_dims_arr[2] = 1;
        input_dims_arr[3] = static_cast<int>(info.input_dims[0]);
        output_dims_arr[2] = 1;
        output_dims_arr[3] = static_cast<int>(info.output_dims[0]);
        filter_dims_arr[2] = 1;
        filter_dims_arr[3] = static_cast<int>(info.kernel_dims[0]);

        pads_arr[0] = 0;
        pads_arr[1] = static_cast<int>(info.pads_info[0]);
        strides_arr[0] = 1;
        strides_arr[1] = static_cast<int>(info.strides_info[0]);
        dilations_arr[0] = 1;
        dilations_arr[1] = static_cast<int>(info.dilations_info[0]);
    } else {
        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims_arr[i + 2] = static_cast<int>(info.input_dims[i]);
            output_dims_arr[i + 2] = static_cast<int>(info.output_dims[i]);
            filter_dims_arr[i + 2] = static_cast<int>(info.kernel_dims[i]);

            pads_arr[i] = static_cast<int>(info.pads_info[i]);
            strides_arr[i] = static_cast<int>(info.strides_info[i]);
            dilations_arr[i] = static_cast<int>(info.dilations_info[i]);
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
    CudnnConvHandler &handler_ref = *info.handler;
    infiniStatus_t status;

    auto create_and_set_descriptors = [&]() {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&handler_ref.x_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&handler_ref.y_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&handler_ref.w_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&handler_ref.conv_desc));

        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            handler_ref.x_desc, CUDNN_TENSOR_NCHW, cudnn_data_type, actual_tensor_ndim, input_dims_arr.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptorEx(
            handler_ref.y_desc, CUDNN_TENSOR_NCHW, cudnn_data_type, actual_tensor_ndim, output_dims_arr.data()));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            handler_ref.w_desc, cudnn_data_type, CUDNN_TENSOR_NCHW, actual_tensor_ndim, filter_dims_arr.data()));
        if (info.bias_dims.empty()) {
            handler_ref.b_desc = nullptr;
            handler_ref.act_desc = nullptr;
        } else {
            std::vector<int> bias_dims_arr(actual_tensor_ndim);
            bias_dims_arr[0] = 1;
            bias_dims_arr[1] = static_cast<int>(info.out_channels);
            for (int i = 2; i < actual_tensor_ndim; ++i) {
                bias_dims_arr[i] = 1;
            }
            std::vector<int> bias_strides_arr(actual_tensor_ndim);
            if (actual_tensor_ndim == 4) {
                bias_strides_arr[0] = static_cast<int>(info.out_channels);
                bias_strides_arr[1] = 1;
                bias_strides_arr[2] = 1;
                bias_strides_arr[3] = 1;
            } else {
                bias_strides_arr[actual_tensor_ndim - 1] = 1;
                for (int d = actual_tensor_ndim - 2; d >= 0; --d) {
                    bias_strides_arr[d] = bias_strides_arr[d + 1] * bias_strides_arr[d + 1];
                }
            }
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&handler_ref.b_desc));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(
                handler_ref.b_desc, cudnn_data_type, bias_dims_arr.size(), bias_dims_arr.data(), bias_strides_arr.data()));
            CHECK_CUDNN(cudnnCreateActivationDescriptor(&handler_ref.act_desc));
            CHECK_CUDNN(cudnnSetActivationDescriptor(
                handler_ref.act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        }
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            handler_ref.conv_desc,
            spatial_ndim_for_conv_desc,
            pads_arr.data(),
            strides_arr.data(),
            dilations_arr.data(),
            CUDNN_CROSS_CORRELATION,
            compute_type));
        if (info.bias_dims.empty()) {
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
                        &handler_ref.workspace_size));
                        return INFINI_STATUS_SUCCESS;
                    }
            ));
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
            int chosenAlgoIndex = 0;
            bool chosen = false;
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
                            &handler_ref.workspace_size));
                        return INFINI_STATUS_SUCCESS;
                    }));
                chosenAlgoIndex = i;
                chosen = true;
                break;
            }
            if (!chosen) {
                return INFINI_STATUS_BAD_PARAM;
            }
            algo = perf_results[chosenAlgoIndex].algo;
        }
        return INFINI_STATUS_SUCCESS;
    };

    status = create_and_set_descriptors();
    if (status != INFINI_STATUS_SUCCESS) {
        // 清理已创建的描述符
        if (handler_ref.x_desc) {
            cudnnDestroyTensorDescriptor(handler_ref.x_desc);
            handler_ref.x_desc = nullptr;
        }
        if (handler_ref.y_desc) {
            cudnnDestroyTensorDescriptor(handler_ref.y_desc);
            handler_ref.y_desc = nullptr;
        }
        if (handler_ref.w_desc) {
            cudnnDestroyFilterDescriptor(handler_ref.w_desc);
            handler_ref.w_desc = nullptr;
        }
        if (handler_ref.conv_desc) {
            cudnnDestroyConvolutionDescriptor(handler_ref.conv_desc);
            handler_ref.conv_desc = nullptr;
        }
        if (handler_ref.act_desc) {
            cudnnDestroyActivationDescriptor(handler_ref.act_desc);
            handler_ref.act_desc = nullptr;
        }
        if (handler_ref.b_desc) {
            cudnnDestroyTensorDescriptor(handler_ref.b_desc);
            handler_ref.b_desc = nullptr;
        }
        return INFINI_STATUS_BAD_PARAM;
    }
    return INFINI_STATUS_SUCCESS;
}
#endif

} // namespace op::conv

#endif // __CONV_INFO_H__
