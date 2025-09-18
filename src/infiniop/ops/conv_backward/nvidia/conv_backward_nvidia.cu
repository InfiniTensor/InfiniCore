#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../cuda/bias_grad_kernel.cuh"
#include "../info.h"
#include "conv_backward_nvidia.cuh"

infiniStatus_t launch_bias_grad_kernel(const void *grad_output, void *grad_bias,
                                       const int *grad_output_dims,
                                       size_t conv_ndim,
                                       cudnnDataType_t data_type,
                                       cudaStream_t stream) {
    int batch_size = grad_output_dims[0];
    int channels = grad_output_dims[1];
    int spatial_size = 1;

    for (size_t i = 2; i < conv_ndim + 2; ++i) {
        spatial_size *= grad_output_dims[i];
    }

    dim3 block(256);
    dim3 grid((channels + block.x - 1) / block.x);

    // 直接调用 bf16 kernel
    compute_bias_grad_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(grad_output),
        reinterpret_cast<__nv_bfloat16 *>(grad_bias), batch_size, channels,
        spatial_size);

    return INFINI_STATUS_SUCCESS;
}

#define DESTROY_CUDNN_DESCRIPTOR(desc_ptr, destroy_func) \
    do {                                                 \
        if (desc_ptr) {                                  \
            destroy_func(desc_ptr);                      \
            desc_ptr = nullptr;                          \
        }                                                \
    } while (0)

#define CLEANUP_CUDNN_DESCRIPTORS()                                               \
    do {                                                                          \
        DESTROY_CUDNN_DESCRIPTOR(input_desc, cudnnDestroyTensorDescriptor);       \
        DESTROY_CUDNN_DESCRIPTOR(grad_output_desc, cudnnDestroyTensorDescriptor); \
        DESTROY_CUDNN_DESCRIPTOR(weight_desc, cudnnDestroyFilterDescriptor);      \
        DESTROY_CUDNN_DESCRIPTOR(grad_input_desc, cudnnDestroyTensorDescriptor);  \
        DESTROY_CUDNN_DESCRIPTOR(grad_weight_desc, cudnnDestroyFilterDescriptor); \
        DESTROY_CUDNN_DESCRIPTOR(grad_bias_desc, cudnnDestroyTensorDescriptor);   \
        DESTROY_CUDNN_DESCRIPTOR(conv_desc, cudnnDestroyConvolutionDescriptor);   \
    } while (0)

namespace op::conv_backward::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t workspace_size = 0;

#ifdef ENABLE_CUDNN_API
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t grad_output_desc = nullptr;
    cudnnFilterDescriptor_t weight_desc = nullptr;
    cudnnTensorDescriptor_t grad_input_desc = nullptr;
    cudnnFilterDescriptor_t grad_weight_desc = nullptr;
    cudnnTensorDescriptor_t grad_bias_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;

    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    size_t bwd_data_workspace_size = 0;
    size_t bwd_filter_workspace_size = 0;
    size_t conv_ndim = 0;
#endif

private:
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

#ifdef ENABLE_CUDNN_API
    infiniStatus_t getCudnnDataType(infiniDtype_t data_type,
                                    cudnnDataType_t &cudnn_data_type) const {
        if (data_type == INFINI_DTYPE_F16 || data_type == INFINI_DTYPE_F32 || data_type == INFINI_DTYPE_BF16) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
            return INFINI_STATUS_SUCCESS;
        }
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniStatus_t calculateStrides(int ndim, const int *input_dims,
                                    std::vector<int> &input_strides) const {
        input_strides.resize(ndim);
        input_strides[ndim - 1] = 1; // 最后一维 stride = 1
        for (int i = ndim - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
        }
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t createTensorAndFilterDescriptors(
        const op::conv_backward::ConvBackwardInfo &info,
        cudnnDataType_t cudnn_data_type, infiniopTensorDescriptor_t bias_desc) {

        int ndim = static_cast<int>(info.ndim + 2);

        // input
        std::vector<int> input_dims = {static_cast<int>(info.batch),
                                       static_cast<int>(info.in_channels)};
        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims.push_back(static_cast<int>(info.input_dims[i]));
        }
        std::vector<int> input_strides;
        CHECK_STATUS(calculateStrides(ndim, input_dims.data(), input_strides));

        // grad_output
        std::vector<int> grad_output_dims = {static_cast<int>(info.batch),
                                             static_cast<int>(info.out_channels)};
        for (size_t i = 0; i < info.ndim; ++i) {
            grad_output_dims.push_back(static_cast<int>(info.grad_output_dims[i]));
        }
        std::vector<int> grad_output_strides;
        CHECK_STATUS(
            calculateStrides(ndim, grad_output_dims.data(), grad_output_strides));

        // weight
        size_t in_channels_per_group = info.in_channels / info.groups;
        std::vector<int> weight_dims = {static_cast<int>(info.out_channels),
                                        static_cast<int>(in_channels_per_group)};
        for (size_t i = 0; i < info.ndim; ++i) {
            weight_dims.push_back(static_cast<int>(info.weight_dims[i]));
        }

        if (info.ndim == 1) {
            input_dims.push_back(1);
            input_strides.push_back(1);
            grad_output_dims.push_back(1);
            grad_output_strides.push_back(1);
            weight_dims.push_back(1);
        }

        // input
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, cudnn_data_type,
                                               input_dims.size(), input_dims.data(),
                                               input_strides.data()));

        // grad_output
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_output_desc));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            grad_output_desc, cudnn_data_type, grad_output_dims.size(),
            grad_output_dims.data(), grad_output_strides.data()));

        // weight
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            weight_desc, cudnn_data_type, CUDNN_TENSOR_NCHW, weight_dims.size(),
            weight_dims.data()));

        // grad_input
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_input_desc));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(grad_input_desc, cudnn_data_type,
                                               input_dims.size(), input_dims.data(),
                                               input_strides.data()));

        // grad_weight
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&grad_weight_desc));
        CHECK_CUDNN(cudnnSetFilterNdDescriptor(
            grad_weight_desc, cudnn_data_type, CUDNN_TENSOR_NCHW,
            weight_dims.size(), weight_dims.data()));

        // grad_bias (optional)
        if (bias_desc) {
            int bias_ndim = (info.ndim == 1) ? 4 : ndim;

            std::vector<int> bias_dims(bias_ndim, 1);
            bias_dims[1] = static_cast<int>(bias_desc->dim(0)); // out_channels

            std::vector<int> bias_strides(bias_ndim, 1);
            for (int i = bias_ndim - 2; i >= 0; --i) {
                bias_strides[i] = bias_strides[i + 1] * bias_dims[i + 1];
            }

            CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_bias_desc));
            CHECK_CUDNN(cudnnSetTensorNdDescriptor(grad_bias_desc, cudnn_data_type,
                                                   bias_ndim, bias_dims.data(),
                                                   bias_strides.data()));
        }

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t
    createConvDescriptor(const op::conv_backward::ConvBackwardInfo &info,
                         cudnnDataType_t cudnn_data_type) {
        int conv_dim = (info.ndim == 1) ? 2 : static_cast<int>(info.ndim);
        std::vector<int> pad_vec(info.pads.begin(), info.pads.end());
        std::vector<int> stride_vec(info.strides.begin(), info.strides.end());
        std::vector<int> dilation_vec(info.dilations.begin(), info.dilations.end());

        if (info.ndim == 1) {
            pad_vec.push_back(0);
            stride_vec.push_back(1);
            dilation_vec.push_back(1);
        }

        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
        cudnnDataType_t compute_type = (cudnn_data_type == CUDNN_DATA_BFLOAT16 || cudnn_data_type == CUDNN_DATA_HALF)
                                         ? CUDNN_DATA_FLOAT
                                         : cudnn_data_type;
        CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            conv_desc, conv_dim, pad_vec.data(), stride_vec.data(),
            dilation_vec.data(), CUDNN_CROSS_CORRELATION, compute_type));
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc,
                                                  static_cast<int>(info.groups)));
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t
    initializeCudnnContext(const op::conv_backward::ConvBackwardInfo &info,
                           infiniDtype_t data_type,
                           infiniopTensorDescriptor_t bias_desc) {

        cudnnDataType_t cudnn_data_type;
        CHECK_STATUS(getCudnnDataType(data_type, cudnn_data_type));
        CHECK_STATUS(
            createTensorAndFilterDescriptors(info, cudnn_data_type, bias_desc));
        CHECK_STATUS(createConvDescriptor(info, cudnn_data_type));

        // Query workspace size
        internal->useCudnn(nullptr, [&](cudnnHandle_t h) {
            // 1. 查找适合的反向数据算法
            int requestedAlgoCount = 8;
            int returnedAlgoCount = 0;
            cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf[8];

            cudnnStatus_t status = cudnnFindConvolutionBackwardDataAlgorithm(
                h, weight_desc, grad_output_desc, conv_desc, grad_input_desc,
                requestedAlgoCount, &returnedAlgoCount, bwd_data_perf);
            bool found = false;
            if (status == CUDNN_STATUS_SUCCESS && returnedAlgoCount > 0) {
                for (int i = 0; i < returnedAlgoCount; i++) {
                    if (bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS) {
                        bwd_data_algo = bwd_data_perf[i].algo;
                        bwd_data_workspace_size = bwd_data_perf[i].memory;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // 如果没找到成功的算法，用默认的
                    bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
                    cudnnGetConvolutionBackwardDataWorkspaceSize(
                        h, weight_desc, grad_output_desc, conv_desc, grad_input_desc,
                        bwd_data_algo, &bwd_data_workspace_size);
                }
            } else {
                // 查找失败，回退到默认算法
                bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
                cudnnGetConvolutionBackwardDataWorkspaceSize(
                    h, weight_desc, grad_output_desc, conv_desc, grad_input_desc,
                    bwd_data_algo, &bwd_data_workspace_size);
            }

            // 2. 查找适合的反向权重算法
            cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf[8];

            status = cudnnFindConvolutionBackwardFilterAlgorithm(
                h, input_desc, grad_output_desc, conv_desc, grad_weight_desc,
                requestedAlgoCount, &returnedAlgoCount, bwd_filter_perf);

            if (status == CUDNN_STATUS_SUCCESS && returnedAlgoCount > 0) {
                found = false;
                for (int i = 0; i < returnedAlgoCount; i++) {
                    if (bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS) {
                        bwd_filter_algo = bwd_filter_perf[i].algo;
                        bwd_filter_workspace_size = bwd_filter_perf[i].memory;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    // 如果没找到成功的算法，用默认的
                    bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
                    cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        h, input_desc, grad_output_desc, conv_desc, grad_weight_desc,
                        bwd_filter_algo, &bwd_filter_workspace_size);
                }
            } else {
                bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
                cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    h, input_desc, grad_output_desc, conv_desc, grad_weight_desc,
                    bwd_filter_algo, &bwd_filter_workspace_size);
            }
            return INFINI_STATUS_SUCCESS;
        });
        workspace_size = std::max(bwd_data_workspace_size, bwd_filter_workspace_size);

        conv_ndim = info.ndim;

        return INFINI_STATUS_SUCCESS;
    }
#endif

public:
    Opaque(Opaque &&other) noexcept
        : internal(std::move(other.internal)),
          workspace_size(other.workspace_size)
#ifdef ENABLE_CUDNN_API
          ,
          input_desc(other.input_desc), grad_output_desc(other.grad_output_desc),
          weight_desc(other.weight_desc), grad_input_desc(other.grad_input_desc),
          grad_weight_desc(other.grad_weight_desc),
          grad_bias_desc(other.grad_bias_desc), conv_desc(other.conv_desc),
          bwd_data_algo(other.bwd_data_algo),
          bwd_filter_algo(other.bwd_filter_algo),
          bwd_data_workspace_size(other.bwd_data_workspace_size),
          bwd_filter_workspace_size(other.bwd_filter_workspace_size),
          conv_ndim(other.conv_ndim)
#endif
    {
#ifdef ENABLE_CUDNN_API
        other.input_desc = nullptr;
        other.grad_output_desc = nullptr;
        other.weight_desc = nullptr;
        other.grad_input_desc = nullptr;
        other.grad_weight_desc = nullptr;
        other.grad_bias_desc = nullptr;
        other.conv_desc = nullptr;
        other.bwd_data_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
        other.bwd_filter_algo = static_cast<cudnnConvolutionBwdFilterAlgo_t>(0);
        other.bwd_data_workspace_size = 0;
        other.bwd_filter_workspace_size = 0;
        other.conv_ndim = 0;
#endif
        other.workspace_size = 0;
    }

    ~Opaque() {
#ifdef ENABLE_CUDNN_API
        CLEANUP_CUDNN_DESCRIPTORS();
#endif
    }

    static inline utils::Result<Opaque>
    create(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr,
           const op::conv_backward::ConvBackwardInfo &info,
           infiniDtype_t data_type, infiniopTensorDescriptor_t bias_desc) {
#ifdef ENABLE_CUDNN_API
        Opaque opaque(internal_ptr);
        auto status = opaque.initializeCudnnContext(info, data_type, bias_desc);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
        return utils::Result<Opaque>(std::move(opaque));
#else
        return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
    }
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t grad_output_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t weight_desc,
                                  infiniopTensorDescriptor_t bias_desc,
                                  void *pads, void *strides, void *dilations,
                                  size_t groups) {
#ifdef ENABLE_CUDNN_API
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto info_result = op::conv_backward::ConvBackwardInfo::create(
        grad_output_desc, input_desc, weight_desc, pads, strides, dilations,
        groups);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    auto opaque_result = Opaque::create(handle->internal(), info, dtype, bias_desc);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, opaque->workspace_size, opaque,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *grad_input, void *grad_weight,
                                     void *grad_bias, const void *grad_output,
                                     const void *input, const void *weight,
                                     void *stream) const {
#ifdef ENABLE_CUDNN_API
    const float alpha = 1.0f, beta = 0.0f;
    auto internal = _opaque->internal;

    return internal->useCudnn((cudaStream_t)stream, [&](cudnnHandle_t h) {
        if (!grad_input || !grad_weight || !grad_output || !input || !weight) {
            printf("Error: Null pointer in calculate function\n");
            return INFINI_STATUS_BAD_PARAM;
        }

        CHECK_CUDNN(cudnnConvolutionBackwardData(
            h, &alpha, _opaque->weight_desc, weight, _opaque->grad_output_desc,
            grad_output, _opaque->conv_desc, _opaque->bwd_data_algo, workspace,
            _opaque->bwd_data_workspace_size, &beta, _opaque->grad_input_desc,
            grad_input));

        CHECK_CUDNN(cudnnConvolutionBackwardFilter(
            h, &alpha, _opaque->input_desc, input, _opaque->grad_output_desc,
            grad_output, _opaque->conv_desc, _opaque->bwd_filter_algo, workspace,
            _opaque->bwd_filter_workspace_size, &beta, _opaque->grad_weight_desc,
            grad_weight));

        // grad_bias = conv_bwd_bias(grad_output)
        if (_opaque->grad_bias_desc && grad_bias) {
            cudnnDataType_t grad_output_type;
            int grad_output_nbDims;
            int grad_output_dims[5], grad_output_strides[5];

            int query_ndim = (_opaque->conv_ndim == 3) ? 5 : 4;

            CHECK_CUDNN(cudnnGetTensorNdDescriptor(
                _opaque->grad_output_desc, query_ndim, &grad_output_type,
                &grad_output_nbDims, grad_output_dims, grad_output_strides));
            if (grad_output_type == CUDNN_DATA_BFLOAT16) {
                CHECK_STATUS(launch_bias_grad_kernel(
                    grad_output, grad_bias, grad_output_dims, _opaque->conv_ndim,
                    grad_output_type, (cudaStream_t)stream));
            } else {
                CHECK_CUDNN(cudnnConvolutionBackwardBias(
                    h, &alpha, _opaque->grad_output_desc, grad_output, &beta,
                    _opaque->grad_bias_desc, grad_bias));
            }
        }
        return INFINI_STATUS_SUCCESS;
    });
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

} // namespace op::conv_backward::nvidia
