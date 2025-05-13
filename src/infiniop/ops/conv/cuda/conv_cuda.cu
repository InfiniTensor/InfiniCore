#include "../../../devices/cuda/cuda_handle.cuh"
#include "conv_cuda.cuh"

namespace op::conv::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
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
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
    auto handle = reinterpret_cast<device::cuda::nvidia::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc,
                                   pads, strides, dilations, n);

    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    void *stream) const {
    int maxAlgoCount = 0;
    CHECK_STATUS(_opaque->internal->useCudnn(
        (cudaStream_t)stream, [&](cudnnHandle_t handle) {
            if (!handle) {
                return INFINI_STATUS_BAD_PARAM;
            }
            CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
                handle, &maxAlgoCount));
            return INFINI_STATUS_SUCCESS;
        }));
    if (maxAlgoCount <= 0) {
        maxAlgoCount = 8;
    }
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(maxAlgoCount);
    int algoCounts = 0;
    CHECK_STATUS(_opaque->internal->useCudnn(
        (cudaStream_t)stream, [&](cudnnHandle_t handle) {
            CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
                handle,
                _info.handler->x_desc,
                _info.handler->w_desc,
                _info.handler->conv_desc,
                _info.handler->y_desc,
                maxAlgoCount,
                &algoCounts,
                perf_results.data()));
            return INFINI_STATUS_SUCCESS;
        }));
    cudnnConvolutionFwdAlgo_t chosenAlgo;
    size_t chosenWs = 0;
    if (_info.ndim == 1) {
        chosenAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        chosenWs = 0;
    } else {
        bool found = false;
        for (int i = 0; i < algoCounts; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) {
                continue;
            }
            size_t ws = 0;
            if (!_opaque->internal->useCudnn(
                    (cudaStream_t)stream,
                    [&](cudnnHandle_t handle) {
                        cudnnStatus_t st = cudnnGetConvolutionForwardWorkspaceSize(
                            handle,
                            _info.handler->x_desc, _info.handler->w_desc, _info.handler->conv_desc, _info.handler->y_desc,
                            perf_results[i].algo,
                            &ws);
                        return st == CUDNN_STATUS_SUCCESS
                                 ? INFINI_STATUS_SUCCESS
                                 : INFINI_STATUS_BAD_PARAM; // 根据cuDNN返回状态判断
                    })) {
                continue;
            }
            if (ws <= workspace_size) {
                chosenAlgo = perf_results[i].algo;
                chosenWs = ws;
                found = true;
                break;
            }
        }
        if (!found) {
            chosenAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            chosenWs = 0;
        }
    }

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_STATUS(_opaque->internal->useCudnn(
        (cudaStream_t)stream, [&](cudnnHandle_t handle) {
            CHECK_CUDNN(cudnnConvolutionForward(
                handle,
                &alpha,
                _info.handler->x_desc,
                x,
                _info.handler->w_desc,
                w,
                _info.handler->conv_desc,
                chosenAlgo,
                workspace,
                chosenWs,
                &beta,
                _info.handler->y_desc,
                y));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::conv::cuda
