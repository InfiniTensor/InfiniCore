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
        int requestedAlgoCount = 1;
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &requestedAlgoCount));
                return INFINI_STATUS_SUCCESS;
            }
        ));
        int algoCounts = 0;
        int chosenAlgoIndex = 0;
        bool chosen = false;
        size_t workspace_size = 0;
        std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_(requestedAlgoCount);
        auto perf_results = perf_results_.data();
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle,
                    _info.handler->x_desc,
                    _info.handler->w_desc,
                    _info.handler->conv_desc,
                    _info.handler->y_desc,
                    requestedAlgoCount,
                    &algoCounts,
                    perf_results));
                return INFINI_STATUS_SUCCESS;
            }
        ));
        if (algoCounts < 1) {
            return INFINI_STATUS_BAD_PARAM;
        }
        for (int i = 0; i < algoCounts; i++) {
            if (_opaque->internal->useCudnn((cudaStream_t)stream, [&](cudnnHandle_t handle) {CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, _info.handler->x_desc, _info.handler->w_desc, _info.handler->conv_desc, _info.handler->y_desc, perf_results[i].algo, &workspace_size)); return INFINI_STATUS_SUCCESS;})) {
                chosenAlgoIndex = i;
                chosen = true;
                break;
            }
        }
        if (!chosen) {
            return INFINI_STATUS_BAD_PARAM;
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_STATUS(_opaque->internal->useCudnn(
            (cudaStream_t)stream, [&](cudnnHandle_t handle) {
                CHECK_CUDNN(cudnnConvolutionForward(handle,
                    &alpha,
                    _info.handler->x_desc,
                    x,
                    _info.handler->w_desc,
                    w,
                    _info.handler->conv_desc,
                    perf_results[chosenAlgoIndex].algo,
                    workspace,
                    workspace_size,
                    &beta,
                    _info.handler->y_desc,
                    y));
                return INFINI_STATUS_SUCCESS;
            }
        ));
        return INFINI_STATUS_SUCCESS;
    }
    
}
