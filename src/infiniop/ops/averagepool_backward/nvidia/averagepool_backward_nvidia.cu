#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "averagepool_backward_nvidia.cuh"

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
        DESTROY_CUDNN_DESCRIPTOR(grad_input_desc, cudnnDestroyTensorDescriptor);  \
        DESTROY_CUDNN_DESCRIPTOR(grad_output_desc, cudnnDestroyTensorDescriptor); \
        DESTROY_CUDNN_DESCRIPTOR(pooling_backward_desc,                           \
                                 cudnnDestroyPoolingDescriptor);                  \
    } while (0)

namespace op::averagepool_backward::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    size_t workspace_size = 0;

#ifdef ENABLE_CUDNN_API
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t grad_input_desc = nullptr;
    cudnnTensorDescriptor_t grad_output_desc = nullptr;
    cudnnPoolingDescriptor_t pooling_backward_desc = nullptr;
#endif

private:
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

#ifdef ENABLE_CUDNN_API
    infiniStatus_t getCudnnDataType(infiniDtype_t data_type,
                                    cudnnDataType_t &cudnn_data_type) const {
        if (data_type == INFINI_DTYPE_F16) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else if (data_type == INFINI_DTYPE_F32) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else if (data_type == INFINI_DTYPE_BF16) {
            cudnn_data_type = device::nvidia::getCudnnDtype(data_type);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return INFINI_STATUS_SUCCESS;
    }

    void calculateStrides(const std::vector<int> &dims, std::vector<int> &strides,
                          int ndim) const {
        strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * dims[d + 1];
        }
    }

    infiniStatus_t createPoolingDescriptors(const AvgPoolBackwardInfo &info,
                                            cudnnDataType_t cudnn_data_type) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_input_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&grad_output_desc));
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_backward_desc));

        std::vector<int> input_dims_vec = {static_cast<int>(info.batch),
                                           static_cast<int>(info.channels)};
        std::vector<int> output_dims_vec = {static_cast<int>(info.batch),
                                            static_cast<int>(info.channels)};

        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims_vec.push_back(static_cast<int>(info.input_dims[i]));
            output_dims_vec.push_back(static_cast<int>(info.output_dims[i]));
        }

        if (info.ndim == 1) {
            input_dims_vec.push_back(1);
            output_dims_vec.push_back(1);
        }

        std::vector<int> input_strides_vec(input_dims_vec.size());
        std::vector<int> output_strides_vec(output_dims_vec.size());
        calculateStrides(input_dims_vec, input_strides_vec, input_dims_vec.size());
        calculateStrides(output_dims_vec, output_strides_vec,
                         output_dims_vec.size());

        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            input_desc, cudnn_data_type, input_dims_vec.size(),
            input_dims_vec.data(), input_strides_vec.data()));

        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            grad_input_desc, cudnn_data_type, input_dims_vec.size(),
            input_dims_vec.data(), input_strides_vec.data()));

        CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            grad_output_desc, cudnn_data_type, output_dims_vec.size(),
            output_dims_vec.data(), output_strides_vec.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupPoolingDescriptor(const AvgPoolBackwardInfo &info) {
        std::vector<int> kernel_vec, stride_vec, pad_vec;
        for (size_t i = 0; i < info.ndim; ++i) {
            kernel_vec.push_back(static_cast<int>(info.kernel_sizes[i]));
            stride_vec.push_back(static_cast<int>(info.strides[i]));
            pad_vec.push_back(static_cast<int>(info.pads[i]));
        }

        if (info.ndim == 1) {
            kernel_vec.push_back(1);
            stride_vec.push_back(1);
            pad_vec.push_back(0);
        }

        CHECK_CUDNN(cudnnSetPoolingNdDescriptor(
            pooling_backward_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,
            kernel_vec.size(), kernel_vec.data(), pad_vec.data(),
            stride_vec.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t initializeCudnnContext(AvgPoolBackwardInfo &info,
                                          infiniDtype_t data_type) {
        cudnnDataType_t cudnn_data_type;
        CHECK_STATUS(getCudnnDataType(data_type, cudnn_data_type));

        CHECK_STATUS(createPoolingDescriptors(info, cudnn_data_type));
        CHECK_STATUS(setupPoolingDescriptor(info));

        CHECK_CUDNN(cudnnGetTensorSizeInBytes(grad_output_desc, &workspace_size));

        return INFINI_STATUS_SUCCESS;
    }
#endif

public:
    Opaque(Opaque &&other) noexcept
        : internal(std::move(other.internal)),
          workspace_size(other.workspace_size)
#ifdef ENABLE_CUDNN_API
          ,
          input_desc(other.input_desc), grad_input_desc(other.grad_input_desc), grad_output_desc(other.grad_output_desc), pooling_backward_desc(other.pooling_backward_desc)
#endif
    {
#ifdef ENABLE_CUDNN_API
        other.input_desc = nullptr;
        other.grad_input_desc = nullptr;
        other.grad_output_desc = nullptr;
        other.pooling_backward_desc = nullptr;
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
           AvgPoolBackwardInfo &info, infiniDtype_t data_type) {
#ifdef ENABLE_CUDNN_API
        Opaque opaque(internal_ptr);
        auto status = opaque.initializeCudnnContext(info, data_type);
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
                                  infiniopTensorDescriptor_t grad_input_desc,
                                  infiniopTensorDescriptor_t grad_output_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  void *kernel_size, void *strides, void *pads,
                                  bool ceil_mode) {

#ifdef ENABLE_CUDNN_API
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = AvgPoolBackwardInfo::create(grad_input_desc, grad_output_desc, input_desc,
                                              kernel_size, strides, pads, ceil_mode);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque_result = Opaque::create(handle->internal(), info, dtype);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, std::move(info), opaque->workspace_size,
                               opaque, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *grad_input, const void *grad_output,
                                     const void *input, void *stream) const {

#ifdef ENABLE_CUDNN_API
    const float alpha = 1.0f, beta = 0.0f;

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_STATUS(_opaque->internal->useCudnn(
        (cudaStream_t)stream, [&](cudnnHandle_t handle) {
            size_t grad_input_size = 0;
            CHECK_CUDNN(cudnnGetTensorSizeInBytes(_opaque->grad_input_desc,
                                                  &grad_input_size));
            CHECK_CUDA(cudaMemset(grad_input, 0, grad_input_size));
            CHECK_CUDA(cudaMemset(workspace, 0, _workspace_size));

            void *temp_output = workspace;
            CHECK_CUDNN(cudnnPoolingForward(
                handle, _opaque->pooling_backward_desc, &alpha, _opaque->input_desc,
                input, &beta, _opaque->grad_output_desc, temp_output));

            CHECK_CUDNN(cudnnPoolingBackward(
                handle, _opaque->pooling_backward_desc, &alpha,
                _opaque->grad_output_desc, temp_output,
                _opaque->grad_output_desc, grad_output,
                _opaque->input_desc, input,
                &beta,
                _opaque->grad_input_desc, grad_input));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

} // namespace op::averagepool_backward::nvidia
