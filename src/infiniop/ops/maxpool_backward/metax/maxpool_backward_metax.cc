#include "maxpool_backward_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#define DESTROY_HCDNN_DESCRIPTOR(desc_ptr, destroy_func) \
    do {                                                 \
        if (desc_ptr) {                                  \
            destroy_func(desc_ptr);                      \
            desc_ptr = nullptr;                          \
        }                                                \
    } while (0)

#define CLEANUP_HCDNN_DESCRIPTORS()                                               \
    do {                                                                          \
        DESTROY_HCDNN_DESCRIPTOR(input_desc, hcdnnDestroyTensorDescriptor);       \
        DESTROY_HCDNN_DESCRIPTOR(grad_input_desc, hcdnnDestroyTensorDescriptor);  \
        DESTROY_HCDNN_DESCRIPTOR(grad_output_desc, hcdnnDestroyTensorDescriptor); \
        DESTROY_HCDNN_DESCRIPTOR(pooling_backward_desc,                           \
                                 hcdnnDestroyPoolingDescriptor);                  \
    } while (0)

namespace op::maxpool_backward::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    size_t workspace_size = 0;

#ifdef ENABLE_HCDNN_API
    hcdnnTensorDescriptor_t input_desc = nullptr;
    hcdnnTensorDescriptor_t grad_input_desc = nullptr;
    hcdnnTensorDescriptor_t grad_output_desc = nullptr;
    hcdnnPoolingDescriptor_t pooling_backward_desc = nullptr;
#endif

private:
    Opaque(std::shared_ptr<device::metax::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

#ifdef ENABLE_HCDNN_API
    void calculateStrides(const std::vector<int> &dims, std::vector<int> &strides,
                          int ndim) const {
        strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d) {
            strides[d] = strides[d + 1] * dims[d + 1];
        }
    }

    infiniStatus_t createPoolingDescriptors(const MaxPoolBackwardInfo &info,
                                            hcdnnDataType_t hcdnn_data_type) {
        // 创建hcdnn描述符
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&input_desc));
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&grad_input_desc));
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&grad_output_desc));
        CHECK_MCDNN(hcdnnCreatePoolingDescriptor(&pooling_backward_desc));

        // 构建输入、输出梯度维度（NCHW格式）
        std::vector<int> input_dims_vec = {static_cast<int>(info.batch),
                                           static_cast<int>(info.channels)};
        std::vector<int> output_dims_vec = {static_cast<int>(info.batch),
                                            static_cast<int>(info.channels)};
        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims_vec.push_back(static_cast<int>(info.input_dims[i]));
            output_dims_vec.push_back(static_cast<int>(info.output_dims[i]));
        }

        // 1D池化补充维度
        if (info.ndim == 1) {
            input_dims_vec.push_back(1);
            output_dims_vec.push_back(1);
        }

        // 计算内存步幅
        std::vector<int> input_strides_vec(input_dims_vec.size());
        std::vector<int> output_strides_vec(output_dims_vec.size());
        calculateStrides(input_dims_vec, input_strides_vec, input_dims_vec.size());
        calculateStrides(output_dims_vec, output_strides_vec, output_dims_vec.size());

        // 设置张量描述符（带步幅）
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            input_desc, hcdnn_data_type, input_dims_vec.size(),
            input_dims_vec.data(), input_strides_vec.data()));

        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            grad_input_desc, hcdnn_data_type, input_dims_vec.size(),
            input_dims_vec.data(), input_strides_vec.data()));

        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            grad_output_desc, hcdnn_data_type, output_dims_vec.size(),
            output_dims_vec.data(), output_strides_vec.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupPoolingDescriptor(const MaxPoolBackwardInfo &info) {
        // 构建池化参数
        std::vector<int> kernel_vec, stride_vec, pad_vec;
        for (size_t i = 0; i < info.ndim; ++i) {
            kernel_vec.push_back(static_cast<int>(info.kernel_sizes[i]));
            stride_vec.push_back(static_cast<int>(info.strides[i]));
            pad_vec.push_back(static_cast<int>(info.pads[i]));
        }

        // 1D池化补充维度
        if (info.ndim == 1) {
            kernel_vec.push_back(1);
            stride_vec.push_back(1);
            pad_vec.push_back(0);
        }

        // 设置最大池化反向描述符（确定性模式）
        CHECK_MCDNN(hcdnnSetPoolingNdDescriptor(
            pooling_backward_desc, HCDNN_POOLING_MAX_DETERMINISTIC, // 确定性最大池化
            HCDNN_NOT_PROPAGATE_NAN,                                // 不传播NaN
            kernel_vec.size(),
            kernel_vec.data(),
            pad_vec.data(),
            stride_vec.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t initializeHcdnnContext(MaxPoolBackwardInfo &info,
                                          infiniDtype_t data_type) {
        hcdnnDataType_t hcdnn_data_type = device::metax::getHcdnnDtype(data_type);

        CHECK_STATUS(createPoolingDescriptors(info, hcdnn_data_type));
        CHECK_STATUS(setupPoolingDescriptor(info));

        // 计算工作空间大小（需存储前向输出用于反向计算）
        CHECK_MCDNN(hcdnnGetTensorSizeInBytes(grad_output_desc, &workspace_size));

        return INFINI_STATUS_SUCCESS;
    }
#endif

public:
    Opaque(Opaque &&other) noexcept
        : internal(std::move(other.internal)),
          workspace_size(other.workspace_size)
#ifdef ENABLE_HCDNN_API
          ,
          input_desc(other.input_desc), grad_input_desc(other.grad_input_desc), grad_output_desc(other.grad_output_desc), pooling_backward_desc(other.pooling_backward_desc)
#endif
    {
#ifdef ENABLE_HCDNN_API
        other.input_desc = nullptr;
        other.grad_input_desc = nullptr;
        other.grad_output_desc = nullptr;
        other.pooling_backward_desc = nullptr;
#endif
        other.workspace_size = 0;
    }

    ~Opaque() {
#ifdef ENABLE_HCDNN_API
        CLEANUP_HCDNN_DESCRIPTORS();
#endif
    }

    static inline utils::Result<Opaque>
    create(std::shared_ptr<device::metax::Handle::Internal> internal_ptr,
           MaxPoolBackwardInfo &info, infiniDtype_t data_type) {
#ifdef ENABLE_HCDNN_API
        Opaque opaque(internal_ptr);
        auto status = opaque.initializeHcdnnContext(info, data_type);
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

#ifdef ENABLE_HCDNN_API
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MaxPoolBackwardInfo::create(grad_input_desc, grad_output_desc, input_desc,
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

#ifdef ENABLE_HCDNN_API
    const float alpha = 1.0f, beta = 0.0f;

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    CHECK_STATUS(_opaque->internal->useMcdnn(
        (hcStream_t)stream, [&](hcdnnHandle_t handle) {
            void *temp_output = workspace;
            CHECK_MCDNN(hcdnnPoolingForward(
                handle, _opaque->pooling_backward_desc, &alpha,
                _opaque->input_desc, input, &beta, _opaque->grad_output_desc, temp_output));

            CHECK_MCDNN(hcdnnPoolingBackward(
                handle, _opaque->pooling_backward_desc, &alpha,
                _opaque->grad_output_desc, temp_output, // 前向输出（用于定位最大值）
                _opaque->grad_output_desc, grad_output, // 输出梯度
                _opaque->input_desc, input,             // 前向输入
                &beta,
                _opaque->grad_input_desc, grad_input // 输入梯度（输出）
                ));
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

} // namespace op::maxpool_backward::metax
