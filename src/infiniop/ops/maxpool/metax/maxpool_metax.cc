#include "maxpool_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

#define DESTROY_hcdnn_DESCRIPTOR(desc_ptr, destroy_func) \
    do {                                                 \
        if (desc_ptr) {                                  \
            destroy_func(desc_ptr);                      \
            desc_ptr = nullptr;                          \
        }                                                \
    } while (0)

#define CLEANUP_hcdnn_DESCRIPTORS()                                            \
    do {                                                                       \
        DESTROY_hcdnn_DESCRIPTOR(input_desc, hcdnnDestroyTensorDescriptor);    \
        DESTROY_hcdnn_DESCRIPTOR(output_desc, hcdnnDestroyTensorDescriptor);   \
        DESTROY_hcdnn_DESCRIPTOR(pooling_desc, hcdnnDestroyPoolingDescriptor); \
    } while (0)

namespace op::maxpool::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    size_t workspace_size = 0;

#ifdef ENABLE_HCDNN_API
    hcdnnTensorDescriptor_t input_desc = nullptr;
    hcdnnTensorDescriptor_t output_desc = nullptr;
    hcdnnPoolingDescriptor_t pooling_desc = nullptr;
#endif

private:
    Opaque(std::shared_ptr<device::metax::Handle::Internal> internal_ptr)
        : internal(internal_ptr) {}

#ifdef ENABLE_HCDNN_API
    infiniStatus_t createPoolingDescriptors(const MaxPoolInfo &info,
                                            hcdnnDataType_t hcdnn_data_type) {
        // 创建输入输出张量描述符
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&input_desc));
        CHECK_MCDNN(hcdnnCreateTensorDescriptor(&output_desc));
        CHECK_MCDNN(hcdnnCreatePoolingDescriptor(&pooling_desc));

        // 构建输入输出维度（NCHW格式）
        std::vector<int> input_dims = {static_cast<int>(info.batch),
                                       static_cast<int>(info.channels)};
        std::vector<int> output_dims = {static_cast<int>(info.batch),
                                        static_cast<int>(info.channels)};
        for (size_t i = 0; i < info.ndim; ++i) {
            input_dims.push_back(static_cast<int>(info.input_dims[i]));
            output_dims.push_back(static_cast<int>(info.output_dims[i]));
        }

        // 1D池化补充维度
        if (info.ndim == 1) {
            input_dims.push_back(1);
            output_dims.push_back(1);
        }

        // 计算输入输出张量的步幅
        std::vector<int> input_strides(input_dims.size(), 1);
        std::vector<int> output_strides(output_dims.size(), 1);
        for (int i = input_dims.size() - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
            output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
        }

        // 设置张量描述符（NCHW格式）
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            input_desc, hcdnn_data_type, input_dims.size(), input_dims.data(), input_strides.data()));
        CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
            output_desc, hcdnn_data_type, output_dims.size(), output_dims.data(),
            output_strides.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t setupPoolingDescriptor(const MaxPoolInfo &info) {
        // 构建池化参数
        std::vector<int> kernel_size, strides, pads;
        for (size_t i = 0; i < info.ndim; ++i) {
            kernel_size.push_back(static_cast<int>(info.kernel_sizes[i]));
            strides.push_back(static_cast<int>(info.strides[i]));
            pads.push_back(static_cast<int>(info.pads[i]));
        }

        // 1D池化补充维度
        if (info.ndim == 1) {
            kernel_size.push_back(1);
            strides.push_back(1);
            pads.push_back(0);
        }

        // 设置最大池化描述符（确定性模式）
        CHECK_MCDNN(hcdnnSetPoolingNdDescriptor(
            pooling_desc, HCDNN_POOLING_MAX_DETERMINISTIC, // 确定性最大池化
            HCDNN_NOT_PROPAGATE_NAN,                       // 不传播NaN
            kernel_size.size(),
            kernel_size.data(),
            pads.data(),
            strides.data()));

        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t initializehcdnnContext(MaxPoolInfo &info,
                                          infiniDtype_t data_type) {
        hcdnnDataType_t hcdnn_data_type = device::metax::getHcdnnDtype(data_type);
        CHECK_STATUS(createPoolingDescriptors(info, hcdnn_data_type));
        CHECK_STATUS(setupPoolingDescriptor(info));

        // 最大池化通常不需要工作空间
        workspace_size = 0;

        return INFINI_STATUS_SUCCESS;
    }
#endif

public:
    Opaque(Opaque &&other) noexcept
        : internal(std::move(other.internal)),
          workspace_size(other.workspace_size)
#ifdef ENABLE_HCDNN_API
          ,
          input_desc(other.input_desc), output_desc(other.output_desc), pooling_desc(other.pooling_desc)
#endif
    {
#ifdef ENABLE_HCDNN_API
        other.input_desc = nullptr;
        other.output_desc = nullptr;
        other.pooling_desc = nullptr;
#endif
        other.workspace_size = 0;
    }

    ~Opaque() {
#ifdef ENABLE_HCDNN_API
        CLEANUP_hcdnn_DESCRIPTORS();
#endif
    }

    static inline utils::Result<Opaque>
    create(std::shared_ptr<device::metax::Handle::Internal> internal_ptr,
           MaxPoolInfo &info, infiniDtype_t data_type) {
#ifdef ENABLE_HCDNN_API
        Opaque opaque(internal_ptr);
        auto status = opaque.initializehcdnnContext(info, data_type);
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
                                  infiniopTensorDescriptor_t output_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  void *kernel_size, void *strides, void *pads,
                                  bool ceil_mode) {

#ifdef ENABLE_HCDNN_API
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MaxPoolInfo::create(output_desc, input_desc, kernel_size,
                                      strides, pads, ceil_mode);
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
                                     void *output, const void *input,
                                     void *stream) const {

#ifdef ENABLE_HCDNN_API
    const float alpha = 1.0f, beta = 0.0f;

    // 执行最大池化前向计算
    CHECK_STATUS(_opaque->internal->useMcdnn(
        (hcStream_t)stream, [&](hcdnnHandle_t handle) {
            CHECK_MCDNN(hcdnnPoolingForward(handle, _opaque->pooling_desc, &alpha,
                                            _opaque->input_desc, input, &beta,
                                            _opaque->output_desc, output));
            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
#else
    return INFINI_STATUS_NOT_IMPLEMENTED;
#endif
}

} // namespace op::maxpool::metax
