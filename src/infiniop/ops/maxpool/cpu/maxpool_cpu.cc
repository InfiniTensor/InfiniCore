#include "maxpool_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace op::maxpool::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    MaxPoolInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const MaxPoolInfo &maxpool_info)
        : handle(handle_ptr), info(maxpool_info) {
        // CPU实现通常不需要额外的工作空间
        workspace_size = 0;
    }

    // 获取数据类型的最小值
    template <typename T>
    static T get_min_value() {
        if constexpr (std::is_same<T, float>::value) {
            return -std::numeric_limits<float>::infinity();
        } else if constexpr (std::is_same<T, fp16_t>::value) {
            return _f32_to_f16(-std::numeric_limits<float>::infinity());
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            return _f32_to_bf16(-std::numeric_limits<float>::infinity());
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }

    // 比较两个值的大小（处理半精度类型）
    template <typename T>
    static bool is_greater(const T &a, const T &b) {
        if constexpr (std::is_same<T, fp16_t>::value) {
            return utils::cast<float>(a) > utils::cast<float>(b);
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            return utils::cast<float>(a) > utils::cast<float>(b);
        } else {
            return a > b;
        }
    }

    // 1D最大池化
    template <typename T>
    void maxpool_1d(T *output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_width = info.input_dims[0];
        size_t output_width = info.output_dims[0];
        size_t kernel_width = info.kernel_sizes[0];
        size_t stride_width = info.strides[0];
        size_t pad_width = info.pads[0];

        // 并行处理每个批次和通道
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_width + c * input_width;
                size_t output_offset = b * channels * output_width + c * output_width;

                for (size_t ow = 0; ow < output_width; ++ow) {
                    T max_val = get_min_value<T>();
                    bool found_valid = false;

                    int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                    int end_w = start_w + static_cast<int>(kernel_width);

                    for (int kw = start_w; kw < end_w; ++kw) {
                        if (kw >= 0 && kw < static_cast<int>(input_width)) {
                            T val = input[input_offset + kw];
                            if (!found_valid || is_greater(val, max_val)) {
                                max_val = val;
                                found_valid = true;
                            }
                        }
                    }

                    output[output_offset + ow] = max_val;
                }
            }
        }
    }

    // 2D最大池化
    template <typename T>
    void maxpool_2d(T *output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_height = info.input_dims[0];
        size_t input_width = info.input_dims[1];
        size_t output_height = info.output_dims[0];
        size_t output_width = info.output_dims[1];
        size_t kernel_height = info.kernel_sizes[0];
        size_t kernel_width = info.kernel_sizes[1];
        size_t stride_height = info.strides[0];
        size_t stride_width = info.strides[1];
        size_t pad_height = info.pads[0];
        size_t pad_width = info.pads[1];

        // 并行处理每个批次和通道
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_height * input_width + c * input_height * input_width;
                size_t output_offset = b * channels * output_height * output_width + c * output_height * output_width;

                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        T max_val = get_min_value<T>();
                        bool found_valid = false;

                        int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                        int end_h = start_h + static_cast<int>(kernel_height);
                        int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                        int end_w = start_w + static_cast<int>(kernel_width);

                        for (int kh = start_h; kh < end_h; ++kh) {
                            for (int kw = start_w; kw < end_w; ++kw) {
                                if (kh >= 0 && kh < static_cast<int>(input_height) && kw >= 0 && kw < static_cast<int>(input_width)) {
                                    T val = input[input_offset + kh * input_width + kw];
                                    if (!found_valid || is_greater(val, max_val)) {
                                        max_val = val;
                                        found_valid = true;
                                    }
                                }
                            }
                        }

                        output[output_offset + oh * output_width + ow] = max_val;
                    }
                }
            }
        }
    }

    // 3D最大池化
    template <typename T>
    void maxpool_3d(T *output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_depth = info.input_dims[0];
        size_t input_height = info.input_dims[1];
        size_t input_width = info.input_dims[2];
        size_t output_depth = info.output_dims[0];
        size_t output_height = info.output_dims[1];
        size_t output_width = info.output_dims[2];
        size_t kernel_depth = info.kernel_sizes[0];
        size_t kernel_height = info.kernel_sizes[1];
        size_t kernel_width = info.kernel_sizes[2];
        size_t stride_depth = info.strides[0];
        size_t stride_height = info.strides[1];
        size_t stride_width = info.strides[2];
        size_t pad_depth = info.pads[0];
        size_t pad_height = info.pads[1];
        size_t pad_width = info.pads[2];

        // 并行处理每个批次和通道
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_depth * input_height * input_width + c * input_depth * input_height * input_width;
                size_t output_offset = b * channels * output_depth * output_height * output_width + c * output_depth * output_height * output_width;

                for (size_t od = 0; od < output_depth; ++od) {
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            T max_val = get_min_value<T>();
                            bool found_valid = false;

                            int start_d = static_cast<int>(od * stride_depth) - static_cast<int>(pad_depth);
                            int end_d = start_d + static_cast<int>(kernel_depth);
                            int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                            int end_h = start_h + static_cast<int>(kernel_height);
                            int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                            int end_w = start_w + static_cast<int>(kernel_width);

                            for (int kd = start_d; kd < end_d; ++kd) {
                                for (int kh = start_h; kh < end_h; ++kh) {
                                    for (int kw = start_w; kw < end_w; ++kw) {
                                        if (kd >= 0 && kd < static_cast<int>(input_depth) && kh >= 0 && kh < static_cast<int>(input_height) && kw >= 0 && kw < static_cast<int>(input_width)) {
                                            T val = input[input_offset + kd * input_height * input_width + kh * input_width + kw];
                                            if (!found_valid || is_greater(val, max_val)) {
                                                max_val = val;
                                                found_valid = true;
                                            }
                                        }
                                    }
                                }
                            }

                            output[output_offset + od * output_height * output_width + oh * output_width + ow] = max_val;
                        }
                    }
                }
            }
        }
    }

    // 主要的最大池化计算函数
    template <typename T>
    void maxpool_cpu(T *output, const T *input) const {
        switch (info.ndim) {
        case 1:
            maxpool_1d(output, input);
            break;
        case 2:
            maxpool_2d(output, input);
            break;
        case 3:
            maxpool_3d(output, input);
            break;
        default:
            break;
        }
    }

public:
    Opaque(Opaque &&other) noexcept
        : handle(other.handle),
          info(std::move(other.info)),
          workspace_size(other.workspace_size) {
        other.handle = nullptr;
        other.workspace_size = 0;
    }

    ~Opaque() = default;

    static inline utils::Result<Opaque>
    create(device::cpu::Handle *handle_ptr,
           MaxPoolInfo &info,
           infiniDtype_t data_type) {
        if (data_type != INFINI_DTYPE_F32 && data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *output, const void *input, infiniDtype_t dtype) const {

        if (!output || !input) {
            return INFINI_STATUS_BAD_PARAM;
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *typed_output = static_cast<float *>(output);
            const float *typed_input = static_cast<const float *>(input);
            maxpool_cpu(typed_output, typed_input);
            break;
        }

        case INFINI_DTYPE_F16: {
            fp16_t *typed_output = static_cast<fp16_t *>(output);
            const fp16_t *typed_input = static_cast<const fp16_t *>(input);
            maxpool_cpu(typed_output, typed_input);
            break;
        }

        case INFINI_DTYPE_BF16: {
            bf16_t *typed_output = static_cast<bf16_t *>(output);
            const bf16_t *typed_input = static_cast<const bf16_t *>(input);
            maxpool_cpu(typed_output, typed_input);
            break;
        }

        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return INFINI_STATUS_SUCCESS;
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
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = MaxPoolInfo::create(output_desc, input_desc, kernel_size,
                                      strides, pads, ceil_mode);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque_result = Opaque::create(handle, info, dtype);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, std::move(info), opaque->workspace_size,
                               opaque, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *output, const void *input,
                                     void *stream) const {
    return _opaque->calculate(workspace, workspace_size, output, input, _dtype);
}

} // namespace op::maxpool::cpu
