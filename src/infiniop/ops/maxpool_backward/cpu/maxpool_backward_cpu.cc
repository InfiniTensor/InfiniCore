#include "maxpool_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace op::maxpool_backward::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    MaxPoolBackwardInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const MaxPoolBackwardInfo &maxpool_info)
        : handle(handle_ptr), info(maxpool_info) {
        workspace_size = 0;
    }

    // F16专用：使用float计算的最大池化反向传播
    void maxpool_backward_f16_as_float(fp16_t *grad_input, const fp16_t *grad_output, const fp16_t *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;

        // 计算总的输入和输出大小
        size_t total_input_size = batch_size * channels;
        size_t total_output_size = batch_size * channels;

        for (size_t i = 0; i < info.ndim; ++i) {
            total_input_size *= info.input_dims[i];
            total_output_size *= info.output_dims[i];
        }

        // 分配float临时缓冲区
        std::vector<float> float_input(total_input_size);
        std::vector<float> float_grad_output(total_output_size);
        std::vector<float> float_grad_input(total_input_size, 0.0f);

        // 转换输入数据为float
        for (size_t i = 0; i < total_input_size; ++i) {
            float_input[i] = utils::cast<float>(input[i]);
        }
        for (size_t i = 0; i < total_output_size; ++i) {
            float_grad_output[i] = utils::cast<float>(grad_output[i]);
        }

        // 使用float精度进行计算
        maxpool_backward_cpu_float(float_grad_input.data(), float_grad_output.data(), float_input.data());

        // 转换结果回F16
        for (size_t i = 0; i < total_input_size; ++i) {
            grad_input[i] = utils::cast<fp16_t>(float_grad_input[i]);
        }
    }

    // Float版本的最大池化反向传播
    void maxpool_backward_cpu_float(float *grad_input, const float *grad_output, const float *input) const {
        switch (info.ndim) {
        case 1:
            maxpool_backward_1d_float(grad_input, grad_output, input);
            break;
        case 2:
            maxpool_backward_2d_float(grad_input, grad_output, input);
            break;
        case 3:
            maxpool_backward_3d_float(grad_input, grad_output, input);
            break;
        default:
            break;
        }
    }

    // 1D float版本
    void maxpool_backward_1d_float(float *grad_input, const float *grad_output, const float *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_width = info.input_dims[0];
        size_t output_width = info.output_dims[0];
        size_t kernel_width = info.kernel_sizes[0];
        size_t stride_width = info.strides[0];
        size_t pad_width = info.pads[0];

        // 初始化梯度输入为零
        size_t total_input_size = batch_size * channels * input_width;
        std::fill(grad_input, grad_input + total_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_width + c * input_width;
                size_t output_offset = b * channels * output_width + c * output_width;

                for (size_t ow = 0; ow < output_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_idx = 0;
                    bool found_max = false;

                    int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                    int end_w = start_w + static_cast<int>(kernel_width);

                    for (int kw = start_w; kw < end_w; ++kw) {
                        if (kw >= 0 && kw < static_cast<int>(input_width)) {
                            size_t real_kw = static_cast<size_t>(kw);
                            float val = input[input_offset + real_kw];

                            if (!found_max || val > max_val || (val == max_val && real_kw < max_idx)) {
                                max_val = val;
                                max_idx = real_kw;
                                found_max = true;
                            }
                        }
                    }

                    if (found_max) {
                        grad_input[input_offset + max_idx] += grad_output[output_offset + ow];
                    }
                }
            }
        }
    }

    // 2D float版本
    void maxpool_backward_2d_float(float *grad_input, const float *grad_output, const float *input) const {
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

        // 初始化梯度输入为零
        size_t total_input_size = batch_size * channels * input_height * input_width;
        std::fill(grad_input, grad_input + total_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_height * input_width + c * input_height * input_width;
                size_t output_offset = b * channels * output_height * output_width + c * output_height * output_width;

                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        size_t max_h = 0, max_w = 0;
                        bool found_max = false;

                        int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                        int end_h = start_h + static_cast<int>(kernel_height);
                        int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                        int end_w = start_w + static_cast<int>(kernel_width);

                        for (int kh = start_h; kh < end_h; ++kh) {
                            for (int kw = start_w; kw < end_w; ++kw) {
                                if (kh >= 0 && kh < static_cast<int>(input_height) && kw >= 0 && kw < static_cast<int>(input_width)) {
                                    size_t real_kh = static_cast<size_t>(kh);
                                    size_t real_kw = static_cast<size_t>(kw);
                                    float val = input[input_offset + real_kh * input_width + real_kw];

                                    size_t linear_idx = real_kh * input_width + real_kw;
                                    size_t old_linear_idx = found_max ? max_h * input_width + max_w : SIZE_MAX;

                                    if (!found_max || val > max_val || (val == max_val && linear_idx < old_linear_idx)) {
                                        max_val = val;
                                        max_h = real_kh;
                                        max_w = real_kw;
                                        found_max = true;
                                    }
                                }
                            }
                        }

                        if (found_max) {
                            size_t grad_input_idx = input_offset + max_h * input_width + max_w;
                            grad_input[grad_input_idx] += grad_output[output_offset + oh * output_width + ow];
                        }
                    }
                }
            }
        }
    }

    // 3D float版本
    void maxpool_backward_3d_float(float *grad_input, const float *grad_output, const float *input) const {
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

        // 初始化梯度输入为零
        size_t total_input_size = batch_size * channels * input_depth * input_height * input_width;
        std::fill(grad_input, grad_input + total_input_size, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_depth * input_height * input_width + c * input_depth * input_height * input_width;
                size_t output_offset = b * channels * output_depth * output_height * output_width + c * output_depth * output_height * output_width;

                for (size_t od = 0; od < output_depth; ++od) {
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            float max_val = -std::numeric_limits<float>::infinity();
                            size_t max_d = 0, max_h = 0, max_w = 0;
                            bool found_max = false;

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

                                            size_t real_kd = static_cast<size_t>(kd);
                                            size_t real_kh = static_cast<size_t>(kh);
                                            size_t real_kw = static_cast<size_t>(kw);

                                            float val = input[input_offset + real_kd * input_height * input_width + real_kh * input_width + real_kw];

                                            size_t linear_idx = real_kd * input_height * input_width + real_kh * input_width + real_kw;
                                            size_t old_linear_idx = found_max ? max_d * input_height * input_width + max_h * input_width + max_w : SIZE_MAX;

                                            if (!found_max || val > max_val || (val == max_val && linear_idx < old_linear_idx)) {
                                                max_val = val;
                                                max_d = real_kd;
                                                max_h = real_kh;
                                                max_w = real_kw;
                                                found_max = true;
                                            }
                                        }
                                    }
                                }
                            }

                            if (found_max) {
                                size_t grad_input_idx = input_offset + max_d * input_height * input_width + max_h * input_width + max_w;
                                grad_input[grad_input_idx] += grad_output[output_offset + od * output_height * output_width + oh * output_width + ow];
                            }
                        }
                    }
                }
            }
        }
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

    // 检查两个值是否相等（处理半精度类型）
    template <typename T>
    static bool values_equal(const T &a, const T &b) {
        if constexpr (std::is_same<T, fp16_t>::value) {
            return utils::cast<float>(a) == utils::cast<float>(b);
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            return utils::cast<float>(a) == utils::cast<float>(b);
        } else {
            return a == b;
        }
    }

    // 原始的通用实现（用于F32和BF16）
    template <typename T>
    void maxpool_backward_cpu(T *grad_input, const T *grad_output, const T *input) const {
        switch (info.ndim) {
        case 1:
            maxpool_backward_1d_generic(grad_input, grad_output, input);
            break;
        case 2:
            maxpool_backward_2d_generic(grad_input, grad_output, input);
            break;
        case 3:
            maxpool_backward_3d_generic(grad_input, grad_output, input);
            break;
        default:
            break;
        }
    }

    template <typename T>
    void maxpool_backward_1d_generic(T *grad_input, const T *grad_output, const T *input) const {
        size_t batch_size = info.batch;
        size_t channels = info.channels;
        size_t input_width = info.input_dims[0];
        size_t output_width = info.output_dims[0];
        size_t kernel_width = info.kernel_sizes[0];
        size_t stride_width = info.strides[0];
        size_t pad_width = info.pads[0];

        size_t total_input_size = batch_size * channels * input_width;
        std::fill(grad_input, grad_input + total_input_size, T{});

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_width + c * input_width;
                size_t output_offset = b * channels * output_width + c * output_width;

                for (size_t ow = 0; ow < output_width; ++ow) {
                    T max_val = get_min_value<T>();
                    size_t max_idx = 0;
                    bool found_max = false;

                    int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                    int end_w = start_w + static_cast<int>(kernel_width);

                    for (int kw = start_w; kw < end_w; ++kw) {
                        if (kw >= 0 && kw < static_cast<int>(input_width)) {
                            size_t real_kw = static_cast<size_t>(kw);
                            T val = input[input_offset + real_kw];

                            if (!found_max || is_greater(val, max_val) || (values_equal(val, max_val) && real_kw < max_idx)) {
                                max_val = val;
                                max_idx = real_kw;
                                found_max = true;
                            }
                        }
                    }

                    if (found_max) {
                        if constexpr (std::is_same<T, bf16_t>::value) {
                            float current = utils::cast<float>(grad_input[input_offset + max_idx]);
                            float to_add = utils::cast<float>(grad_output[output_offset + ow]);
                            grad_input[input_offset + max_idx] = utils::cast<T>(current + to_add);
                        } else {
                            grad_input[input_offset + max_idx] += grad_output[output_offset + ow];
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void maxpool_backward_2d_generic(T *grad_input, const T *grad_output, const T *input) const {
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

        size_t total_input_size = batch_size * channels * input_height * input_width;
        std::fill(grad_input, grad_input + total_input_size, T{});

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_height * input_width + c * input_height * input_width;
                size_t output_offset = b * channels * output_height * output_width + c * output_height * output_width;

                for (size_t oh = 0; oh < output_height; ++oh) {
                    for (size_t ow = 0; ow < output_width; ++ow) {
                        T max_val = get_min_value<T>();
                        size_t max_h = 0, max_w = 0;
                        bool found_max = false;

                        int start_h = static_cast<int>(oh * stride_height) - static_cast<int>(pad_height);
                        int end_h = start_h + static_cast<int>(kernel_height);
                        int start_w = static_cast<int>(ow * stride_width) - static_cast<int>(pad_width);
                        int end_w = start_w + static_cast<int>(kernel_width);

                        for (int kh = start_h; kh < end_h; ++kh) {
                            for (int kw = start_w; kw < end_w; ++kw) {
                                if (kh >= 0 && kh < static_cast<int>(input_height) && kw >= 0 && kw < static_cast<int>(input_width)) {
                                    size_t real_kh = static_cast<size_t>(kh);
                                    size_t real_kw = static_cast<size_t>(kw);
                                    T val = input[input_offset + real_kh * input_width + real_kw];

                                    size_t linear_idx = real_kh * input_width + real_kw;
                                    size_t old_linear_idx = found_max ? max_h * input_width + max_w : SIZE_MAX;

                                    if (!found_max || is_greater(val, max_val) || (values_equal(val, max_val) && linear_idx < old_linear_idx)) {
                                        max_val = val;
                                        max_h = real_kh;
                                        max_w = real_kw;
                                        found_max = true;
                                    }
                                }
                            }
                        }

                        if (found_max) {
                            size_t grad_input_idx = input_offset + max_h * input_width + max_w;
                            if constexpr (std::is_same<T, bf16_t>::value) {
                                float current = utils::cast<float>(grad_input[grad_input_idx]);
                                float to_add = utils::cast<float>(grad_output[output_offset + oh * output_width + ow]);
                                grad_input[grad_input_idx] = utils::cast<T>(current + to_add);
                            } else {
                                grad_input[grad_input_idx] += grad_output[output_offset + oh * output_width + ow];
                            }
                        }
                    }
                }
            }
        }
    }

    template <typename T>
    void maxpool_backward_3d_generic(T *grad_input, const T *grad_output, const T *input) const {
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

        size_t total_input_size = batch_size * channels * input_depth * input_height * input_width;
        std::fill(grad_input, grad_input + total_input_size, T{});

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                size_t input_offset = b * channels * input_depth * input_height * input_width + c * input_depth * input_height * input_width;
                size_t output_offset = b * channels * output_depth * output_height * output_width + c * output_depth * output_height * output_width;

                for (size_t od = 0; od < output_depth; ++od) {
                    for (size_t oh = 0; oh < output_height; ++oh) {
                        for (size_t ow = 0; ow < output_width; ++ow) {
                            T max_val = get_min_value<T>();
                            size_t max_d = 0, max_h = 0, max_w = 0;
                            bool found_max = false;

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

                                            size_t real_kd = static_cast<size_t>(kd);
                                            size_t real_kh = static_cast<size_t>(kh);
                                            size_t real_kw = static_cast<size_t>(kw);

                                            T val = input[input_offset + real_kd * input_height * input_width + real_kh * input_width + real_kw];

                                            size_t linear_idx = real_kd * input_height * input_width + real_kh * input_width + real_kw;
                                            size_t old_linear_idx = found_max ? max_d * input_height * input_width + max_h * input_width + max_w : SIZE_MAX;

                                            if (!found_max || is_greater(val, max_val) || (values_equal(val, max_val) && linear_idx < old_linear_idx)) {
                                                max_val = val;
                                                max_d = real_kd;
                                                max_h = real_kh;
                                                max_w = real_kw;
                                                found_max = true;
                                            }
                                        }
                                    }
                                }
                            }

                            if (found_max) {
                                size_t grad_input_idx = input_offset + max_d * input_height * input_width + max_h * input_width + max_w;
                                if constexpr (std::is_same<T, bf16_t>::value) {
                                    float current = utils::cast<float>(grad_input[grad_input_idx]);
                                    float to_add = utils::cast<float>(grad_output[output_offset + od * output_height * output_width + oh * output_width + ow]);
                                    grad_input[grad_input_idx] = utils::cast<T>(current + to_add);
                                } else {
                                    grad_input[grad_input_idx] += grad_output[output_offset + od * output_height * output_width + oh * output_width + ow];
                                }
                            }
                        }
                    }
                }
            }
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
           MaxPoolBackwardInfo &info,
           infiniDtype_t data_type) {
        if (data_type != INFINI_DTYPE_F32 && data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *grad_input, const void *grad_output,
                             const void *input, infiniDtype_t dtype) const {

        if (!grad_input || !grad_output || !input) {
            return INFINI_STATUS_BAD_PARAM;
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            float *typed_grad_input = static_cast<float *>(grad_input);
            const float *typed_grad_output = static_cast<const float *>(grad_output);
            const float *typed_input = static_cast<const float *>(input);
            maxpool_backward_cpu(typed_grad_input, typed_grad_output, typed_input);
            break;
        }

        case INFINI_DTYPE_F16: {
            // F16特殊处理：转换为float计算
            fp16_t *typed_grad_input = static_cast<fp16_t *>(grad_input);
            const fp16_t *typed_grad_output = static_cast<const fp16_t *>(grad_output);
            const fp16_t *typed_input = static_cast<const fp16_t *>(input);
            maxpool_backward_f16_as_float(typed_grad_input, typed_grad_output, typed_input);
            break;
        }

        case INFINI_DTYPE_BF16: {
            bf16_t *typed_grad_input = static_cast<bf16_t *>(grad_input);
            const bf16_t *typed_grad_output = static_cast<const bf16_t *>(grad_output);
            const bf16_t *typed_input = static_cast<const bf16_t *>(input);
            maxpool_backward_cpu(typed_grad_input, typed_grad_output, typed_input);
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
                                  infiniopTensorDescriptor_t grad_input_desc,
                                  infiniopTensorDescriptor_t grad_output_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  void *kernel_size, void *strides, void *pads,
                                  bool ceil_mode) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto result = MaxPoolBackwardInfo::create(grad_input_desc, grad_output_desc, input_desc,
                                              kernel_size, strides, pads, ceil_mode);
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
                                     void *grad_input, const void *grad_output,
                                     const void *input, void *stream) const {
    return _opaque->calculate(workspace, workspace_size, grad_input, grad_output, input, _dtype);
}

} // namespace op::maxpool_backward::cpu
