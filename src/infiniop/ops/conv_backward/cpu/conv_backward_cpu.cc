#include "conv_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace op::conv_backward::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    op::conv_backward::ConvBackwardInfo info;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr,
           const op::conv_backward::ConvBackwardInfo &conv_info)
        : handle(handle_ptr), info(conv_info) {
        workspace_size = 0;
    }

    // 递归函数：计算数据梯度的N维卷积反向传播
    template <typename GradOutData, typename WeightData, typename GradInData>
    void _applyDataGradient(
        size_t grad_out_index, size_t weight_index, size_t grad_in_index,
        size_t ndim, const GradOutData *grad_output, const WeightData *weight,
        GradInData *grad_input, const size_t *grad_in_shape) const {

        if (ndim >= info.ndim + 2) {
            // 到达最深层，执行实际计算
            // 始终使用float精度进行计算，避免半精度累积误差
            float grad_out_f32 = utils::cast<float>(grad_output[grad_out_index]);
            float weight_f32 = utils::cast<float>(weight[weight_index]);
            float current_grad_in = utils::cast<float>(grad_input[grad_in_index]);
            float result = current_grad_in + grad_out_f32 * weight_f32;
            grad_input[grad_in_index] = utils::cast<GradInData>(result);
            return;
        }

        size_t dim_idx = ndim - 2;
        size_t grad_out_dim = info.grad_output_dims[dim_idx];
        size_t weight_dim = info.weight_dims[dim_idx];
        size_t grad_in_dim = grad_in_shape[ndim];
        size_t stride = info.strides[dim_idx];
        size_t pad = info.pads[dim_idx];
        size_t dilation = info.dilations[dim_idx];

        // 遍历输出维度
        for (size_t oh = 0; oh < grad_out_dim; ++oh) {
            size_t curr_grad_out_index = grad_out_index * grad_out_dim + oh;

            // 遍历卷积核维度
            for (size_t kh = 0; kh < weight_dim; ++kh) {
                size_t curr_weight_index = weight_index * weight_dim + kh;

                // 计算对应的输入位置
                int ih = static_cast<int>(oh * stride + kh * dilation) - static_cast<int>(pad);

                if (ih >= 0 && ih < static_cast<int>(grad_in_dim)) {
                    size_t curr_grad_in_index = grad_in_index * grad_in_dim + ih;

                    _applyDataGradient(curr_grad_out_index, curr_weight_index, curr_grad_in_index,
                                       ndim + 1, grad_output, weight, grad_input, grad_in_shape);
                }
            }
        }
    }

    // 递归函数：计算权重梯度的N维卷积反向传播
    template <typename InputData, typename GradOutData, typename GradWeightData>
    void _applyWeightGradient(
        size_t input_index, size_t grad_out_index, size_t grad_weight_index,
        size_t ndim, const InputData *input, const GradOutData *grad_output,
        GradWeightData *grad_weight, const size_t *input_shape) const {

        if (ndim >= info.ndim + 2) {
            // 到达最深层，执行实际计算
            // 始终使用float精度进行计算，避免半精度累积误差
            float input_f32 = utils::cast<float>(input[input_index]);
            float grad_out_f32 = utils::cast<float>(grad_output[grad_out_index]);
            float current_grad_weight = utils::cast<float>(grad_weight[grad_weight_index]);
            float result = current_grad_weight + input_f32 * grad_out_f32;
            grad_weight[grad_weight_index] = utils::cast<GradWeightData>(result);
            return;
        }

        size_t dim_idx = ndim - 2;
        size_t input_dim = input_shape[ndim];
        size_t grad_out_dim = info.grad_output_dims[dim_idx];
        size_t weight_dim = info.weight_dims[dim_idx];
        size_t stride = info.strides[dim_idx];
        size_t pad = info.pads[dim_idx];
        size_t dilation = info.dilations[dim_idx];

        // 遍历卷积核维度
        for (size_t kh = 0; kh < weight_dim; ++kh) {
            size_t curr_grad_weight_index = grad_weight_index * weight_dim + kh;

            // 遍历输出维度
            for (size_t oh = 0; oh < grad_out_dim; ++oh) {
                size_t curr_grad_out_index = grad_out_index * grad_out_dim + oh;

                // 计算对应的输入位置
                int ih = static_cast<int>(oh * stride + kh * dilation) - static_cast<int>(pad);

                if (ih >= 0 && ih < static_cast<int>(input_dim)) {
                    size_t curr_input_index = input_index * input_dim + ih;

                    _applyWeightGradient(curr_input_index, curr_grad_out_index, curr_grad_weight_index,
                                         ndim + 1, input, grad_output, grad_weight, input_shape);
                }
            }
        }
    }

    // 获取零值
    template <typename T>
    static T get_zero() {
        if constexpr (std::is_same<T, float>::value) {
            return 0.0f;
        } else if constexpr (std::is_same<T, fp16_t>::value) {
            return _f32_to_f16(0.0f);
        } else if constexpr (std::is_same<T, bf16_t>::value) {
            return _f32_to_bf16(0.0f);
        } else {
            return T{};
        }
    }

    // 计算数据梯度 (grad_input) - 使用更直接的实现避免递归
    template <typename GradOutData, typename WeightData, typename GradInData>
    void compute_data_gradient(GradInData *grad_input, const GradOutData *grad_output,
                               const WeightData *weight) const {

        size_t batch_size = info.batch;
        size_t in_channels = info.in_channels;
        size_t out_channels = info.out_channels;
        size_t groups = info.groups;
        size_t channels_per_group = in_channels / groups;
        size_t out_channels_per_group = out_channels / groups;

        // 计算空间大小
        size_t input_spatial_size = 1;
        size_t output_spatial_size = 1;
        for (size_t i = 0; i < info.ndim; ++i) {
            input_spatial_size *= info.input_dims[i];
            output_spatial_size *= info.grad_output_dims[i];
        }

        // 初始化为零
        size_t total_grad_input_size = batch_size * in_channels * input_spatial_size;
        GradInData zero_val = get_zero<GradInData>();
        std::fill(grad_input, grad_input + total_grad_input_size, zero_val);

        // 对每个批次和组并行处理
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t g = 0; g < groups; ++g) {
                // 对每个输出通道
                for (size_t oc = 0; oc < out_channels_per_group; ++oc) {
                    size_t abs_oc = g * out_channels_per_group + oc;

                    // 对每个输入通道
                    for (size_t ic = 0; ic < channels_per_group; ++ic) {
                        size_t abs_ic = g * channels_per_group + ic;

                        // 对每个输出空间位置
                        for (size_t out_spatial = 0; out_spatial < output_spatial_size; ++out_spatial) {

                            // 将一维空间索引转换为多维坐标
                            std::vector<size_t> out_coords(info.ndim);
                            size_t temp = out_spatial;
                            for (int d = info.ndim - 1; d >= 0; --d) {
                                out_coords[d] = temp % info.grad_output_dims[d];
                                temp /= info.grad_output_dims[d];
                            }

                            // 对每个卷积核空间位置
                            size_t kernel_spatial_size = 1;
                            for (size_t i = 0; i < info.ndim; ++i) {
                                kernel_spatial_size *= info.weight_dims[i];
                            }

                            for (size_t kernel_spatial = 0; kernel_spatial < kernel_spatial_size; ++kernel_spatial) {

                                // 将一维卷积核索引转换为多维坐标
                                std::vector<size_t> kernel_coords(info.ndim);
                                temp = kernel_spatial;
                                for (int d = info.ndim - 1; d >= 0; --d) {
                                    kernel_coords[d] = temp % info.weight_dims[d];
                                    temp /= info.weight_dims[d];
                                }

                                // 计算对应的输入坐标
                                std::vector<int> input_coords(info.ndim);
                                bool valid = true;

                                for (size_t d = 0; d < info.ndim; ++d) {
                                    input_coords[d] = static_cast<int>(out_coords[d] * info.strides[d] + kernel_coords[d] * info.dilations[d]) - static_cast<int>(info.pads[d]);

                                    if (input_coords[d] < 0 || input_coords[d] >= static_cast<int>(info.input_dims[d])) {
                                        valid = false;
                                        break;
                                    }
                                }

                                if (valid) {
                                    // 计算线性索引
                                    size_t grad_out_idx = b * out_channels * output_spatial_size + abs_oc * output_spatial_size + out_spatial;

                                    size_t weight_idx = abs_oc * channels_per_group * kernel_spatial_size + ic * kernel_spatial_size + kernel_spatial;

                                    size_t input_spatial_idx = 0;
                                    size_t multiplier = 1;
                                    for (int d = info.ndim - 1; d >= 0; --d) {
                                        input_spatial_idx += input_coords[d] * multiplier;
                                        multiplier *= info.input_dims[d];
                                    }

                                    size_t grad_in_idx = b * in_channels * input_spatial_size + abs_ic * input_spatial_size + input_spatial_idx;

                                    // 执行计算
                                    float grad_out_f32 = utils::cast<float>(grad_output[grad_out_idx]);
                                    float weight_f32 = utils::cast<float>(weight[weight_idx]);
                                    float current_grad_in = utils::cast<float>(grad_input[grad_in_idx]);
                                    float result = current_grad_in + grad_out_f32 * weight_f32;
                                    grad_input[grad_in_idx] = utils::cast<GradInData>(result);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 计算权重梯度 (grad_weight) - 使用更直接的实现
    template <typename InputData, typename GradOutData, typename GradWeightData>
    void compute_weight_gradient(GradWeightData *grad_weight, const GradOutData *grad_output,
                                 const InputData *input) const {

        size_t batch_size = info.batch;
        size_t in_channels = info.in_channels;
        size_t out_channels = info.out_channels;
        size_t groups = info.groups;
        size_t channels_per_group = in_channels / groups;
        size_t out_channels_per_group = out_channels / groups;

        // 计算空间大小
        size_t input_spatial_size = 1;
        size_t output_spatial_size = 1;
        size_t kernel_spatial_size = 1;
        for (size_t i = 0; i < info.ndim; ++i) {
            input_spatial_size *= info.input_dims[i];
            output_spatial_size *= info.grad_output_dims[i];
            kernel_spatial_size *= info.weight_dims[i];
        }

        // 初始化为零
        size_t total_weight_size = out_channels * channels_per_group * kernel_spatial_size;
        GradWeightData zero_val = get_zero<GradWeightData>();
        std::fill(grad_weight, grad_weight + total_weight_size, zero_val);

        // 对每个权重元素并行处理
#pragma omp parallel for collapse(3) schedule(dynamic)
        for (size_t abs_oc = 0; abs_oc < out_channels; ++abs_oc) {
            for (size_t ic = 0; ic < channels_per_group; ++ic) {
                for (size_t kernel_spatial = 0; kernel_spatial < kernel_spatial_size; ++kernel_spatial) {

                    size_t g = abs_oc / out_channels_per_group;
                    size_t abs_ic = g * channels_per_group + ic;

                    // 将一维卷积核索引转换为多维坐标
                    std::vector<size_t> kernel_coords(info.ndim);
                    size_t temp = kernel_spatial;
                    for (int d = info.ndim - 1; d >= 0; --d) {
                        kernel_coords[d] = temp % info.weight_dims[d];
                        temp /= info.weight_dims[d];
                    }

                    float accumulator = 0.0f;

                    // 对所有批次和输出位置累积梯度
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t out_spatial = 0; out_spatial < output_spatial_size; ++out_spatial) {

                            // 将一维输出空间索引转换为多维坐标
                            std::vector<size_t> out_coords(info.ndim);
                            temp = out_spatial;
                            for (int d = info.ndim - 1; d >= 0; --d) {
                                out_coords[d] = temp % info.grad_output_dims[d];
                                temp /= info.grad_output_dims[d];
                            }

                            // 计算对应的输入坐标
                            std::vector<int> input_coords(info.ndim);
                            bool valid = true;

                            for (size_t d = 0; d < info.ndim; ++d) {
                                input_coords[d] = static_cast<int>(out_coords[d] * info.strides[d] + kernel_coords[d] * info.dilations[d]) - static_cast<int>(info.pads[d]);

                                if (input_coords[d] < 0 || input_coords[d] >= static_cast<int>(info.input_dims[d])) {
                                    valid = false;
                                    break;
                                }
                            }

                            if (valid) {
                                // 计算线性索引
                                size_t grad_out_idx = b * out_channels * output_spatial_size + abs_oc * output_spatial_size + out_spatial;

                                size_t input_spatial_idx = 0;
                                size_t multiplier = 1;
                                for (int d = info.ndim - 1; d >= 0; --d) {
                                    input_spatial_idx += input_coords[d] * multiplier;
                                    multiplier *= info.input_dims[d];
                                }

                                size_t input_idx = b * in_channels * input_spatial_size + abs_ic * input_spatial_size + input_spatial_idx;

                                // 累积梯度
                                float input_f32 = utils::cast<float>(input[input_idx]);
                                float grad_out_f32 = utils::cast<float>(grad_output[grad_out_idx]);
                                accumulator += input_f32 * grad_out_f32;
                            }
                        }
                    }

                    // 写入结果
                    size_t weight_idx = abs_oc * channels_per_group * kernel_spatial_size + ic * kernel_spatial_size + kernel_spatial;
                    grad_weight[weight_idx] = utils::cast<GradWeightData>(accumulator);
                }
            }
        }
    }

    // 计算偏置梯度 (grad_bias)
    template <typename GradOutData, typename GradBiasData>
    void compute_bias_gradient(GradBiasData *grad_bias, const GradOutData *grad_output) const {
        size_t batch_size = info.batch;
        size_t out_channels = info.out_channels;

        size_t output_spatial_size = 1;
        for (size_t i = 0; i < info.ndim; ++i) {
            output_spatial_size *= info.grad_output_dims[i];
        }

        // 并行处理每个输出通道
#pragma omp parallel for
        for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(out_channels); ++c) {
            float sum = 0.0f;

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < output_spatial_size; ++s) {
                    size_t idx = b * out_channels * output_spatial_size + c * output_spatial_size + s;
                    sum += utils::cast<float>(grad_output[idx]);
                }
            }

            grad_bias[c] = utils::cast<GradBiasData>(sum);
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
           const op::conv_backward::ConvBackwardInfo &info,
           infiniDtype_t data_type) {
        if (data_type != INFINI_DTYPE_F32 && data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        Opaque opaque(handle_ptr, info);
        return utils::Result<Opaque>(std::move(opaque));
    }

    // CPU 实现的卷积反向传播
    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *grad_input, void *grad_weight, void *grad_bias,
                             const void *grad_output, const void *input,
                             const void *weight, infiniDtype_t dtype) const {

        if (!grad_output || !input || !weight) {
            return INFINI_STATUS_BAD_PARAM;
        }

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            const float *grad_output_f32 = static_cast<const float *>(grad_output);
            const float *input_f32 = static_cast<const float *>(input);
            const float *weight_f32 = static_cast<const float *>(weight);

            if (grad_input) {
                float *grad_input_f32 = static_cast<float *>(grad_input);
                compute_data_gradient(grad_input_f32, grad_output_f32, weight_f32);
            }

            if (grad_weight) {
                float *grad_weight_f32 = static_cast<float *>(grad_weight);
                compute_weight_gradient(grad_weight_f32, grad_output_f32, input_f32);
            }

            if (grad_bias) {
                float *grad_bias_f32 = static_cast<float *>(grad_bias);
                compute_bias_gradient(grad_bias_f32, grad_output_f32);
            }
            break;
        }

        case INFINI_DTYPE_F16: {
            const fp16_t *grad_output_f16 = static_cast<const fp16_t *>(grad_output);
            const fp16_t *input_f16 = static_cast<const fp16_t *>(input);
            const fp16_t *weight_f16 = static_cast<const fp16_t *>(weight);

            if (grad_input) {
                fp16_t *grad_input_f16 = static_cast<fp16_t *>(grad_input);
                compute_data_gradient(grad_input_f16, grad_output_f16, weight_f16);
            }

            if (grad_weight) {
                fp16_t *grad_weight_f16 = static_cast<fp16_t *>(grad_weight);
                compute_weight_gradient(grad_weight_f16, grad_output_f16, input_f16);
            }

            if (grad_bias) {
                fp16_t *grad_bias_f16 = static_cast<fp16_t *>(grad_bias);
                compute_bias_gradient(grad_bias_f16, grad_output_f16);
            }
            break;
        }

        case INFINI_DTYPE_BF16: {
            const bf16_t *grad_output_bf16 = static_cast<const bf16_t *>(grad_output);
            const bf16_t *input_bf16 = static_cast<const bf16_t *>(input);
            const bf16_t *weight_bf16 = static_cast<const bf16_t *>(weight);

            if (grad_input) {
                bf16_t *grad_input_bf16 = static_cast<bf16_t *>(grad_input);
                compute_data_gradient(grad_input_bf16, grad_output_bf16, weight_bf16);
            }

            if (grad_weight) {
                bf16_t *grad_weight_bf16 = static_cast<bf16_t *>(grad_weight);
                compute_weight_gradient(grad_weight_bf16, grad_output_bf16, input_bf16);
            }

            if (grad_bias) {
                bf16_t *grad_bias_bf16 = static_cast<bf16_t *>(grad_bias);
                compute_bias_gradient(grad_bias_bf16, grad_output_bf16);
            }
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
                                  infiniopTensorDescriptor_t grad_output_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t weight_desc,
                                  infiniopTensorDescriptor_t bias_desc,
                                  void *pads, void *strides, void *dilations,
                                  size_t groups) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    auto info_result = op::conv_backward::ConvBackwardInfo::create(
        grad_output_desc, input_desc, weight_desc, pads, strides, dilations, groups);
    CHECK_RESULT(info_result);
    auto info = info_result.take();

    auto opaque_result = Opaque::create(handle, info, dtype);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, opaque->workspace_size, opaque,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *grad_input, void *grad_weight,
                                     void *grad_bias, const void *grad_output,
                                     const void *input, const void *weight,
                                     void *stream) const {
    return _opaque->calculate(workspace, workspace_size, grad_input, grad_weight,
                              grad_bias, grad_output, input, weight, _dtype);
}

} // namespace op::conv_backward::cpu
