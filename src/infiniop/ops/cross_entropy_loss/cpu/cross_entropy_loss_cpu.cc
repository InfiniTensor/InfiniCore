#include "cross_entropy_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../info.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace op::cross_entropy_loss::cpu {

struct Descriptor::Opaque {
    device::cpu::Handle *handle;
    std::vector<size_t> logits_shape;
    size_t workspace_size = 0;

private:
    Opaque(device::cpu::Handle *handle_ptr, const std::vector<size_t> &shape)
        : handle(handle_ptr), logits_shape(shape) {
        // 计算workspace大小：需要存储per-sample loss
        size_t N = logits_shape[0];
        size_t inner_size = 1;
        for (size_t i = 2; i < logits_shape.size(); ++i) {
            inner_size *= logits_shape[i];
        }
        workspace_size = N * inner_size * sizeof(float);
    }

    void cross_entropy_f16_as_float(float *workspace, float *loss_result,
                                    const fp16_t *logits, const int64_t *target) const {
        size_t N = logits_shape[0];
        size_t C = logits_shape[1];
        size_t inner_size = 1;
        for (size_t i = 2; i < logits_shape.size(); ++i) {
            inner_size *= logits_shape[i];
        }

        // 转换F16 logits为float
        size_t total_logits_size = N * C * inner_size;
        std::vector<float> float_logits(total_logits_size);
        for (size_t i = 0; i < total_logits_size; ++i) {
            float_logits[i] = utils::cast<float>(logits[i]);
        }

        // 使用float精度计算
        cross_entropy_cpu_float(workspace, loss_result, float_logits.data(), target);
    }

    // 通用的float版本交叉熵计算
    void cross_entropy_cpu_float(float *workspace, float *loss_result,
                                 const float *logits, const int64_t *target) const {
        size_t N = logits_shape[0];
        size_t C = logits_shape[1];
        size_t inner_size = 1;
        for (size_t i = 2; i < logits_shape.size(); ++i) {
            inner_size *= logits_shape[i];
        }

        const int64_t ignore_index = -100;
        float *per_sample_loss = workspace;

        // 计算每个样本的损失
        for (size_t n = 0; n < N; ++n) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                size_t sample_idx = n * inner_size + inner;
                int64_t t = target[sample_idx];

                // 检查ignore_index或无效target
                if (t == ignore_index || t < 0 || t >= static_cast<int64_t>(C)) {
                    per_sample_loss[sample_idx] = 0.0f;
                    continue;
                }

                // 计算这个位置的logits基址
                size_t base_offset = n * C * inner_size + inner;

                // 数值稳定的softmax计算：先找最大值
                float max_logit = -std::numeric_limits<float>::infinity();
                for (size_t c = 0; c < C; ++c) {
                    size_t logit_idx = base_offset + c * inner_size;
                    max_logit = std::max(max_logit, logits[logit_idx]);
                }

                // 计算exp的和（减去最大值保证数值稳定）
                float sum_exp = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    size_t logit_idx = base_offset + c * inner_size;
                    sum_exp += std::exp(logits[logit_idx] - max_logit);
                }

                // 计算目标类别的logit
                size_t target_logit_idx = base_offset + static_cast<size_t>(t) * inner_size;
                float target_logit = logits[target_logit_idx];

                // 计算交叉熵损失：log_softmax[target] = logit[target] - log(sum_exp) - max_logit
                // 所以 -log_softmax[target] = log(sum_exp) + max_logit - logit[target]
                per_sample_loss[sample_idx] = std::log(sum_exp) + max_logit - target_logit;
            }
        }

        // 计算平均损失（忽略ignore_index的样本）
        double total_loss = 0.0;
        size_t valid_count = 0;
        size_t total_samples = N * inner_size;

        for (size_t i = 0; i < total_samples; ++i) {
            if (target[i] != ignore_index && target[i] >= 0 && target[i] < static_cast<int64_t>(C)) {
                total_loss += static_cast<double>(per_sample_loss[i]);
                valid_count++;
            }
        }

        *loss_result = valid_count > 0 ? static_cast<float>(total_loss / valid_count) : 0.0f;
    }

    // 通用模板版本（用于F32和BF16）
    template <typename T>
    void cross_entropy_cpu_generic(float *workspace, T *loss_result,
                                   const T *logits, const int64_t *target) const {
        size_t N = logits_shape[0];
        size_t C = logits_shape[1];
        size_t inner_size = 1;
        for (size_t i = 2; i < logits_shape.size(); ++i) {
            inner_size *= logits_shape[i];
        }

        const int64_t ignore_index = -100;
        float *per_sample_loss = workspace;

        // 计算每个样本的损失
        for (size_t n = 0; n < N; ++n) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                size_t sample_idx = n * inner_size + inner;
                int64_t t = target[sample_idx];

                // 检查ignore_index或无效target
                if (t == ignore_index || t < 0 || t >= static_cast<int64_t>(C)) {
                    per_sample_loss[sample_idx] = 0.0f;
                    continue;
                }

                // 计算这个位置的logits基址
                size_t base_offset = n * C * inner_size + inner;

                // 数值稳定的softmax计算：先找最大值
                float max_logit = -std::numeric_limits<float>::infinity();
                for (size_t c = 0; c < C; ++c) {
                    size_t logit_idx = base_offset + c * inner_size;
                    float logit_val;
                    if constexpr (std::is_same<T, bf16_t>::value) {
                        logit_val = utils::cast<float>(logits[logit_idx]);
                    } else {
                        logit_val = logits[logit_idx];
                    }
                    max_logit = std::max(max_logit, logit_val);
                }

                // 计算exp的和
                float sum_exp = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    size_t logit_idx = base_offset + c * inner_size;
                    float logit_val;
                    if constexpr (std::is_same<T, bf16_t>::value) {
                        logit_val = utils::cast<float>(logits[logit_idx]);
                    } else {
                        logit_val = logits[logit_idx];
                    }
                    sum_exp += std::exp(logit_val - max_logit);
                }

                // 计算目标类别的logit
                size_t target_logit_idx = base_offset + static_cast<size_t>(t) * inner_size;
                float target_logit;
                if constexpr (std::is_same<T, bf16_t>::value) {
                    target_logit = utils::cast<float>(logits[target_logit_idx]);
                } else {
                    target_logit = logits[target_logit_idx];
                }

                // 计算交叉熵损失
                per_sample_loss[sample_idx] = std::log(sum_exp) + max_logit - target_logit;
            }
        }

        // 计算平均损失
        double total_loss = 0.0;
        size_t valid_count = 0;
        size_t total_samples = N * inner_size;

        for (size_t i = 0; i < total_samples; ++i) {
            if (target[i] != ignore_index && target[i] >= 0 && target[i] < static_cast<int64_t>(C)) {
                total_loss += static_cast<double>(per_sample_loss[i]);
                valid_count++;
            }
        }

        float mean_loss = valid_count > 0 ? static_cast<float>(total_loss / valid_count) : 0.0f;

        // 转换回输出类型
        if constexpr (std::is_same<T, bf16_t>::value) {
            *loss_result = utils::cast<T>(mean_loss);
        } else {
            *loss_result = static_cast<T>(mean_loss);
        }
    }

public:
    Opaque(Opaque &&other) noexcept
        : handle(other.handle),
          logits_shape(std::move(other.logits_shape)),
          workspace_size(other.workspace_size) {
        other.handle = nullptr;
        other.workspace_size = 0;
    }

    ~Opaque() = default;

    static inline utils::Result<Opaque>
    create(device::cpu::Handle *handle_ptr, const std::vector<size_t> &shape) {
        Opaque opaque(handle_ptr, shape);
        return utils::Result<Opaque>(std::move(opaque));
    }

    infiniStatus_t calculate(void *workspace, size_t workspace_size,
                             void *loss, const void *logits, const void *target,
                             infiniDtype_t dtype) const {
        if (!workspace || !loss || !logits || !target) {
            return INFINI_STATUS_BAD_PARAM;
        }

        if (workspace_size < this->workspace_size) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }

        float *workspace_ptr = static_cast<float *>(workspace);
        const int64_t *target_ptr = static_cast<const int64_t *>(target);

        switch (dtype) {
        case INFINI_DTYPE_F32: {
            const float *logits_ptr = static_cast<const float *>(logits);
            float *loss_ptr = static_cast<float *>(loss);
            cross_entropy_cpu_generic(workspace_ptr, loss_ptr, logits_ptr, target_ptr);
            break;
        }

        case INFINI_DTYPE_F16: {
            const fp16_t *logits_ptr = static_cast<const fp16_t *>(logits);
            fp16_t *loss_ptr = static_cast<fp16_t *>(loss);

            // F16特殊处理：使用float计算
            float temp_loss;
            cross_entropy_f16_as_float(workspace_ptr, &temp_loss, logits_ptr, target_ptr);
            *loss_ptr = utils::cast<fp16_t>(temp_loss);
            break;
        }

        case INFINI_DTYPE_BF16: {
            const bf16_t *logits_ptr = static_cast<const bf16_t *>(logits);
            bf16_t *loss_ptr = static_cast<bf16_t *>(loss);
            cross_entropy_cpu_generic(workspace_ptr, loss_ptr, logits_ptr, target_ptr);
            break;
        }

        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return INFINI_STATUS_SUCCESS;
    }

    size_t get_workspace_size() const {
        return workspace_size;
    }
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t /*loss_desc*/,
                                  infiniopTensorDescriptor_t logits_desc,
                                  infiniopTensorDescriptor_t /*target_desc*/) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = logits_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    const auto &orig_shape = logits_desc->shape();
    std::vector<size_t> logits_shape;

    if (orig_shape.size() == 1) {
        logits_shape = {1, orig_shape[0]};
    } else {
        logits_shape = orig_shape;
    }

    if (logits_shape.size() < 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto opaque_result = Opaque::create(handle, logits_shape);
    CHECK_RESULT(opaque_result);
    auto opaque = new Opaque(opaque_result.take());

    *desc_ptr = new Descriptor(dtype, opaque->get_workspace_size(), opaque,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *loss, const void *logits,
                                     const void *target, void *stream) const {
    return _opaque->calculate(workspace, workspace_size, loss, logits, target, _dtype);
}

} // namespace op::cross_entropy_loss::cpu
