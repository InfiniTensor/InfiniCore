#include "binary_cross_entropy_with_logits_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <cmath>
#include <algorithm>

namespace op::bce_with_logits::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t pos_weight_desc,
    infiniopReduction_t reduction) {
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = logits_desc->dtype();

    // 1. 类型检查
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    // 2. 解析维度信息 (利用之前定义的 BCEWithLogitsInfo)
    auto result = BCEWithLogitsInfo::create(out_desc, logits_desc, target_desc, 
                                            weight_desc, pos_weight_desc, reduction);
    CHECK_RESULT(result);

    // 3. 实例化描述符
    *desc_ptr = new Descriptor(
        dtype, result.take(), reduction, 0,
        nullptr,
        handle->device, handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

/**
 * 核心数值稳定逻辑：L = -w * [pw * y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
 * 变形为：L = w * [max(x, 0) - x * y * pw + (1 + (pw-1) * y) * log(1 + exp(-|x|))]
 * 当 pw=1 时简化为：L = w * [max(x, 0) - x * y + log(1 + exp(-|x|))]
 */
template <typename Tdata>
void calculate_bce(
    const BCEWithLogitsInfo &info,
    void *out,
    const void *logits,
    const void *target,
    const void *weight,
    const void *pos_weight) {

    size_t n = info.num_elements;
    float total_loss = 0.0f;

    // 获取各张量指针
    const Tdata* l_ptr = reinterpret_cast<const Tdata*>(logits);
    const Tdata* t_ptr = reinterpret_cast<const Tdata*>(target);
    const Tdata* w_ptr = reinterpret_cast<const Tdata*>(weight);
    const Tdata* pw_ptr = reinterpret_cast<const Tdata*>(pos_weight);
    Tdata* o_ptr = reinterpret_cast<Tdata*>(out);

    // 

#pragma omp parallel for reduction(+:total_loss)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i) {
        float x = utils::cast<float>(l_ptr[i]);
        float y = utils::cast<float>(t_ptr[i]);
        
        // 处理 pos_weight 广播 (假设 logits 形状 [..., C], pos_weight 为 [C])
        float pw = pw_ptr ? utils::cast<float>(pw_ptr[i % info.pos_weight.total_elements]) : 1.0f;
        float w = w_ptr ? utils::cast<float>(w_ptr[i]) : 1.0f;

        // 数值稳定的 BCE 计算
        float loss = (std::max(x, 0.0f) - x * y * pw + 
                     (1.0f + (pw - 1.0f) * y) * std::log(1.0f + std::exp(-std::abs(x))));
        
        loss *= w;

        if (info.reduction == INFINIOP_REDUCTION_NONE) {
            o_ptr[i] = utils::cast<Tdata>(loss);
        } else {
            total_loss += loss;
        }
    }

    // 处理归约输出
    if (info.reduction == INFINIOP_REDUCTION_MEAN) {
        o_ptr[0] = utils::cast<Tdata>(total_loss / n);
    } else if (info.reduction == INFINIOP_REDUCTION_SUM) {
        o_ptr[0] = utils::cast<Tdata>(total_loss);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *logits,
    const void *target,
    const void *weight,
    const void *pos_weight,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cpu::calculate_bce<fp16_t>(_info, out, logits, target, weight, pos_weight);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        cpu::calculate_bce<bf16_t>(_info, out, logits, target, weight, pos_weight);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        cpu::calculate_bce<float>(_info, out, logits, target, weight, pos_weight);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::bce_with_logits::cpu