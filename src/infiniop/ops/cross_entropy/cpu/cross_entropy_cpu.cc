#include "cross_entropy_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include <cmath>
#include <algorithm>

namespace op::cross_entropy::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 1. Create 函数 (修正：必须写出完整参数列表)
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t target_desc) {
    
    auto x_dtype = x_desc->dtype();
    auto t_dtype = target_desc->dtype();
    
    CHECK_DTYPE(x_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_DTYPE(t_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    // 填充 Info 结构体
    CrossEntropyInfo info{};
    info.dtype = x_dtype;
    info.target_dtype = t_dtype;
    
    // 计算 outer_size (Batch * Seq)
    info.outer_size = target_desc->numel();
    
    // 计算 vocab_size
    info.vocab_size = x_desc->shape().back();
    
    // 计算 stride (假设连续)
    info.x_stride = static_cast<ptrdiff_t>(info.vocab_size);

    *desc_ptr = new Descriptor(nullptr, info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 2. Kernel 函数
// ==================================================================
template <typename T, typename Tidx>
infiniStatus_t cross_entropy_kernel(const CrossEntropyInfo *info,
                                    T *y, const T *x, const void *target) {
    const Tidx *label = reinterpret_cast<const Tidx *>(target);

    #pragma omp parallel for
    for (size_t i = 0; i < info->outer_size; ++i) {
        const T *row = x + i * info->x_stride;
        Tidx idx = label[i];

        // 边界检查
        if (idx < 0 || static_cast<size_t>(idx) >= info->vocab_size) {
            y[i] = utils::cast<T>(0.f);
            continue;
        }

        // 1. Find Max
        float max_val = op::common_cpu::reduce_op::max(row, info->vocab_size, 1);

        // 2. Sum Exp
        float sum_exp = 0.f;
        for (size_t j = 0; j < info->vocab_size; ++j) {
            sum_exp += std::exp(utils::cast<float>(row[j]) - max_val);
        }

        // 3. Compute Loss
        float log_term = std::log(sum_exp) + max_val;
        float target_logit = utils::cast<float>(row[idx]);
        y[i] = utils::cast<T>(log_term - target_logit);
    }
    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 3. Dispatch Helper (修正：直接接收 Info 指针，避开 Private 权限问题)
// ==================================================================
template <typename T>
infiniStatus_t dispatch_target_type(const CrossEntropyInfo *info,
                                    T *y, const T *x, const void *target) {
    // 使用 info->target_dtype 而不是 desc->_info
    if (info->target_dtype == INFINI_DTYPE_I32) {
        return cross_entropy_kernel<T, int32_t>(info, y, x, target);
    } else if (info->target_dtype == INFINI_DTYPE_I64) {
        return cross_entropy_kernel<T, int64_t>(info, y, x, target);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

// ==================================================================
// 4. Calculate 函数 (修正：必须写出完整参数列表)
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *target,
    void *stream) const {

    // 修正：调用 dispatch 时传入 &_info，而不是 this
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return dispatch_target_type(&_info, (fp16_t *)y, (const fp16_t *)x, target);
    case INFINI_DTYPE_BF16:
        return dispatch_target_type(&_info, (bf16_t *)y, (const bf16_t *)x, target);
    case INFINI_DTYPE_F32:
        return dispatch_target_type(&_info, (float *)y, (const float *)x, target);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::cross_entropy::cpu