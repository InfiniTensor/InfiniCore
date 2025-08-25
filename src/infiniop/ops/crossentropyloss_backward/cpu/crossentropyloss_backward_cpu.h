#ifndef __CROSSENTROPYLOSS_BACKWARD_CPU_H__
#define __CROSSENTROPYLOSS_BACKWARD_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../../../../utils/custom_types.h"

ELEMENTWISE_DESCRIPTOR(crossentropyloss_backward, cpu)

namespace op::crossentropyloss_backward::cpu {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    size_t batch_size;

    CrossEntropyLossBackwardOp(size_t batch_size = 1) : batch_size(batch_size) {}
    
    template <typename T, typename... Args>
    T operator()(const T &probs, const T &target, Args&&... args) const {
        // According to competition requirements: grad_logits = (probs - target) / N
        // N is the batch size, which is the product of all dimensions except the last one
        return (probs - target) / static_cast<T>(batch_size);
    }

    // Specialization for bf16_t to use double for intermediate calculations
    template <typename... Args>
    bf16_t operator()(const bf16_t &probs, const bf16_t &target, Args&&... args) const {
        double probs_val = _bf16_to_f32(probs);
        double target_val = _bf16_to_f32(target);
        return _f32_to_bf16(static_cast<float>((probs_val - target_val) / static_cast<double>(batch_size)));
    }

    // Specialization for fp16_t to use float for intermediate calculations
    template <typename... Args>
    fp16_t operator()(const fp16_t &probs, const fp16_t &target, Args&&... args) const {
        float probs_val = _f16_to_f32(probs);
        float target_val = _f16_to_f32(target);
        return _f32_to_f16(static_cast<float>((probs_val - target_val) / static_cast<float>(batch_size)));
    }
} CrossEntropyLossBackwardOp;
} // namespace op::crossentropyloss_backward::cpu

#endif // __CROSSENTROPYLOSS_BACKWARD_CPU_H__