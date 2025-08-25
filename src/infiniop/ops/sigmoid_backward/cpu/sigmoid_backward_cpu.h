#ifndef __SIGMOID_BACKWARD_CPU_H__
#define __SIGMOID_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sigmoid_backward, cpu)

namespace op::sigmoid_backward::cpu {
typedef struct SigmoidBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &x, const T &grad_out) const {
        using ComputeT = std::conditional_t<std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>,
                                            float, T>;
        ComputeT xv = utils::cast<ComputeT, T>(x);
        ComputeT gov = utils::cast<ComputeT, T>(grad_out);

        // sigmoid(x) = 1 / (1 + exp(-x))
        ComputeT s = static_cast<ComputeT>(1) / (static_cast<ComputeT>(1) + std::exp(-xv));

        // grad_input = grad_output * s * (1 - s)
        ComputeT gin = gov * s * (static_cast<ComputeT>(1) - s);

        return utils::cast<T, ComputeT>(gin);
    }
} SigmoidBackwardOp;
} // namespace op::sigmoid_backward::cpu

#endif // __SIGMOID_BACKWARD_CPU_H__
