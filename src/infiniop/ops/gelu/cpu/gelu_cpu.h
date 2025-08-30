#ifndef __GELU_CPU_H__
#define __GELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
ELEMENTWISE_DESCRIPTOR(gelu, cpu)

namespace op::gelu::cpu {
typedef struct GeLUOp {
public:
    static constexpr size_t num_inputs = 1;
    const float sqrt_2_over_pi = std::sqrt(2 / 3.14159f);
    template <typename T>
    T operator()(const T &x) const {
        float x_f = utils::cast<float>(x);
        return 0.5 * x_f * (1 + std::tanh(sqrt_2_over_pi * (x_f + 0.044715f * x_f * x_f * x_f)));
    }
} GeLUOp;
} // namespace op::gelu::cpu


#endif // __GELU_CPU_H__
