#ifndef __RELU_BACKWARD_CPU_H__
#define __RELU_BACKWARD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
ELEMENTWISE_DESCRIPTOR(relu_backward, cpu)

namespace op::relu_backward::cpu {
typedef struct ReLUBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T& x, const T& grad_y) const {
        return (utils::cast<float>(x) > 0) ? grad_y : 0;
    }
} ReLUBackwardOp;
} // namespace op::relu_backward::cpu


#endif // __RELU_BACKWARD_CPU_H__
