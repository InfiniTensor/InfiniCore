#ifndef __FMIN_CPU_H__
#define __FMIN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(fmin, cpu)

namespace op::fmin::cpu {
typedef struct FminOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            float a_f = utils::cast<float>(a);
            float b_f = utils::cast<float>(b);
            float result = std::fminf(a_f, b_f);
            return utils::cast<T>(result);
        } else {
            return std::fmin(a, b);
        }
    }
} FminOp;
} // namespace op::fmin::cpu

#endif // __FMIN_CPU_H__
